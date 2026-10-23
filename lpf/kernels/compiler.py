"""
CUDA kernels of the reaction-diffusion models for CuPy: from the AOT fatbin
when it is installed, otherwise compiled at run time with NVRTC (JIT).

Both come from one C++ source, csrc/include/rd_kernels.cuh: fused kernels in
which a space (grid, domain or mesh) evaluates dy/dt and an epilogue writes
the solver's update (an Euler step, dy/dt, a Runge-Kutta stage, ...).
kernels.cu instantiates them for every model and dtype into the fatbin; the
JIT path compiles the same extern "C" entry points for one model and one
space. Kernel names are

    lpf_<epilogue>_<space>_<Model>_<dtype>, e.g. lpf_stage_mesh_Liaw_f32,

so AOT and JIT run identical code. lpf.kernels.operators launches them.
"""

import hashlib
import os.path as osp

import numpy as np

from lpf.kernels.aot.loader import _MODEL_SHORT, _DTYPE_SUFFIX, TILE_X, TILE_Y

_INCLUDE_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "csrc", "include")

EPILOGUES = ("euler", "rhs", "stage", "heun", "rk23", "rk4", "ab2")
SPACES = ("grid", "domain", "mesh")

_CTYPE = {
    np.dtype(np.float16): "__half",
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
}

# NVRTC options. --fmad=false keeps the bit-exact arithmetic of the
# reference path (see csrc/kernels.cu); fast math is never used.
_OPTIONS = ("--fmad=false", "-std=c++17", "-I" + _INCLUDE_DIR)

# Module-level cache: (model_name, dtype_str, fast_math) -> CuKernelManager
_KERNEL_CACHE = {}


def get_kernel_manager(model_name, dtype=np.float32, fast_math=False):
    """Get or create a cached CuKernelManager for the given configuration."""
    key = (model_name, np.dtype(dtype).str, fast_math)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = CuKernelManager(model_name, dtype, fast_math)
    return _KERNEL_CACHE[key]


def _headers_digest():
    """Digest of the kernel headers: CuPy's compile cache is keyed by the
    source text, which only #includes them."""
    h = hashlib.sha1()
    for name in ("types.cuh", "reactions.cuh", "spaces.cuh", "epilogues.cuh",
                 "rd_kernels.cuh"):
        with open(osp.join(_INCLUDE_DIR, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


class CuKernelManager:
    """Kernels of one model and dtype, AOT when available, else JIT.

    Parameters
    ----------
    model_name : str
        Reaction-diffusion model name, e.g. "LiawModel".
    dtype : numpy dtype
        Floating-point precision (float16, float32, float64).
    fast_math : bool
        Kept for API compatibility; never applied (it breaks IEEE-754
        reproducibility).
    """

    def __init__(self, model_name, dtype=np.float32, fast_math=False):
        import cupy as cp
        self._cp = cp

        if model_name not in _MODEL_SHORT:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Supported: {list(_MODEL_SHORT.keys())}")

        self._model_name = model_name
        self._dtype = np.dtype(dtype)
        if self._dtype not in _CTYPE:
            raise ValueError(f"Unsupported dtype for CUDA kernels: {self._dtype}")
        self._fast_math = fast_math
        self._kernels = {}      # (name, using AOT) -> kernel function
        self._jit_modules = {}  # space -> NVRTC module

        # Policy: if no fatbin is present (e.g. source/editable install without
        # an AOT build), silently use JIT — that is a legitimate path. But if
        # a fatbin *is* present and we fail to load or validate it, raise:
        # silent JIT fallback would mask a corrupted/mismatched wheel.
        self._aot = None
        self._use_aot = False
        from lpf.kernels.aot.loader import AOTKernelLoader
        loader = AOTKernelLoader()
        if loader.is_available():
            loader.get_function(self.kernel_name("euler", "grid"))
            self._aot = loader
            self._use_aot = True

    @property
    def model_name(self):
        return self._model_name

    @property
    def dtype(self):
        return self._dtype

    @property
    def using_aot(self):
        return self._use_aot

    def kernel_name(self, epilogue, space):
        return "lpf_%s_%s_%s_%s" % (epilogue, space,
                                    _MODEL_SHORT[self._model_name],
                                    _DTYPE_SUFFIX[self._dtype])

    def kernel(self, epilogue, space):
        """The fused kernel lpf_<epilogue>_<space>_<Model>_<dtype>."""
        name = self.kernel_name(epilogue, space)
        key = (name, self._use_aot)
        if key not in self._kernels:
            if self._use_aot:
                self._kernels[key] = self._aot.get_function(name)
            else:
                self._kernels[key] = self._jit_module(space).get_function(name)
        return self._kernels[key]

    def _jit_module(self, space):
        """The entry points of one space for this model, compiled with NVRTC."""
        if space not in self._jit_modules:
            source = ("// lpf kernel headers %s\n"
                      '#include "rd_kernels.cuh"\n'
                      "LPF_SPACE_ENTRY_POINTS(%s, %s, %s, %s)\n"
                      % (_headers_digest(), space,
                         _MODEL_SHORT[self._model_name], _CTYPE[self._dtype],
                         _DTYPE_SUFFIX[self._dtype]))
            self._jit_modules[space] = self._cp.RawModule(
                code=source, options=_OPTIONS)
        return self._jit_modules[space]

    # ================================================================
    # Full-grid launches (the 0.2 API; see lpf.kernels.operators for the
    # solvers' launches on any geometry)
    # ================================================================

    def launch_euler(self, y_in, y_out, params, B, H, W, dt, dx2_inv):
        """Fused Euler step on the full grid: y_out = y_in + dt * dy/dt."""
        self._launch_grid("euler", B, H, W, dx2_inv,
                          (y_in, params, y_out, np.float64(dt)))

    def launch_pdefunc(self, y_in, dydt, params, B, H, W, dx2_inv):
        """dy/dt on the full grid."""
        self._launch_grid("rhs", B, H, W, dx2_inv, (y_in, params, dydt))

    def _launch_grid(self, epilogue, B, H, W, dx2_inv, args):
        grid = ((W + TILE_X - 1) // TILE_X, (H + TILE_Y - 1) // TILE_Y, B)
        self.kernel(epilogue, "grid")(grid, (TILE_X, TILE_Y), (
            np.int32(B), np.int32(H), np.int32(W), np.float64(dx2_inv), *args))
