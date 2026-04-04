"""
CUDA kernel compilation and management using CuPy RawKernel (JIT)
with optional AOT fatbin acceleration.

CuKernelManager provides unified launch methods that transparently
dispatch to AOT (shared-memory tiled, vectorised) or JIT (1-D grid)
kernels.  AOT is preferred when the pre-compiled fatbin exists;
otherwise NVRTC JIT compilation is used as fallback.
"""

import numpy as np

from lpf.kernels.reactions import REACTION_REGISTRY
from lpf.kernels.sources import (
    FUSED_EULER_STEP_TEMPLATE,
    FUSED_PDEFUNC_TEMPLATE,
    RK_STAGE_UPDATE_SOURCE,
    RK4_COMBINE_SOURCE,
    HEUN_COMBINE_SOURCE,
    SCALE_SOURCE,
    LINEAR_COMBINE2_SOURCE,
    LINEAR_COMBINE3_SOURCE,
)

# Module-level cache: (model_name, dtype_str, fast_math) -> CuKernelManager
_KERNEL_CACHE = {}


def get_kernel_manager(model_name, dtype=np.float32, fast_math=False):
    """Get or create a cached CuKernelManager for the given configuration."""
    key = (model_name, np.dtype(dtype).str, fast_math)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = CuKernelManager(model_name, dtype, fast_math)
    return _KERNEL_CACHE[key]


class CuKernelManager:
    """Unified kernel manager with AOT / JIT fallback.

    Parameters
    ----------
    model_name : str
        Reaction-diffusion model name (key in REACTION_REGISTRY).
    dtype : numpy dtype
        Floating-point precision (float16, float32, float64).
    fast_math : bool
        Use --use_fast_math for JIT compilation.
    """

    def __init__(self, model_name, dtype=np.float32, fast_math=False):
        import cupy as cp
        self._cp = cp

        if model_name not in REACTION_REGISTRY:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Supported: {list(REACTION_REGISTRY.keys())}"
            )

        self._model_name = model_name
        self._dtype = np.dtype(dtype)
        self._fast_math = fast_math
        self._reaction_info = REACTION_REGISTRY[model_name]

        # NVRTC compile options
        self._options = ()
        if fast_math:
            self._options += ('--use_fast_math',)

        # Lazily compiled JIT kernels
        self._jit_euler = None
        self._jit_pdefunc = None
        self._jit_stage = None
        self._jit_rk4_combine = None
        self._jit_heun_combine = None

        # Try to load AOT fatbin
        self._aot = None
        self._use_aot = False
        try:
            from lpf.csrc.loader import AOTKernelLoader
            loader = AOTKernelLoader()
            if loader.is_available():
                # Validate that the required kernels exist
                loader.get_euler_kernel(model_name, dtype)
                self._aot = loader
                self._use_aot = True
        except Exception:
            pass

    # ================================================================
    # Properties
    # ================================================================

    @property
    def n_params(self):
        return self._reaction_info["n_params"]

    @property
    def scalar_type(self):
        if self._dtype == np.float16:
            return np.float32
        elif self._dtype == np.float32:
            return np.float32
        else:
            return np.float64

    @property
    def using_aot(self):
        return self._use_aot

    # ================================================================
    # Unified launch methods
    # ================================================================

    def launch_euler(self, y_in, y_out, params, B, H, W, dt, dx2_inv):
        """Launch fused Euler step kernel."""
        if self._use_aot:
            kernel = self._aot.get_euler_kernel(self._model_name, self._dtype)
            grid, block = self._aot.stencil_grid_block(B, H, W)
            kernel(grid, block, (
                y_in, y_out, params,
                np.int32(B), np.int32(H), np.int32(W),
                np.float64(dt), np.float64(dx2_inv),
            ))
        else:
            kernel = self._get_jit_euler()
            total = B * H * W
            block = 256
            grid = ((total + block - 1) // block,)
            scalar = self.scalar_type
            kernel(grid, (block,), (
                y_in, y_out, params,
                np.int32(B), np.int32(H), np.int32(W),
                scalar(dt), scalar(dx2_inv),
            ))

    def launch_pdefunc(self, y_in, dydt, params, B, H, W, dx2_inv):
        """Launch fused PDE function kernel."""
        if self._use_aot:
            kernel = self._aot.get_pdefunc_kernel(self._model_name, self._dtype)
            grid, block = self._aot.stencil_grid_block(B, H, W)
            kernel(grid, block, (
                y_in, dydt, params,
                np.int32(B), np.int32(H), np.int32(W),
                np.float64(dx2_inv),
            ))
        else:
            kernel = self._get_jit_pdefunc()
            total = B * H * W
            block = 256
            grid = ((total + block - 1) // block,)
            scalar = self.scalar_type
            kernel(grid, (block,), (
                y_in, dydt, params,
                np.int32(B), np.int32(H), np.int32(W),
                scalar(dx2_inv),
            ))

    def launch_rk_stage(self, y, k, y_out, alpha, N):
        """Launch RK stage update: y_out = y + alpha * k."""
        if self._use_aot:
            kernel = self._aot.get_rk_stage_kernel(self._dtype)
            grid, block = self._aot.util_grid_block(N, self._dtype)
            kernel(grid, block, (
                y, k, y_out, np.float64(alpha), np.int32(N),
            ))
        else:
            kernel = self._get_jit_stage()
            block = 256
            grid = ((N + block - 1) // block,)
            scalar = self.scalar_type
            kernel(grid, (block,), (
                y, k, y_out, scalar(alpha), np.int32(N),
            ))

    def launch_rk4_combine(self, delta, k1, k2, k3, k4, dt_over_6, N):
        """Launch RK4 combine: delta = (dt/6)*(k1 + 2*k2 + 2*k3 + k4)."""
        if self._use_aot:
            kernel = self._aot.get_rk4_combine_kernel(self._dtype)
            grid, block = self._aot.util_grid_block(N, self._dtype)
            kernel(grid, block, (
                delta, k1, k2, k3, k4,
                np.float64(dt_over_6), np.int32(N),
            ))
        else:
            kernel = self._get_jit_rk4_combine()
            block = 256
            grid = ((N + block - 1) // block,)
            scalar = self.scalar_type
            kernel(grid, (block,), (
                delta, k1, k2, k3, k4,
                scalar(dt_over_6), np.int32(N),
            ))

    def launch_heun_combine(self, delta, k1, k2, N):
        """Launch Heun combine: delta = 0.5 * (k1 + k2)."""
        if self._use_aot:
            kernel = self._aot.get_heun_combine_kernel(self._dtype)
            grid, block = self._aot.util_grid_block(N, self._dtype)
            kernel(grid, block, (delta, k1, k2, np.int32(N)))
        else:
            kernel = self._get_jit_heun_combine()
            block = 256
            grid = ((N + block - 1) // block,)
            kernel(grid, (block,), (delta, k1, k2, np.int32(N)))

    def launch_scale(self, x, out, alpha, N):
        """Launch scale: out = alpha * x."""
        kernel = self._get_jit_scale()
        scalar = self.scalar_type
        block = 256
        grid = ((N + block - 1) // block,)
        kernel(grid, (block,), (x, out, scalar(alpha), np.int32(N)))

    def launch_linear_combine2(self, x, y, out, a, b, N):
        """Launch linear combine: out = a*x + b*y."""
        kernel = self._get_jit_lc2()
        scalar = self.scalar_type
        block = 256
        grid = ((N + block - 1) // block,)
        kernel(grid, (block,), (x, y, out, scalar(a), scalar(b), np.int32(N)))

    def launch_linear_combine3(self, x, y, z, out, a, b, c, N):
        """Launch linear combine: out = a*x + b*y + c*z."""
        kernel = self._get_jit_lc3()
        scalar = self.scalar_type
        block = 256
        grid = ((N + block - 1) // block,)
        kernel(grid, (block,), (
            x, y, z, out, scalar(a), scalar(b), scalar(c), np.int32(N)))

    # ================================================================
    # JIT kernel compilation (lazy)
    # ================================================================

    def _build_preamble(self):
        lines = []
        if self._dtype == np.float32:
            lines.append("#define REAL float")
            lines.append("#define REAL_CONST(x) x##f")
        elif self._dtype == np.float64:
            lines.append("#define REAL double")
            lines.append("#define REAL_CONST(x) x")
        else:
            raise ValueError(f"JIT compilation not supported for {self._dtype}. "
                             "Use AOT build: python -m lpf.csrc.build")
        lines.append(f"#define N_PARAMS {self._reaction_info['n_params']}")
        return "\n".join(lines) + "\n"

    def _inject_reaction(self, template):
        return template.replace("${REACTION_CODE}", self._reaction_info["code"])

    def _compile(self, template, kernel_name):
        source = self._build_preamble() + self._inject_reaction(template)
        return self._cp.RawKernel(source, kernel_name, options=self._options)

    def _compile_utility(self, source, kernel_name):
        full_source = self._build_preamble() + source
        return self._cp.RawKernel(full_source, kernel_name, options=self._options)

    def _get_jit_euler(self):
        if self._jit_euler is None:
            self._jit_euler = self._compile(
                FUSED_EULER_STEP_TEMPLATE, 'fused_euler_step')
        return self._jit_euler

    def _get_jit_pdefunc(self):
        if self._jit_pdefunc is None:
            self._jit_pdefunc = self._compile(
                FUSED_PDEFUNC_TEMPLATE, 'fused_pdefunc')
        return self._jit_pdefunc

    def _get_jit_stage(self):
        if self._jit_stage is None:
            self._jit_stage = self._compile_utility(
                RK_STAGE_UPDATE_SOURCE, 'rk_stage_update')
        return self._jit_stage

    def _get_jit_rk4_combine(self):
        if self._jit_rk4_combine is None:
            self._jit_rk4_combine = self._compile_utility(
                RK4_COMBINE_SOURCE, 'rk4_combine')
        return self._jit_rk4_combine

    def _get_jit_heun_combine(self):
        if self._jit_heun_combine is None:
            self._jit_heun_combine = self._compile_utility(
                HEUN_COMBINE_SOURCE, 'heun_combine')
        return self._jit_heun_combine

    def _get_jit_scale(self):
        if not hasattr(self, '_jit_scale') or self._jit_scale is None:
            self._jit_scale = self._compile_utility(SCALE_SOURCE, 'scale')
        return self._jit_scale

    def _get_jit_lc2(self):
        if not hasattr(self, '_jit_lc2') or self._jit_lc2 is None:
            self._jit_lc2 = self._compile_utility(
                LINEAR_COMBINE2_SOURCE, 'linear_combine2')
        return self._jit_lc2

    def _get_jit_lc3(self):
        if not hasattr(self, '_jit_lc3') or self._jit_lc3 is None:
            self._jit_lc3 = self._compile_utility(
                LINEAR_COMBINE3_SOURCE, 'linear_combine3')
        return self._jit_lc3

    # ================================================================
    # Legacy property API (kept for backward compat)
    # ================================================================

    @property
    def euler_kernel(self):
        if self._use_aot:
            return self._aot.get_euler_kernel(self._model_name, self._dtype)
        return self._get_jit_euler()

    @property
    def pdefunc_kernel(self):
        if self._use_aot:
            return self._aot.get_pdefunc_kernel(self._model_name, self._dtype)
        return self._get_jit_pdefunc()

    @property
    def stage_kernel(self):
        if self._use_aot:
            return self._aot.get_rk_stage_kernel(self._dtype)
        return self._get_jit_stage()

    @property
    def rk4_combine_kernel(self):
        if self._use_aot:
            return self._aot.get_rk4_combine_kernel(self._dtype)
        return self._get_jit_rk4_combine()

    @property
    def heun_combine_kernel(self):
        if self._use_aot:
            return self._aot.get_heun_combine_kernel(self._dtype)
        return self._get_jit_heun_combine()
