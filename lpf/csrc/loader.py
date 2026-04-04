"""
Runtime loader for AOT-compiled CUDA kernels.

Loads the pre-compiled fatbin and provides kernels by name.
Falls back gracefully if the fatbin does not exist.
"""

import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FATBIN_PATH = os.path.join(_THIS_DIR, "kernels.fatbin")

# Model name → short name used in kernel function names
_MODEL_SHORT = {
    "GrayScottModel":        "GrayScott",
    "LiawModel":             "Liaw",
    "SchnakenbergModel":     "Schnakenberg",
    "GiererMeinhardtModel":  "GiererMeinhardt",
    "BrusselatorModel":      "Brusselator",
    "FitzHughNagumoModel":   "FitzHughNagumo",
    "LengyelEpsteinModel":   "LengyelEpstein",
    "ThomasModel":           "Thomas",
    "BarkleyModel":          "Barkley",
}

_DTYPE_SUFFIX = {
    np.dtype(np.float16): "f16",
    np.dtype(np.float32): "f32",
    np.dtype(np.float64): "f64",
}

# Vectorisation width per dtype (elements per thread)
_VEC_WIDTH = {
    np.dtype(np.float16): 2,
    np.dtype(np.float32): 4,
    np.dtype(np.float64): 2,
}

# Stencil tile sizes (must match kernels.cu instantiation)
TILE_X = 32
TILE_Y = 8


class AOTKernelLoader:
    """Load and serve pre-compiled CUDA kernels from a fatbin.

    Usage::

        loader = AOTKernelLoader()
        if loader.is_available():
            kernel = loader.get_euler_kernel("LiawModel", np.float32)
    """

    def __init__(self, fatbin_path=None):
        self._path = fatbin_path or _FATBIN_PATH
        self._module = None
        self._available = False
        self._cache = {}
        self._try_load()

    def _try_load(self):
        if not os.path.isfile(self._path):
            return
        try:
            import cupy as cp
            self._module = cp.RawModule(path=self._path)
            self._available = True
        except Exception:
            self._available = False

    def is_available(self):
        return self._available

    def _get(self, name):
        """Get a kernel function by name (cached)."""
        if name not in self._cache:
            self._cache[name] = self._module.get_function(name)
        return self._cache[name]

    def _resolve(self, prefix, model_name, dtype):
        short = _MODEL_SHORT.get(model_name)
        if short is None:
            raise ValueError(f"Unknown model: {model_name}")
        suffix = _DTYPE_SUFFIX.get(np.dtype(dtype))
        if suffix is None:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return f"{prefix}_{short}_{suffix}"

    # ---- stencil kernels ----

    def get_euler_kernel(self, model_name, dtype):
        return self._get(self._resolve("euler", model_name, dtype))

    def get_pdefunc_kernel(self, model_name, dtype):
        return self._get(self._resolve("pdefunc", model_name, dtype))

    # ---- utility kernels ----

    def get_rk_stage_kernel(self, dtype):
        suffix = _DTYPE_SUFFIX[np.dtype(dtype)]
        return self._get(f"rk_stage_{suffix}")

    def get_rk4_combine_kernel(self, dtype):
        suffix = _DTYPE_SUFFIX[np.dtype(dtype)]
        return self._get(f"rk4_combine_{suffix}")

    def get_heun_combine_kernel(self, dtype):
        suffix = _DTYPE_SUFFIX[np.dtype(dtype)]
        return self._get(f"heun_combine_{suffix}")

    # ---- launch helpers ----

    @staticmethod
    def stencil_grid_block(B, H, W):
        """Return (grid, block) for stencil kernels (2D tiled)."""
        grid = (
            (W + TILE_X - 1) // TILE_X,
            (H + TILE_Y - 1) // TILE_Y,
            B,
        )
        block = (TILE_X, TILE_Y, 1)
        return grid, block

    @staticmethod
    def util_grid_block(N, dtype):
        """Return (grid, block) for vectorised utility kernels."""
        vw = _VEC_WIDTH[np.dtype(dtype)]
        n_threads = (N + vw - 1) // vw
        block = 256
        grid = ((n_threads + block - 1) // block,)
        return grid, (block,)
