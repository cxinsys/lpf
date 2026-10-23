"""
Runtime loader for AOT-compiled CUDA kernels.

Loads the pre-compiled fatbin (built from csrc/kernels.cu by
``python -m lpf.kernels.aot.build``) and provides kernels by name.
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

# Stencil tile sizes (must match GridSpace / DomainSpace in spaces.cuh)
TILE_X = 32
TILE_Y = 8


class AOTKernelLoader:
    """Load and serve pre-compiled CUDA kernels from a fatbin.

    Usage::

        loader = AOTKernelLoader()
        if loader.is_available():
            kernel = loader.get_function("lpf_euler_grid_Liaw_f32")
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

    def get_function(self, name):
        """Kernel function by name (cached)."""
        if name not in self._cache:
            self._cache[name] = self._module.get_function(name)
        return self._cache[name]

    @staticmethod
    def stencil_grid_block(B, H, W):
        """Return (grid, block) for the 2D stencil kernels (tiled)."""
        grid = (
            (W + TILE_X - 1) // TILE_X,
            (H + TILE_Y - 1) // TILE_Y,
            B,
        )
        block = (TILE_X, TILE_Y, 1)
        return grid, block
