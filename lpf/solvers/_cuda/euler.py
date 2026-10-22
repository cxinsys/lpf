"""CUDA Euler solver — fused ping-pong kernel, one launch per step."""

import numpy as np
from lpf.solvers._cuda.base import CuSolverBase


class CuEulerSolver(CuSolverBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "CuEulerSolver"

    def _alloc_buffers(self, model):
        self._cached_shape = model.shape_grid
        self._bufs_ready = True

    def _get_work_bufs(self):
        return []

    def _fast_step(self, model, t, dt, y_cur, y_next):
        self._km.launch_model_euler(model, y_cur, y_next, dt)

    def step(self, model, t, dt, y_mesh):
        """Fallback step() for parent solve() (file-I/O path)."""
        import cupy as cp
        self._ensure_cuda(model)
        if not self._bufs_ready:
            self._alloc_buffers(model)
        if not hasattr(self, '_dydt') or self._dydt is None \
                or self._dydt.shape != y_mesh.shape:
            self._dydt = cp.empty_like(y_mesh)
        self._km.launch_model_pdefunc(model, y_mesh, self._dydt)
        return self._dydt * dt
