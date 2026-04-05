"""CUDA Euler solver — fused ping-pong kernel, one launch per step."""

import numpy as np
from lpf.solvers._cuda.base import CuSolverBase


class CuEulerSolver(CuSolverBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "CuEulerSolver"

    def _alloc_buffers(self, model):
        self._dx2_inv = 1.0 / (model.dx ** 2)
        self._bufs_ready = True

    def _get_work_bufs(self):
        return []

    def _fast_step(self, model, t, dt, y_cur, y_next):
        B, H, W = model.batch_size, model.height, model.width
        self._km.launch_euler(
            y_cur, y_next, model.params, B, H, W, dt, self._dx2_inv)

    def step(self, model, t, dt, y_mesh):
        """Fallback step() for parent solve() (file-I/O path)."""
        import cupy as cp
        self._ensure_cuda(model)
        if not self._bufs_ready:
            self._alloc_buffers(model)
        B, H, W = model.batch_size, model.height, model.width
        if not hasattr(self, '_dydt') or self._dydt is None \
                or self._dydt.shape != y_mesh.shape:
            self._dydt = cp.empty_like(y_mesh)
        self._km.launch_pdefunc(
            y_mesh, self._dydt, model.params, B, H, W, self._dx2_inv)
        return self._dydt * dt
