"""CUDA Heun solver — all kernel launches, zero CuPy array ops."""

import numpy as np
from lpf.solvers._cuda.base import CuSolverBase


class CuHeunSolver(CuSolverBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "CuHeunSolver"

    def _alloc_buffers(self, model):
        import cupy as cp
        shape, dtype = model.shape_grid, model.y_mesh.dtype
        with model.am:
            self._k1 = cp.empty(shape, dtype=dtype)
            self._k2 = cp.empty(shape, dtype=dtype)
            self._y_temp = cp.empty(shape, dtype=dtype)
        self._N = int(np.prod(model.shape_grid))
        self._cached_shape = shape
        self._bufs_ready = True

    def _get_work_bufs(self):
        return [self._k1, self._k2, self._y_temp]

    def _fast_step(self, model, t, dt, y_cur, y_next):
        km = self._km
        N = self._N

        km.launch_model_pdefunc(model, y_cur, self._k1)
        km.launch_rk_stage(y_cur, self._k1, self._y_temp, dt, N)
        km.launch_model_pdefunc(model, self._y_temp, self._k2)
        # y_next = y_cur + 0.5*dt*(k1 + k2)  →  y_next = y_cur + delta
        km.launch_linear_combine2(
            self._k1, self._k2, self._y_temp,   # reuse y_temp as delta
            0.5 * dt, 0.5 * dt, N)
        km.launch_rk_stage(y_cur, self._y_temp, y_next, 1.0, N)

    def step(self, model, t, dt, y_mesh):
        self._ensure_cuda(model)
        if not self._bufs_ready:
            self._alloc_buffers(model)
        import cupy as cp
        delta = cp.empty_like(y_mesh)
        km = self._km
        N = self._N
        km.launch_model_pdefunc(model, y_mesh, self._k1)
        km.launch_rk_stage(y_mesh, self._k1, self._y_temp, dt, N)
        km.launch_model_pdefunc(model, self._y_temp, self._k2)
        km.launch_linear_combine2(self._k1, self._k2, delta, 0.5*dt, 0.5*dt, N)
        return delta
