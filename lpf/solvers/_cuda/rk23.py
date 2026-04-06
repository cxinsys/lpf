"""CUDA RK23 (Bogacki-Shampine) solver — all kernel launches, zero CuPy array ops."""

import numpy as np
from lpf.solvers._cuda.base import CuSolverBase


class CuRK23Solver(CuSolverBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "CuRK23Solver"

    def _alloc_buffers(self, model):
        import cupy as cp
        shape, dtype = model.shape_grid, model.y_mesh.dtype
        with model.am:
            self._k1 = cp.empty(shape, dtype=dtype)
            self._k2 = cp.empty(shape, dtype=dtype)
            self._k3 = cp.empty(shape, dtype=dtype)
            self._y_temp = cp.empty(shape, dtype=dtype)
            self._delta = cp.empty(shape, dtype=dtype)
        B, H, W = model.batch_size, model.height, model.width
        self._N = 2 * B * H * W
        self._dx2_inv = 1.0 / (model.dx ** 2)
        self._cached_shape = shape
        self._bufs_ready = True

    def _get_work_bufs(self):
        return [self._k1, self._k2, self._k3, self._y_temp, self._delta]

    def _fast_step(self, model, t, dt, y_cur, y_next):
        km, p = self._km, model.params
        B, H, W = model.batch_size, model.height, model.width
        N, dx2i = self._N, self._dx2_inv

        km.launch_pdefunc(y_cur, self._k1, p, B, H, W, dx2i)
        km.launch_rk_stage(y_cur, self._k1, self._y_temp, 0.5 * dt, N)

        km.launch_pdefunc(self._y_temp, self._k2, p, B, H, W, dx2i)
        km.launch_rk_stage(y_cur, self._k2, self._y_temp, 0.75 * dt, N)

        km.launch_pdefunc(self._y_temp, self._k3, p, B, H, W, dx2i)

        # delta = dt*(2/9*k1 + 1/3*k2 + 4/9*k3)
        km.launch_linear_combine3(
            self._k1, self._k2, self._k3, self._delta,
            dt * (2.0 / 9.0), dt * (1.0 / 3.0), dt * (4.0 / 9.0), N)
        km.launch_rk_stage(y_cur, self._delta, y_next, 1.0, N)

    def step(self, model, t, dt, y_mesh):
        self._ensure_cuda(model)
        if not self._bufs_ready:
            self._alloc_buffers(model)
        km, p = self._km, model.params
        B, H, W = model.batch_size, model.height, model.width
        N, dx2i = self._N, self._dx2_inv

        km.launch_pdefunc(y_mesh, self._k1, p, B, H, W, dx2i)
        km.launch_rk_stage(y_mesh, self._k1, self._y_temp, 0.5 * dt, N)
        km.launch_pdefunc(self._y_temp, self._k2, p, B, H, W, dx2i)
        km.launch_rk_stage(y_mesh, self._k2, self._y_temp, 0.75 * dt, N)
        km.launch_pdefunc(self._y_temp, self._k3, p, B, H, W, dx2i)
        km.launch_linear_combine3(
            self._k1, self._k2, self._k3, self._delta,
            dt * (2.0 / 9.0), dt * (1.0 / 3.0), dt * (4.0 / 9.0), N)
        return self._delta
