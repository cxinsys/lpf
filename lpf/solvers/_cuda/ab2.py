"""CUDA Adams-Bashforth 2-step solver — all kernel launches, zero CuPy array ops."""

import numpy as np
from lpf.solvers._cuda.base import CuSolverBase


class CuAdamsBashforth2Solver(CuSolverBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "CuAdamsBashforth2Solver"
        self._first_step = True

    def _alloc_buffers(self, model):
        import cupy as cp
        shape, dtype = model.shape_grid, model.y_mesh.dtype
        self._dydt_cur = cp.empty(shape, dtype=dtype)
        self._dydt_prev = cp.empty(shape, dtype=dtype)
        self._delta = cp.empty(shape, dtype=dtype)
        B, H, W = model.batch_size, model.height, model.width
        self._N = 2 * B * H * W
        self._dx2_inv = 1.0 / (model.dx ** 2)
        self._bufs_ready = True

    def _get_work_bufs(self):
        return [self._dydt_cur, self._dydt_prev, self._delta]

    def _on_solve_begin(self):
        self._first_step = True

    def _fast_step(self, model, t, dt, y_cur, y_next):
        km, p = self._km, model.params
        B, H, W = model.batch_size, model.height, model.width
        N, dx2i = self._N, self._dx2_inv

        km.launch_pdefunc(y_cur, self._dydt_cur, p, B, H, W, dx2i)

        if self._first_step:
            # Euler bootstrap: y_next = y_cur + dt * dydt
            km.launch_rk_stage(y_cur, self._dydt_cur, y_next, dt, N)
            self._first_step = False
        else:
            # AB2: delta = dt*(1.5*fn - 0.5*fnm1), y_next = y_cur + delta
            km.launch_linear_combine2(
                self._dydt_cur, self._dydt_prev, self._delta,
                1.5 * dt, -0.5 * dt, N)
            km.launch_rk_stage(y_cur, self._delta, y_next, 1.0, N)

        # Swap: current → previous for next step
        self._dydt_cur, self._dydt_prev = self._dydt_prev, self._dydt_cur

    def step(self, model, t, dt, y_mesh):
        self._ensure_cuda(model)
        if not self._bufs_ready:
            self._alloc_buffers(model)
        km, p = self._km, model.params
        B, H, W = model.batch_size, model.height, model.width
        N, dx2i = self._N, self._dx2_inv

        km.launch_pdefunc(y_mesh, self._dydt_cur, p, B, H, W, dx2i)

        if self._first_step:
            km.launch_scale(self._dydt_cur, self._delta, dt, N)
            self._first_step = False
        else:
            km.launch_linear_combine2(
                self._dydt_cur, self._dydt_prev, self._delta,
                1.5 * dt, -0.5 * dt, N)

        self._dydt_cur, self._dydt_prev = self._dydt_prev, self._dydt_cur
        return self._delta
