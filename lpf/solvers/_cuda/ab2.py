"""CUDA Adams-Bashforth 2-step solver — one fused launch per step."""

from lpf.solvers._cuda.base import CuSolverBase


class CuAdamsBashforth2Solver(CuSolverBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "CuAdamsBashforth2Solver"
        self._first_step = True

    def _alloc_buffers(self, model):
        self._dydt_cur, self._dydt_prev = self._empty_like_state(model, 2)
        self._cached_shape = model.shape_grid
        self._bufs_ready = True

    def _get_work_bufs(self):
        return [self._dydt_cur, self._dydt_prev]

    def _on_solve_begin(self):
        self._first_step = True

    def _fast_step(self, model, t, dt, y_cur, y_next):
        op, p = self._op, model.params
        if self._first_step:
            # Euler bootstrap: y_next = y_cur + dt * dydt
            op.stage(y_cur, y_cur, self._dydt_cur, y_next, p, dt)
            self._first_step = False
        else:
            # y_next = y_cur + dt*(1.5*f_n - 0.5*f_(n-1))
            op.ab2(y_cur, self._dydt_prev, self._dydt_cur, y_next, p,
                   1.5 * dt, -0.5 * dt)
        # Swap: current → previous for next step
        self._dydt_cur, self._dydt_prev = self._dydt_prev, self._dydt_cur
