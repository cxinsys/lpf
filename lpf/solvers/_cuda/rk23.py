"""CUDA RK23 (Bogacki-Shampine) solver — three fused launches per step."""

from lpf.solvers._cuda.base import CuSolverBase


class CuRK23Solver(CuSolverBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "CuRK23Solver"

    def _alloc_buffers(self, model):
        self._k1, self._k2, self._y_t1, self._y_t2 = \
            self._empty_like_state(model, 4)
        self._cached_shape = model.shape_grid
        self._bufs_ready = True

    def _get_work_bufs(self):
        return [self._k1, self._k2, self._y_t1, self._y_t2]

    def _fast_step(self, model, t, dt, y_cur, y_next):
        op, p = self._op, model.params
        op.stage(y_cur, y_cur, self._k1, self._y_t1, p, 0.5 * dt)
        op.stage(self._y_t1, y_cur, self._k2, self._y_t2, p, 0.75 * dt)
        # y_next = y + dt*(2/9*k1 + 1/3*k2 + 4/9*k3)
        op.rk23(self._y_t2, y_cur, self._k1, self._k2, y_next, p,
                dt * (2.0 / 9.0), dt * (1.0 / 3.0), dt * (4.0 / 9.0))
