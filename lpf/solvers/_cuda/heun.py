"""CUDA Heun solver — two fused launches per step."""

from lpf.solvers._cuda.base import CuSolverBase


class CuHeunSolver(CuSolverBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "CuHeunSolver"

    def _alloc_buffers(self, model):
        self._k1, self._y_temp = self._empty_like_state(model, 2)
        self._cached_shape = model.shape_grid
        self._bufs_ready = True

    def _get_work_bufs(self):
        return [self._k1, self._y_temp]

    def _fast_step(self, model, t, dt, y_cur, y_next):
        op, p = self._op, model.params
        # k1 = f(y), y_temp = y + dt*k1; y_next = y + (dt/2*k1 + dt/2*f(y_temp))
        op.stage(y_cur, y_cur, self._k1, self._y_temp, p, dt)
        op.heun(self._y_temp, y_cur, self._k1, y_next, p, 0.5 * dt, 0.5 * dt)
