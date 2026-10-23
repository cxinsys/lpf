"""CUDA RK4 solver — four fused launches per step."""

from lpf.solvers._cuda.base import CuSolverBase


class CuRungekuttaSolver(CuSolverBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "CuRungekuttaSolver"

    def _alloc_buffers(self, model):
        (self._k1, self._k2, self._k3,
         self._y_t1, self._y_t2) = self._empty_like_state(model, 5)
        self._cached_shape = model.shape_grid
        self._bufs_ready = True

    def _get_work_bufs(self):
        return [self._k1, self._k2, self._k3, self._y_t1, self._y_t2]

    def _fast_step(self, model, t, dt, y_cur, y_next):
        op, p = self._op, model.params
        k1, k2, k3, t1, t2 = self._k1, self._k2, self._k3, self._y_t1, self._y_t2
        # Stage inputs alternate between t1 and t2 (neighbors are read).
        op.stage(y_cur, y_cur, k1, t1, p, 0.5 * dt)
        op.stage(t1, y_cur, k2, t2, p, 0.5 * dt)
        op.stage(t2, y_cur, k3, t1, p, dt)
        op.rk4(t1, y_cur, k1, k2, k3, y_next, p, dt / 6.0)
