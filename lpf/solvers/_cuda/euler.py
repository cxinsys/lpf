"""CUDA Euler solver — fused ping-pong kernel, one launch per step."""

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
        self._op.euler(y_cur, y_next, model.params, dt)
