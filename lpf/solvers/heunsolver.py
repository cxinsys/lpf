from lpf.solvers.solver import Solver


class HeunSolver(Solver):
    """Heun method (improved Euler / predictor-corrector) solver.

    Automatically uses fused CUDA kernels when the model is on a CuPy
    device (``device="cuda:*"``).
    """

    def __init__(self, *args, fast_math=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "HeunSolver"
        self._fast_math = fast_math

    def _make_cuda_solver(self):
        from lpf.solvers._cuda.heun import CuHeunSolver
        return CuHeunSolver(fast_math=self._fast_math)

    def step(self, model, t, dt, y_mesh):
        # pdefunc() returns an internal buffer that is overwritten on the
        # next call, so each stage result must be copied.
        k1 = dt * model.pdefunc(t, y_mesh).copy()
        k2 = dt * model.pdefunc(t + dt, y_mesh + k1)
        return 0.5 * (k1 + k2)
