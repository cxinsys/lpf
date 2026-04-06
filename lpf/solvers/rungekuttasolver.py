from lpf.solvers.solver import Solver


class RungeKuttaSolver(Solver):
    """Runge-Kutta 4th-order solver.

    Automatically uses fused CUDA kernels when the model is on a CuPy
    device (``device="cuda:*"``).
    """

    def __init__(self, *args, fast_math=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "RungeKuttaSolver"
        self._fast_math = fast_math

    def _make_cuda_solver(self):
        from lpf.solvers._cuda.rk4 import CuRungekuttaSolver
        return CuRungekuttaSolver(fast_math=self._fast_math)

    def step(self, model, t, dt, y_mesh):
        # pdefunc() returns an internal buffer that is overwritten on the
        # next call, so each stage result must be copied.
        k1 = dt * model.am.copy(model.pdefunc(t, y_mesh))
        k2 = dt * model.am.copy(model.pdefunc(t + 0.5 * dt, y_mesh + 0.5 * k1))
        k3 = dt * model.am.copy(model.pdefunc(t + 0.5 * dt, y_mesh + 0.5 * k2))
        k4 = dt * model.pdefunc(t + dt, y_mesh + k3)
        return (k1 + 2*k2 + 2*k3 + k4) / 6
