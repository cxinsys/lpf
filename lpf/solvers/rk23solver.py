from lpf.solvers.solver import Solver


class RK23Solver(Solver):
    """Bogacki-Shampine RK23 (fixed-step 3rd-order) solver.

    Automatically uses fused CUDA kernels when the model is on a CuPy
    device (``device="cuda:*"``).
    """

    def __init__(self, *args, fast_math=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "RK23Solver"
        self._fast_math = fast_math

    def _make_cuda_solver(self):
        from lpf.solvers._cuda.rk23 import CuRK23Solver
        return CuRK23Solver(fast_math=self._fast_math)

    def step(self, model, t, dt, y_mesh):
        # pdefunc() returns an internal buffer that is overwritten on the
        # next call, so each stage result must be copied.
        k1 = model.am.copy(model.pdefunc(t, y_mesh))
        k2 = model.am.copy(model.pdefunc(t + 0.5 * dt, y_mesh + 0.5 * dt * k1))
        k3 = model.pdefunc(t + 0.75 * dt, y_mesh + 0.75 * dt * k2)
        return dt * ((2.0 / 9.0) * k1 + (1.0 / 3.0) * k2 + (4.0 / 9.0) * k3)
