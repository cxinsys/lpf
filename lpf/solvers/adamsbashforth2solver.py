from lpf.solvers.solver import Solver


class AdamsBashforth2Solver(Solver):
    """Explicit two-step Adams-Bashforth method."""

    def __init__(self, *args, fast_math=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "AdamsBashforth2Solver"
        self._prev_dydt = None
        self._fast_math = fast_math

    def _make_cuda_solver(self):
        from lpf.solvers._cuda.ab2 import CuAdamsBashforth2Solver
        return CuAdamsBashforth2Solver(fast_math=self._fast_math)

    def solve(self, model=None, **kwargs):
        self._prev_dydt = None  # reset multi-step state before any path
        return super().solve(model=model, **kwargs)

    def step(self, model, t, dt, y_mesh):
        dydt = model.pdefunc(t, y_mesh)
        if self._prev_dydt is None:
            delta = dt * dydt
        else:
            delta = dt * (1.5 * dydt - 0.5 * self._prev_dydt)
        self._prev_dydt = dydt
        return delta
