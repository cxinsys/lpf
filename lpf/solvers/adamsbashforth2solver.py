
from lpf.solvers.solver import Solver

class AdamsBashforth2Solver(Solver):
    """Explicit two-step Adams–Bashforth method.
       Requires derivative from previous step; bootstraps with Euler.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "AdamsBashforth2Solver"
        self._prev_dydt = None
        self._prev_t = None

    def step(self, model, t, dt, y_mesh):
        dydt = model.pdefunc(t, y_mesh)
        if self._prev_dydt is None:
            # Bootstrap with Euler for the first step
            delta = dt * dydt
        else:
            # AB2: y_{n+1} = y_n + dt*(3/2 f_n - 1/2 f_{n-1})
            delta = dt * (1.5 * dydt - 0.5 * self._prev_dydt)

        # update stored derivative for next step
        self._prev_dydt = dydt
        self._prev_t = t
        return delta
