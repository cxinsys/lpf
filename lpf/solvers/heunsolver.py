from lpf.solvers.solver import Solver


class HeunSolver(Solver):
    """Heun solver
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "HeunSolver"

    def step(self, model, t, dt, y_mesh):
        k1 = dt * model.pdefunc(t, y_mesh)
        k2 = dt * model.pdefunc(t + dt, y_mesh + k1)
        return 0.5 * (k1 + k2)

