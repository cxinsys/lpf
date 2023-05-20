from lpf.solvers.solver import Solver


class RungeKuttaSolver(Solver):
    """Runge-Kutta solver (RK45)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "RungeKuttaSolver"

    def step(self, model, t, dt, y_mesh):
        k1 = dt * model.pdefunc(t, y_mesh)
        k2 = dt * model.pdefunc(t + 0.5 * dt, y_mesh + 0.5 * k1)
        k3 = dt * model.pdefunc(t + 0.5 * dt, y_mesh + 0.5 * k2)
        k4 = dt * model.pdefunc(t + dt, y_mesh + k3)

        return (k1 + 2*k2 + 2*k3 + k4) / 6

