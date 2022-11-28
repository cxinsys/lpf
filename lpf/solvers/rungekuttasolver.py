from lpf.solvers.solver import Solver


class RungeKuttaSolver(Solver):
    """Runge-Kutta solver (RK45)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "RungeKutta"

    def step(self, model, t, dt, y_linear):
        k1 = model.pdefunc(t, y_linear)
        k2 = model.pdefunc(t + 0.5 * dt, y_linear + 0.5 * dt * k1)
        k3 = model.pdefunc(t + 0.5 * dt, y_linear + 0.5 * dt * k2)
        k4 = model.pdefunc(t + dt, y_linear + dt * 0.5 * k3)

        return dt * (k1 + 2*k2 + 2*k3 + k4) / 6

