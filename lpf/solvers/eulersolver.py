from lpf.solvers.solver import Solver


class EulerSolver(Solver):
    """Euler method solver
    """
    def step(self, model, t, dt, y_linear):
        dydt = model.pdefunc(t, y_linear)
        return dydt * dt
