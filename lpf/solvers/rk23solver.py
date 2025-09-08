
from lpf.solvers.solver import Solver

class RK23Solver(Solver):
    """Bogacki–Shampine RK23 (fixed-step 3rd order update).  
       Useful for moderately stiff problems, and can be embedded for adaptivity.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "RK23Solver"

    def step(self, model, t, dt, y_mesh):
        k1 = model.pdefunc(t, y_mesh)
        k2 = model.pdefunc(t + 0.5 * dt, y_mesh + 0.5 * dt * k1)
        k3 = model.pdefunc(t + 0.75 * dt, y_mesh + 0.75 * dt * k2)
        # Third-order solution
        y_next = y_mesh + dt * ((2.0/9.0) * k1 + (1.0/3.0) * k2 + (4.0/9.0) * k3)
        return y_next - y_mesh
