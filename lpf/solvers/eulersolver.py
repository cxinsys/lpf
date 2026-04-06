from lpf.solvers.solver import Solver


class EulerSolver(Solver):
    """Euler method solver.

    Automatically uses fused CUDA kernels when the model is on a CuPy
    device (``device="cuda:*"``).  No manual backend selection needed.
    """

    def __init__(self, *args, fast_math=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "EulerSolver"
        self._fast_math = fast_math

    def _make_cuda_solver(self):
        from lpf.solvers._cuda.euler import CuEulerSolver
        return CuEulerSolver(fast_math=self._fast_math)

    def _make_jax_solver(self):
        from lpf.solvers._jax.solvers import JaxEulerSolver
        return JaxEulerSolver()

    def step(self, model, t, dt, y_mesh):
        dydt = model.pdefunc(t, y_mesh)
        return dydt * dt
