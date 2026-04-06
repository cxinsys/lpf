"""Concrete JAX-accelerated solver subclasses.

These are instantiated lazily by ``Solver._get_jax_solver()`` when a
JAX-backed model is passed to ``solve()``.  Users normally never touch
them directly — they just set ``device="jax:gpu:0"`` on the model and
call their usual ``EulerSolver.solve(model)`` etc.
"""

from lpf.solvers._jax.base import JaxSolverBase
from lpf.solvers._jax.adaptive_base import JaxAdaptiveBase


# ---------------------------------------------------------------------------
# Fixed-step solvers
# ---------------------------------------------------------------------------

class JaxEulerSolver(JaxSolverBase):
    def __init__(self, *args, **kwargs):
        super().__init__("EulerSolver", *args, **kwargs)


class JaxHeunSolver(JaxSolverBase):
    def __init__(self, *args, **kwargs):
        super().__init__("HeunSolver", *args, **kwargs)


class JaxRungeKuttaSolver(JaxSolverBase):
    def __init__(self, *args, **kwargs):
        super().__init__("RungeKuttaSolver", *args, **kwargs)


class JaxRK23Solver(JaxSolverBase):
    def __init__(self, *args, **kwargs):
        super().__init__("RK23Solver", *args, **kwargs)


class JaxAdamsBashforth2Solver(JaxSolverBase):
    def __init__(self, *args, **kwargs):
        super().__init__("AdamsBashforth2Solver", *args, **kwargs)


# ---------------------------------------------------------------------------
# Adaptive solvers
# ---------------------------------------------------------------------------

class JaxAdaptiveRKF45Solver(JaxAdaptiveBase):
    def __init__(self, tolerance=1e-6, safety_factor=0.9,
                 min_factor=0.1, max_factor=5.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "AdaptiveRKF45Solver"
        self._adaptive_kind = "rkf45"
        self._hyperparams = dict(
            tolerance=tolerance,
            safety=safety_factor,
            min_factor=min_factor,
            max_factor=max_factor,
        )


class JaxDOPRI5Solver(JaxAdaptiveBase):
    def __init__(self, tolerance=1e-6, safety=0.9,
                 min_factor=0.2, max_factor=5.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "DOPRI5Solver"
        self._adaptive_kind = "dopri5"
        self._hyperparams = dict(
            tolerance=tolerance,
            safety=safety,
            min_factor=min_factor,
            max_factor=max_factor,
        )
