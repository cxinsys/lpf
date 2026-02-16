"""SolverFactory must match "adaptive_rk45" to AdaptiveRKF45Solver,
not RungeKuttaSolver.

Before fix: "rk45" in "adaptiverk45" was True, so the elif chain matched
RungeKuttaSolver before reaching the AdaptiveRKF45Solver branch.
"""

import pytest

from lpf.solvers.solverfactory import SolverFactory
from lpf.solvers.adaptiverk45solver import AdaptiveRKF45Solver
from lpf.solvers.dopri5solver import DOPRI5Solver
from lpf.solvers import RungeKuttaSolver, EulerSolver, HeunSolver
from lpf.solvers.adamsbashforth2solver import AdamsBashforth2Solver
from lpf.solvers.rk23solver import RK23Solver


class TestSolverFactoryMatching:

    def test_adaptive_rk45_creates_correct_solver(self):
        """'adaptive_rk45' must create AdaptiveRKF45Solver, not RungeKuttaSolver."""
        solver = SolverFactory.create("adaptive_rk45")
        assert isinstance(solver, AdaptiveRKF45Solver), \
            f"Expected AdaptiveRKF45Solver, got {type(solver).__name__}"

    def test_rkf45_creates_correct_solver(self):
        solver = SolverFactory.create("rkf45")
        assert isinstance(solver, AdaptiveRKF45Solver)

    def test_fehlberg_creates_correct_solver(self):
        solver = SolverFactory.create("fehlberg")
        assert isinstance(solver, AdaptiveRKF45Solver)

    def test_rk45_still_creates_rungekutta(self):
        """Plain 'rk45' should still create the classic RK4 solver."""
        solver = SolverFactory.create("rk45")
        assert isinstance(solver, RungeKuttaSolver)

    def test_rungekutta_creates_rungekutta(self):
        solver = SolverFactory.create("rungekutta")
        assert isinstance(solver, RungeKuttaSolver)

    def test_dopri5_creates_correct_solver(self):
        solver = SolverFactory.create("dopri5")
        assert isinstance(solver, DOPRI5Solver)

    def test_euler_creates_correct_solver(self):
        solver = SolverFactory.create("euler")
        assert isinstance(solver, EulerSolver)

    def test_heun_creates_correct_solver(self):
        solver = SolverFactory.create("heun")
        assert isinstance(solver, HeunSolver)

    def test_ab2_creates_correct_solver(self):
        solver = SolverFactory.create("ab2")
        assert isinstance(solver, AdamsBashforth2Solver)

    def test_rk23_creates_correct_solver(self):
        solver = SolverFactory.create("rk23")
        assert isinstance(solver, RK23Solver)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="not a supported solver"):
            SolverFactory.create("nonexistent_solver")

    def test_case_insensitive_with_dashes(self):
        """Names with dashes, underscores, mixed case should all work."""
        solver = SolverFactory.create("Adaptive-RK45")
        assert isinstance(solver, AdaptiveRKF45Solver)

        solver = SolverFactory.create("DOPRI5")
        assert isinstance(solver, DOPRI5Solver)

    def test_recommended_stiff_solver_resolves(self):
        """The recommended solver for stiff problems should be creatable."""
        name = SolverFactory.get_recommended_solver("stiff")
        solver = SolverFactory.create(name)
        # It should at least be a valid solver, not crash
        assert solver is not None
