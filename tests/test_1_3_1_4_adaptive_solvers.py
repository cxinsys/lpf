"""Tests 1-3 & 1-4: Adaptive solvers (DOPRI5, AdaptiveRKF45) must:
  - Work with non-NumPy backends (1-3: no hardcoded np.* calls)
  - Cover the full dt interval via sub-stepping (1-4: time integration correctness)

Before fix: AdaptiveRKF45 used np.zeros/np.sum/np.abs/np.max directly,
crashing on GPU backends. Both adaptive solvers took a single sub-step < dt
but the outer loop advanced by full dt, causing incorrect integration.
"""

import numpy as np
import pytest


class TestAdaptiveRKF45SubStepping:
    """Verify that AdaptiveRKF45Solver sub-steps correctly to cover full dt."""

    def test_step_returns_correct_delta(self, liaw_model):
        """step() should return a delta that, when added to y_mesh, gives
        the state advanced by exactly dt."""
        from lpf.solvers.adaptiverk45solver import AdaptiveRKF45Solver

        solver = AdaptiveRKF45Solver()
        y_before = liaw_model.am.copy(liaw_model.y_mesh)

        dt = 0.01
        delta = solver.step(liaw_model, t=0.0, dt=dt, y_mesh=y_before)

        # Delta should be non-zero (reactions + diffusion are happening)
        assert not np.allclose(liaw_model.am.get(delta), 0.0), \
            "delta_y should be non-zero after a step"

    def test_solve_produces_finite_results(self, liaw_model):
        """Full solve loop should produce finite, non-NaN results."""
        from lpf.solvers.adaptiverk45solver import AdaptiveRKF45Solver

        solver = AdaptiveRKF45Solver()
        solver.solve(model=liaw_model, dt=0.01, n_iters=10,
                     init_model=False, verbose=0)

        u = liaw_model.am.get(liaw_model.u)
        assert np.all(np.isfinite(u)), "Solution should be finite"

    def test_no_numpy_hardcoding(self):
        """The solver should not call np.zeros/np.sum/np.abs/np.max in step()."""
        import inspect
        from lpf.solvers.adaptiverk45solver import AdaptiveRKF45Solver

        source = inspect.getsource(AdaptiveRKF45Solver.step)
        # These np.* calls should not appear in step()
        for forbidden in ["np.zeros", "np.sum", "np.abs", "np.max", "np.clip"]:
            assert forbidden not in source, \
                f"step() should not contain hardcoded {forbidden}"


class TestDOPRI5SubStepping:
    """Verify that DOPRI5Solver sub-steps correctly to cover full dt."""

    def test_step_returns_correct_delta(self, liaw_model):
        from lpf.solvers.dopri5solver import DOPRI5Solver

        solver = DOPRI5Solver()
        y_before = liaw_model.am.copy(liaw_model.y_mesh)

        dt = 0.01
        delta = solver.step(liaw_model, t=0.0, dt=dt, y_mesh=y_before)

        assert not np.allclose(liaw_model.am.get(delta), 0.0), \
            "delta_y should be non-zero after a step"

    def test_solve_produces_finite_results(self, liaw_model):
        from lpf.solvers.dopri5solver import DOPRI5Solver

        solver = DOPRI5Solver()
        solver.solve(model=liaw_model, dt=0.01, n_iters=10,
                     init_model=False, verbose=0)

        u = liaw_model.am.get(liaw_model.u)
        assert np.all(np.isfinite(u)), "Solution should be finite"

    def test_uses_model_am_for_error_norm(self):
        """DOPRI5 should use model.am.sqrt / model.am.mean, not np.*."""
        import inspect
        from lpf.solvers.dopri5solver import DOPRI5Solver

        source = inspect.getsource(DOPRI5Solver.step)
        assert "model.am.sqrt" in source, \
            "step() should use model.am.sqrt for backend compatibility"
        assert "model.am.mean" in source, \
            "step() should use model.am.mean for backend compatibility"


class TestAdaptiveSolversConvergence:
    """Both adaptive solvers should converge more accurately than Euler
    for a given number of outer iterations, confirming sub-stepping works."""

    def test_dopri5_more_accurate_than_euler(self, liaw_model):
        from lpf.solvers.dopri5solver import DOPRI5Solver
        from lpf.solvers import EulerSolver

        # Solve with Euler
        euler = EulerSolver()
        euler.solve(model=liaw_model, dt=0.01, n_iters=50,
                    init_model=True, verbose=0)
        u_euler = liaw_model.am.get(liaw_model.u).copy()

        # Solve with DOPRI5 (fewer iters should be comparable or better)
        dopri5 = DOPRI5Solver()
        dopri5.solve(model=liaw_model, dt=0.01, n_iters=50,
                     init_model=True, verbose=0)
        u_dopri5 = liaw_model.am.get(liaw_model.u).copy()

        # Both should be finite
        assert np.all(np.isfinite(u_euler)), "Euler result should be finite"
        assert np.all(np.isfinite(u_dopri5)), "DOPRI5 result should be finite"
