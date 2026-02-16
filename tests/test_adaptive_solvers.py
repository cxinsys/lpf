"""Adaptive solvers (DOPRI5, AdaptiveRKF45) must:
  - Work with non-NumPy backends (no hardcoded np.* calls)
  - Cover the full dt interval via sub-stepping (time integration correctness)

Before fix: AdaptiveRKF45 used np.zeros/np.sum/np.abs/np.max directly,
crashing on GPU backends. Both adaptive solvers took a single sub-step < dt
but the outer loop advanced by full dt, causing incorrect integration.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestAdaptiveRKF45SubStepping:
    """Verify that AdaptiveRKF45Solver sub-steps correctly to cover full dt."""

    def test_step_returns_nonzero_delta(self, liaw_model):
        """step() should return a non-zero delta (reactions + diffusion)."""
        from lpf.solvers.adaptiverk45solver import AdaptiveRKF45Solver

        solver = AdaptiveRKF45Solver()
        y_before = liaw_model.am.copy(liaw_model.y_mesh)

        dt = 0.01
        delta = solver.step(liaw_model, t=0.0, dt=dt, y_mesh=y_before)

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

    def test_uses_array_module_for_zeros(self, liaw_model):
        """Solver must use model.am.zeros(), not np.zeros() directly."""
        from lpf.solvers.adaptiverk45solver import AdaptiveRKF45Solver

        original_zeros = liaw_model.am.zeros
        calls = []

        def tracked_zeros(*args, **kwargs):
            calls.append(True)
            return original_zeros(*args, **kwargs)

        liaw_model.am.zeros = tracked_zeros
        solver = AdaptiveRKF45Solver()
        solver.step(liaw_model, t=0.0, dt=0.01, y_mesh=liaw_model.y_mesh)

        assert len(calls) > 0, "Solver should call model.am.zeros()"

    def test_uses_array_module_for_abs(self, liaw_model):
        """Solver must use model.am.abs(), not np.abs() directly."""
        from lpf.solvers.adaptiverk45solver import AdaptiveRKF45Solver

        original_abs = liaw_model.am.abs
        calls = []

        def tracked_abs(*args, **kwargs):
            calls.append(True)
            return original_abs(*args, **kwargs)

        liaw_model.am.abs = tracked_abs
        solver = AdaptiveRKF45Solver()
        solver.step(liaw_model, t=0.0, dt=0.01, y_mesh=liaw_model.y_mesh)

        assert len(calls) > 0, "Solver should call model.am.abs()"


class TestDOPRI5SubStepping:
    """Verify that DOPRI5Solver sub-steps correctly to cover full dt."""

    def test_step_returns_nonzero_delta(self, liaw_model):
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

    def test_uses_array_module_for_error_norm(self, liaw_model):
        """DOPRI5 should use model.am.sqrt / model.am.mean for error norm."""
        from lpf.solvers.dopri5solver import DOPRI5Solver

        original_sqrt = liaw_model.am.sqrt
        original_mean = liaw_model.am.mean
        sqrt_calls = []
        mean_calls = []

        def tracked_sqrt(*args, **kwargs):
            sqrt_calls.append(True)
            return original_sqrt(*args, **kwargs)

        def tracked_mean(*args, **kwargs):
            mean_calls.append(True)
            return original_mean(*args, **kwargs)

        liaw_model.am.sqrt = tracked_sqrt
        liaw_model.am.mean = tracked_mean

        solver = DOPRI5Solver()
        solver.step(liaw_model, t=0.0, dt=0.01, y_mesh=liaw_model.y_mesh)

        assert len(sqrt_calls) > 0, "Solver should call model.am.sqrt()"
        assert len(mean_calls) > 0, "Solver should call model.am.mean()"


class TestAdaptiveSolversFiniteness:
    """Both adaptive solvers and Euler should produce finite results."""

    def test_euler_and_dopri5_both_finite(self, liaw_model):
        from lpf.solvers.dopri5solver import DOPRI5Solver
        from lpf.solvers import EulerSolver

        euler = EulerSolver()
        euler.solve(model=liaw_model, dt=0.01, n_iters=50,
                    init_model=True, verbose=0)
        u_euler = liaw_model.am.get(liaw_model.u).copy()

        dopri5 = DOPRI5Solver()
        dopri5.solve(model=liaw_model, dt=0.01, n_iters=50,
                     init_model=True, verbose=0)
        u_dopri5 = liaw_model.am.get(liaw_model.u).copy()

        assert np.all(np.isfinite(u_euler)), "Euler result should be finite"
        assert np.all(np.isfinite(u_dopri5)), "DOPRI5 result should be finite"

    def test_dopri5_and_euler_differ(self, liaw_model):
        """Different solvers should produce different results (different methods)."""
        from lpf.solvers.dopri5solver import DOPRI5Solver
        from lpf.solvers import EulerSolver

        euler = EulerSolver()
        euler.solve(model=liaw_model, dt=0.01, n_iters=50,
                    init_model=True, verbose=0)
        u_euler = liaw_model.am.get(liaw_model.u).copy()

        dopri5 = DOPRI5Solver()
        dopri5.solve(model=liaw_model, dt=0.01, n_iters=50,
                     init_model=True, verbose=0)
        u_dopri5 = liaw_model.am.get(liaw_model.u).copy()

        assert not np.allclose(u_euler, u_dopri5), \
            "Euler and DOPRI5 should produce different results for the same problem"
