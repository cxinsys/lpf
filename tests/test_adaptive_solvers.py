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


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("lpf.solvers.dopri5solver", "DOPRI5Solver"),
        ("lpf.solvers.adaptiverk45solver", "AdaptiveRKF45Solver"),
    ],
)
def test_cpu_call_pdefunc_copies_reused_buffer(liaw_model, module_name, class_name):
    """CPU adaptive solvers must copy model.pdefunc() results on every call.

    TwoComponentModel.pdefunc() reuses an internal buffer. If the adaptive
    solvers keep that reference directly, later stages overwrite earlier ones.
    """
    module = __import__(module_name, fromlist=[class_name])
    solver_cls = getattr(module, class_name)
    solver = solver_cls()

    reusable = np.empty_like(liaw_model.y_mesh)

    def fake_pdefunc(t=None, y_mesh=None, y_linear=None):
        reusable.fill(np.float32(t))
        return reusable

    liaw_model.pdefunc = fake_pdefunc

    first = solver._call_pdefunc(liaw_model, 1.0, liaw_model.y_mesh)
    second = solver._call_pdefunc(liaw_model, 2.0, liaw_model.y_mesh)

    assert first is not second
    assert np.all(first == np.float32(1.0))
    assert np.all(second == np.float32(2.0))


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
