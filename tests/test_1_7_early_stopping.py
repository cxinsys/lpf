"""Test 1-7: is_early_stopping() must not raise AttributeError.

Before fix: self._f and self._g were never assigned in pdefunc(),
so is_early_stopping() raised AttributeError on self._f.
"""

import numpy as np
import pytest


class TestEarlyStopping:

    def test_f_g_stored_after_pdefunc(self, liaw_model):
        """After calling pdefunc(), self._f and self._g must exist."""
        liaw_model.pdefunc(t=0.0, y_mesh=liaw_model.y_mesh)

        assert hasattr(liaw_model, "_f"), "self._f should be set after pdefunc()"
        assert hasattr(liaw_model, "_g"), "self._g should be set after pdefunc()"
        assert liaw_model._f is not None
        assert liaw_model._g is not None

    def test_is_early_stopping_does_not_raise(self, liaw_model):
        """is_early_stopping() should run without AttributeError after pdefunc()."""
        # First, call pdefunc so _f and _g are set
        liaw_model.pdefunc(t=0.0, y_mesh=liaw_model.y_mesh)

        # This should not raise
        result = liaw_model.is_early_stopping(rtol=1e-6)
        assert isinstance(result, (bool, np.bool_))

    def test_is_early_stopping_after_solve(self, liaw_model):
        """After a solve, is_early_stopping should work (pdefunc was called
        at least once during the solve loop)."""
        from lpf.solvers import EulerSolver

        solver = EulerSolver()
        solver.solve(model=liaw_model, dt=0.01, n_iters=5,
                     init_model=False, verbose=0)

        result = liaw_model.is_early_stopping(rtol=1e-3)
        assert isinstance(result, (bool, np.bool_))

    def test_f_g_shape_matches_interior(self, liaw_model):
        """self._f and self._g should have the same shape as the interior grid."""
        liaw_model.pdefunc(t=0.0, y_mesh=liaw_model.y_mesh)

        # Interior is [1:-1, 1:-1] of (batch, H, W) → (batch, H-2, W-2)
        expected_shape = (1, 30, 30)  # (batch=1, 32-2, 32-2)
        f_shape = liaw_model._f.shape
        g_shape = liaw_model._g.shape

        assert f_shape == expected_shape, f"_f shape {f_shape} != {expected_shape}"
        assert g_shape == expected_shape, f"_g shape {g_shape} != {expected_shape}"
