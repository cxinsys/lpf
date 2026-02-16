"""Test 1-6: colorize() must accept numpy arrays for thr_color without crashing.

Before fix: `if not thr_color:` on a multi-element numpy array raised
ValueError: The truth value of an array with more than one element is ambiguous.
"""

import numpy as np
import pytest


class TestColorizeTruthiness:

    def test_colorize_with_array_thr_color(self, liaw_model):
        """Passing a numpy array as thr_color should not raise ValueError."""
        thr = np.array([[[0.3]]])  # shape (1, 1, 1)
        color = liaw_model.colorize(thr_color=thr)
        assert color.shape == (1, 32, 32, 3)

    def test_colorize_with_none_uses_default(self, liaw_model):
        """Passing thr_color=None should use model's default."""
        color = liaw_model.colorize(thr_color=None)
        assert color.shape == (1, 32, 32, 3)

    def test_colorize_with_zero_thr(self, liaw_model):
        """thr_color=0.0 (a falsy value) should be used, not replaced by default."""
        thr_zero = np.zeros((1, 1, 1))
        # All u values should be > 0 threshold → all color_u
        color = liaw_model.colorize(thr_color=thr_zero)
        assert color.shape == (1, 32, 32, 3)

    def test_colorize_with_multi_batch_thr(self):
        """Multi-batch thr_color array should not raise."""
        from lpf.models import LiawModel
        from lpf.initializers import LiawInitializer

        batch_size = 3
        params = np.tile(
            [1e-3, 1e-2, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01],
            (batch_size, 1)
        ).astype(np.float32)

        init_pts = np.array([[[16, 16]]] * batch_size, dtype=np.uint32)
        init_states = np.array([[0.5, 0.5]] * batch_size, dtype=np.float32)
        initializer = LiawInitializer(init_states=init_states, init_pts=init_pts)

        model = LiawModel(
            initializer=initializer,
            params=params,
            width=32, height=32, dx=0.1,
            device="cpu",
        )
        model.initialize()

        thr = 0.5 * np.ones((batch_size, 1, 1))
        color = model.colorize(thr_color=thr)
        assert color.shape == (batch_size, 32, 32, 3)
