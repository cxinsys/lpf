"""Shared fixtures for regression tests."""

import pytest
import numpy as np


@pytest.fixture
def liaw_params():
    """Minimal Liaw model parameters (batch_size=1, 8 params).
    [Du, Dv, ru, rv, k, su, sv, mu]
    """
    return np.array([[1e-3, 1e-2, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01]],
                    dtype=np.float32)


@pytest.fixture
def liaw_model(liaw_params):
    """A fully initialized Liaw model ready for solving."""
    from lpf.models import LiawModel
    from lpf.initializers import LiawInitializer

    init_pts = np.array([[[16, 16]]], dtype=np.uint32)
    init_states = np.array([[0.5, 0.5]], dtype=np.float32)
    initializer = LiawInitializer(init_states=init_states, init_pts=init_pts)

    model = LiawModel(
        initializer=initializer,
        params=liaw_params,
        width=32,
        height=32,
        dx=0.1,
        device="cpu",
    )
    model.initialize()
    return model
