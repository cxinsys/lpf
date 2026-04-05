"""Diploidy model must use `is` for identity check, not id() comparison.

Regression tests for:
- 4-7: Error message had "parms" typo (should be "params")
- 7-10: Used id(x) == id(y) instead of `x is y`
"""

import numpy as np
import pytest


def _make_model():
    from lpf.models import LiawModel
    from lpf.initializers import LiawInitializer

    init_pts = np.array([[[8, 8]]], dtype=np.uint32)
    init_states = np.array([[0.5, 0.5]], dtype=np.float32)
    initializer = LiawInitializer(init_states=init_states, init_pts=init_pts)
    params = np.array([[1e-3, 1e-2, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01]],
                      dtype=np.float32)
    return LiawModel(
        initializer=initializer, params=params,
        width=16, height=16, dx=0.1, device="cpu",
    )


class TestDiploidyIdentityCheck:

    def test_same_object_raises(self):
        """Passing the same object for both parents should raise ValueError."""
        from lpf.models.diploidy import Diploidy

        model = _make_model()

        with pytest.raises(ValueError, match="must be different objects"):
            class ConcreteDiploid(Diploidy, type(model)):
                def pdefunc(self, **kwargs):
                    raise NotImplementedError

            ConcreteDiploid(
                paternal_model=model,
                maternal_model=model,
                initializer=model.initializer,
                params=model._params,
                width=16, height=16, dx=0.1, device="cpu",
            )

    def test_dtype_mismatch_message(self):
        """Error message should say 'params', not 'parms'."""
        from lpf.models import diploidy
        source = open(diploidy.__file__).read()
        assert "maternal_model.params" in source
        assert "maternal_model.parms" not in source
