"""parse_params() must work correctly as a @classmethod.

Regression test for Section 4 issue:
- All parse_params methods had `self` as first parameter instead of `cls`.
  While Python doesn't enforce the name, `self` signals instance method
  semantics and is misleading. The fix ensures `cls` is used.

This test verifies parse_params can be called as a classmethod (on the class
itself, without an instance) and produces correct results.
"""

import numpy as np
import pytest


class TestParseParamsClassmethod:
    """parse_params() should be callable on the class directly."""

    def _make_liaw_dict(self):
        return {
            "Du": 1e-3, "Dv": 1e-2,
            "ru": 1.0, "rv": 1.0,
            "k": 0.01, "su": 0.01, "sv": 0.01, "mu": 0.01,
        }

    def _make_grayscott_dict(self):
        return {
            "Du": 2e-5, "Dv": 1e-5,
            "F": 0.04, "k": 0.06,
        }

    def _make_schnakenberg_dict(self):
        return {
            "Du": 1e-3, "Dv": 1e-2,
            "rho": 1.0, "su": 0.5, "sv": 0.5, "mu": 0.1,
        }

    def _make_gierer_meinhardt_dict(self):
        return {
            "Du": 1e-3, "Dv": 1e-2,
            "ru": 1.0, "rv": 1.0, "mu": 0.5, "nu": 0.3,
        }

    def test_liaw_parse_params_as_classmethod(self):
        from lpf.models import LiawModel
        model_dict = self._make_liaw_dict()
        params = LiawModel.parse_params([model_dict])
        assert params.shape == (1, 8)
        np.testing.assert_allclose(params[0, 0], 1e-3)
        np.testing.assert_allclose(params[0, 1], 1e-2)

    def test_grayscott_parse_params_as_classmethod(self):
        from lpf.models.grayscottmodel import GrayScottModel
        model_dict = self._make_grayscott_dict()
        params = GrayScottModel.parse_params([model_dict])
        assert params.shape == (1, 4)
        np.testing.assert_allclose(params[0, 0], 2e-5)

    def test_schnakenberg_parse_params_as_classmethod(self):
        from lpf.models.schnakenbergmodel import SchnakenbergModel
        model_dict = self._make_schnakenberg_dict()
        params = SchnakenbergModel.parse_params([model_dict])
        assert params.shape == (1, 6)
        np.testing.assert_allclose(params[0, 2], 1.0)  # rho

    def test_gierer_meinhardt_parse_params_as_classmethod(self):
        from lpf.models.gierermeinhardtmodel import GiererMeinhardtModel
        model_dict = self._make_gierer_meinhardt_dict()
        params = GiererMeinhardtModel.parse_params([model_dict])
        assert params.shape == (1, 6)
        np.testing.assert_allclose(params[0, 4], 0.5)  # mu

    def test_parse_params_batch(self):
        """parse_params should handle multiple model dicts."""
        from lpf.models import LiawModel
        dicts = [self._make_liaw_dict(), self._make_liaw_dict()]
        params = LiawModel.parse_params(dicts)
        assert params.shape == (2, 8)

    def test_parse_params_single_dict(self):
        """parse_params should accept a single dict (not wrapped in list)."""
        from lpf.models import LiawModel
        model_dict = self._make_liaw_dict()
        params = LiawModel.parse_params(model_dict)
        assert params.shape == (1, 8)

    def test_parse_params_dtype_override(self):
        """parse_params should respect dtype parameter."""
        from lpf.models import LiawModel
        model_dict = self._make_liaw_dict()
        params = LiawModel.parse_params([model_dict], dtype=np.float64)
        assert params.dtype == np.float64
