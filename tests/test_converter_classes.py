"""Converter classes must have correct names and be fully functional.

Regression tests for:
- 1-10: Missing comma between "nu" and "u0" caused "nuu0" concatenation
- 4-4: GiererMeinhardtConverter class was named GiererMeinhardtModel
- 4-5: SchnakenbergConverter class was named SchnakenbergModel
"""

import numpy as np
import pytest

from lpf.converters import Converter


class TestGiererMeinhardtConverter:
    """GiererMeinhardtConverter naming, param names, and functionality."""

    def test_old_name_not_importable(self):
        """The old incorrect name 'GiererMeinhardtModel' should not exist."""
        with pytest.raises(ImportError):
            from lpf.converters.gierermeinhardtconverter import GiererMeinhardtModel  # noqa: F401

    def test_class_name_and_inheritance(self):
        from lpf.converters.gierermeinhardtconverter import GiererMeinhardtConverter
        assert GiererMeinhardtConverter.__name__ == "GiererMeinhardtConverter"
        assert issubclass(GiererMeinhardtConverter, Converter)
        assert GiererMeinhardtConverter()._name == "GiererMeinhardtConverter"

    def test_param_names_no_concatenation(self):
        """Before fix: missing comma produced 'nuu0'. Must be 8 separate names."""
        from lpf.converters.gierermeinhardtconverter import GiererMeinhardtConverter
        names = GiererMeinhardtConverter().get_param_names()
        assert names == ["Du", "Dv", "ru", "rv", "mu", "nu", "u0", "v0"]
        assert "nuu0" not in names

    def test_to_params_values(self):
        from lpf.converters.gierermeinhardtconverter import GiererMeinhardtConverter
        converter = GiererMeinhardtConverter()
        dv = np.array([[-3, -2, 0.5, 0.5, -1, -1, 0.5, 0.5, 0, 0]], dtype=np.float32)
        params = converter.to_params(dv)
        assert params.shape == (1, 8)
        np.testing.assert_allclose(params[0, 0], 1e-3)   # Du = 10**-3
        np.testing.assert_allclose(params[0, 5], 0.1)     # nu = 10**-1

    def test_to_init_states_shape(self):
        from lpf.converters.gierermeinhardtconverter import GiererMeinhardtConverter
        init_states = GiererMeinhardtConverter().to_init_states(np.zeros((1, 10), dtype=np.float32))
        assert init_states.shape == (1, 2)


class TestSchnakenbergConverter:
    """SchnakenbergConverter naming and functionality."""

    def test_old_name_not_importable(self):
        with pytest.raises(ImportError):
            from lpf.converters.schnakenbergconverter import SchnakenbergModel  # noqa: F401

    def test_class_name_and_inheritance(self):
        from lpf.converters.schnakenbergconverter import SchnakenbergConverter
        assert SchnakenbergConverter.__name__ == "SchnakenbergConverter"
        assert issubclass(SchnakenbergConverter, Converter)
        assert SchnakenbergConverter()._name == "SchnakenbergConverter"

    def test_param_names(self):
        from lpf.converters.schnakenbergconverter import SchnakenbergConverter
        names = SchnakenbergConverter().get_param_names()
        assert names == ["Du", "Dv", "rho", "su", "sv", "mu", "u0", "v0"]

    def test_to_params_and_init_pts(self):
        from lpf.converters.schnakenbergconverter import SchnakenbergConverter
        converter = SchnakenbergConverter()
        dv = np.array([[0.1]*8 + [10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        assert converter.to_params(dv).shape == (1, 8)
        pts = converter.to_init_pts(dv)
        assert pts.shape == (1, 2, 2)  # batch=1, 2 points, (row, col)
