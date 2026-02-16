"""GiererMeinhardtConverter.get_param_names() must return 8 names.

Before fix: Missing comma between "nu" and "u0" caused Python to
concatenate them into "nuu0", producing 7 names instead of 8.
This caused index misalignment in to_dv() and to_params().
"""

import numpy as np
import pytest

from lpf.converters.gierermeinhardtconverter import GiererMeinhardtModel as GiererMeinhardtConverter


class TestGiererMeinhardtConverter:

    def test_param_names_count(self):
        """get_param_names() should return exactly 8 names."""
        converter = GiererMeinhardtConverter()
        names = converter.get_param_names()
        assert len(names) == 8, \
            f"Expected 8 param names, got {len(names)}: {names}"

    def test_no_concatenated_names(self):
        """No name should be 'nuu0' (the concatenation artifact)."""
        converter = GiererMeinhardtConverter()
        names = converter.get_param_names()
        assert "nuu0" not in names, \
            f"Found concatenated 'nuu0' in names: {names}"

    def test_nu_and_u0_are_separate(self):
        """Both 'nu' and 'u0' should be separate entries."""
        converter = GiererMeinhardtConverter()
        names = converter.get_param_names()
        assert "nu" in names, f"'nu' not found in {names}"
        assert "u0" in names, f"'u0' not found in {names}"

    def test_expected_param_names(self):
        """Full expected list of parameter names."""
        converter = GiererMeinhardtConverter()
        names = converter.get_param_names()
        expected = ["Du", "Dv", "ru", "rv", "mu", "nu", "u0", "v0"]
        assert names == expected, f"Expected {expected}, got {names}"

    def test_to_params_values(self):
        """to_params should map decision vector values to correct param indices."""
        converter = GiererMeinhardtConverter()
        dv = np.array([[-3, -2, 0.5, 0.5, -1, -1, 0.5, 0.5]]).reshape(1, -1)
        params = converter.to_params(dv)

        assert params.shape == (1, 8), f"Expected (1, 8), got {params.shape}"
        # First 6 params are 10**dv[0, i]
        np.testing.assert_allclose(params[0, 0], 10**(-3))   # Du
        np.testing.assert_allclose(params[0, 1], 10**(-2))   # Dv
        np.testing.assert_allclose(params[0, 2], 10**(0.5))  # ru
        np.testing.assert_allclose(params[0, 3], 10**(0.5))  # rv
        np.testing.assert_allclose(params[0, 4], 10**(-1))   # mu
        np.testing.assert_allclose(params[0, 5], 10**(-1))   # nu
