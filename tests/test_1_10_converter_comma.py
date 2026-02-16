"""Test 1-10: GiererMeinhardtConverter.get_param_names() must return 8 names.

Before fix: Missing comma between "nu" and "u0" caused Python to
concatenate them into "nuu0", producing 7 names instead of 8.
This caused index misalignment in to_dv() and to_params().
"""

import pytest

from lpf.converters.gierermeinhardtconverter import GiererMeinhardtModel as GiererMeinhardtConverter


class TestGiererMeinhardtConverterComma:

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

    def test_to_params_shape(self):
        """to_params should produce (1, 8) array matching 8 parameters."""
        import numpy as np

        converter = GiererMeinhardtConverter()
        # Decision vector with 8 values (log10 scale for first 6, direct for last 2)
        dv = np.array([[-3, -2, 0.5, 0.5, -1, -1, 0.5, 0.5]]).reshape(1, -1)
        params = converter.to_params(dv)
        assert params.shape == (1, 8), f"Expected (1, 8), got {params.shape}"
