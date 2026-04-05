"""LiawConverter must correctly convert between decision vectors and parameters.

Tests the full round-trip: decision vector → params/init_states → model.
Also tests ConverterFactory can create converters by name.
"""

import numpy as np
import pytest


class TestLiawConverterRoundtrip:

    def test_to_params_applies_log10(self):
        """Parameters should be 10**dv (log-scale encoding)."""
        from lpf.converters import LiawConverter
        converter = LiawConverter()

        dv = np.array([[-3, -2, 0, 0, -1, 0, 0, 0, 0, 0]], dtype=np.float32)
        params = converter.to_params(dv)

        np.testing.assert_allclose(params[0, 0], 1e-3, rtol=1e-5)  # Du
        np.testing.assert_allclose(params[0, 1], 1e-2, rtol=1e-5)  # Dv
        np.testing.assert_allclose(params[0, 2], 1.0, rtol=1e-5)   # ru

    def test_to_init_states(self):
        from lpf.converters import LiawConverter
        converter = LiawConverter()

        dv = np.array([[0]*8 + [0.5, 0.3] + [0, 0]], dtype=np.float32)
        init_states = converter.to_init_states(dv)

        assert init_states.shape == (1, 2)
        np.testing.assert_allclose(init_states[0, 0], 10**0.5, rtol=1e-5)
        np.testing.assert_allclose(init_states[0, 1], 10**0.3, rtol=1e-5)

    def test_to_init_pts(self):
        from lpf.converters import LiawConverter
        converter = LiawConverter()

        # 10 base params + 4 coords (2 points)
        dv = np.array([[0]*10 + [15.7, 20.3, 25.1, 30.9]], dtype=np.float32)
        pts = converter.to_init_pts(dv)

        assert pts.shape == (1, 2, 2)  # batch=1, 2 points, (row, col)
        assert pts[0, 0, 0] == 15  # int(15.7)
        assert pts[0, 0, 1] == 20  # int(20.3)

    def test_param_names_count(self):
        from lpf.converters import LiawConverter
        converter = LiawConverter()
        names = converter.get_param_names()
        assert len(names) == 10
        assert names[:2] == ["Du", "Dv"]


class TestConverterFactory:

    def test_create_liaw(self):
        from lpf.converters import ConverterFactory, LiawConverter
        converter = ConverterFactory.create("liaw")
        assert isinstance(converter, LiawConverter)

    def test_create_liaw_case_insensitive(self):
        from lpf.converters import ConverterFactory, LiawConverter
        converter = ConverterFactory.create("Liaw")
        assert isinstance(converter, LiawConverter)

    def test_invalid_converter_raises(self):
        from lpf.converters import ConverterFactory
        with pytest.raises((ValueError, KeyError)):
            ConverterFactory.create("nonexistent")
