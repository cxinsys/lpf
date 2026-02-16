"""All array module backends must provide mean() and sqrt().

Before fix: mean()/sqrt() only existed on TorchModule, so DOPRI5Solver
crashed with AttributeError on NumPy/CuPy/JAX backends.
"""

import numpy as np
import pytest

from lpf.array.module import NumpyModule


class TestNumpyModuleMeanSqrt:

    def setup_method(self):
        self.am = NumpyModule("cpu", 0)

    def test_has_mean(self):
        assert hasattr(self.am, "mean"), "NumpyModule must have mean()"

    def test_has_sqrt(self):
        assert hasattr(self.am, "sqrt"), "NumpyModule must have sqrt()"

    def test_mean_returns_correct_value(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = self.am.mean(arr)
        assert np.isclose(result, 2.5)

    def test_sqrt_returns_correct_value(self):
        arr = np.array([4.0, 9.0, 16.0])
        result = self.am.sqrt(arr)
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0])

    def test_error_norm_pattern(self):
        """The exact pattern used in DOPRI5Solver: am.sqrt(am.mean(error**2))"""
        error = np.array([1.0, -2.0, 3.0])
        result = self.am.sqrt(self.am.mean(error**2))
        expected = np.sqrt(np.mean(error**2))
        assert np.isclose(result, expected)


class TestTorchModuleMeanSqrt:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch = pytest.importorskip("torch")
        from lpf.array.module import TorchModule
        self.am = TorchModule("cpu", 0)
        self.torch = torch

    def test_mean_returns_correct_value(self):
        arr = self.torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = self.am.mean(arr)
        assert self.torch.isclose(result, self.torch.tensor(2.5))

    def test_sqrt_returns_correct_value(self):
        arr = self.torch.tensor([4.0, 9.0, 16.0])
        result = self.am.sqrt(arr)
        expected = self.torch.tensor([2.0, 3.0, 4.0])
        assert self.torch.allclose(result, expected)

    def test_error_norm_pattern(self):
        error = self.torch.tensor([1.0, -2.0, 3.0])
        result = self.am.sqrt(self.am.mean(error**2))
        expected = self.torch.sqrt(self.torch.mean(error**2))
        assert self.torch.isclose(result, expected)


def _cupy_gpu_available():
    try:
        import cupy as cp
        a = cp.array([1.0, 2.0])
        float(cp.mean(a))  # triggers actual GPU kernel compilation (needs NVRTC)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _cupy_gpu_available(), reason="CuPy GPU runtime not available")
class TestCupyModuleMeanSqrt:

    @pytest.fixture(autouse=True)
    def setup(self):
        import cupy as cp
        from lpf.array.module import CupyModule
        self.am = CupyModule("gpu", 0)
        self.cp = cp

    def test_has_mean(self):
        assert hasattr(self.am, "mean")

    def test_has_sqrt(self):
        assert hasattr(self.am, "sqrt")

    def test_error_norm_pattern(self):
        error = self.cp.array([1.0, -2.0, 3.0])
        result = self.am.sqrt(self.am.mean(error**2))
        expected = float(self.cp.sqrt(self.cp.mean(error**2)))
        assert np.isclose(float(result), expected)
