"""parse_device() must format error messages correctly and use `is None`.

Regression tests for:
- 7-1: Exception constructors passed tuples instead of formatted strings
- 7-2: `== None` instead of `is None` in device parsing
"""

import pytest
from lpf.array.module import parse_device


class TestExceptionMessages:
    """Exceptions should contain readable messages, not tuples."""

    def test_illegal_device_runtime_error_message(self):
        with pytest.raises(RuntimeError, match="Illegal device"):
            parse_device("a:b:c:d")

    def test_illegal_device_value_error_message(self):
        with pytest.raises(ValueError, match="Illegal device"):
            parse_device("nonexistent_device")


class TestDeviceParsing:
    """Device string parsing and None-backend defaults."""

    def test_none_returns_numpy_cpu(self):
        assert parse_device(None) == ("numpy", "cpu", 0)

    def test_cpu_returns_numpy(self):
        assert parse_device("cpu") == ("numpy", "cpu", 0)

    def test_cupy_gpu_with_id(self):
        backend, device, did = parse_device("cupy:0")
        assert backend == "cupy"
        assert device == "gpu"
        assert did == 0

    def test_torch_cpu(self):
        backend, device, _ = parse_device("torch:cpu")
        assert backend == "torch"
        assert device == "cpu"

    def test_jax_cpu(self):
        backend, device, _ = parse_device("jax:cpu")
        assert backend == "jax"
        assert device == "cpu"
