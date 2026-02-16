"""JaxModule.is_array() and clear_memory() must not crash.

Before fix:
- is_array() referenced jnp.generic which doesn't exist in JAX
- clear_memory() called self._module.clear_backends() and
  self._module.numpy which don't exist on jax.numpy
"""

import numpy as np
import pytest


jax = pytest.importorskip("jax", reason="JAX not installed")


class TestJaxModuleAPI:

    @pytest.fixture(autouse=True)
    def setup(self):
        from lpf.array.module import JaxModule
        self.am = JaxModule("cpu", 0)

    def test_is_array_with_jax_array(self):
        """is_array() should return True for a JAX array."""
        arr = self.am.array([1.0, 2.0, 3.0])
        assert self.am.is_array(arr), \
            "is_array should return True for JAX arrays"

    def test_is_array_with_numpy_array(self):
        """is_array() should return False for a plain NumPy array."""
        arr = np.array([1.0, 2.0, 3.0])
        assert not self.am.is_array(arr), \
            "is_array should return False for NumPy arrays"

    def test_is_array_with_scalar(self):
        """is_array() should return False for a Python scalar."""
        assert not self.am.is_array(42)

    def test_clear_memory_does_not_crash(self):
        """clear_memory() should run without AttributeError."""
        self.am.clear_memory()
