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

    def test_set_returns_new_array(self):
        """JAX am.set() returns a new array (functional update).

        Callers that ignore the return value will silently get stale data.
        This test documents the known limitation: code using am.set()
        must use the return value when running on JAX.
        """
        arr = self.am.zeros((2, 3))
        result = self.am.set(arr, (0, 0), 99.0)
        # JAX: arr is unchanged, result has the update
        assert float(result[0, 0]) == 99.0
        assert float(arr[0, 0]) == 0.0, \
            "JAX arrays are immutable; am.set() must not mutate in-place"

    def test_set_return_value_required_for_correctness(self):
        """Demonstrates that ignoring am.set() return silently loses updates.

        This is the root cause of C2: twocomponentmodel.pdefunc() calls
        am.set(dydt_mesh, ...) without using the return value, which means
        dydt_mesh stays zero on JAX.
        """
        arr = self.am.zeros((3,))
        # Simulates the pattern in twocomponentmodel.py — return value ignored
        self.am.set(arr, 0, 1.0)  # return value discarded
        self.am.set(arr, 1, 2.0)
        # arr is still all zeros
        assert float(arr[0]) == 0.0
        assert float(arr[1]) == 0.0
