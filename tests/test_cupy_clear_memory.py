"""CupyModule.clear_memory() must use the context manager variable,
not self._module.dev which doesn't exist.

Before fix: self._module.dev.synchronize() raised AttributeError because
cupy has no .dev attribute. Should use the `dev` variable from
`with self._module.cuda.Device(...) as dev:`.
"""

import numpy as np
import pytest


def _cupy_gpu_available():
    try:
        import cupy as cp
        a = cp.array([1.0, 2.0])
        float(cp.mean(a))  # triggers actual GPU kernel compilation (needs NVRTC)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _cupy_gpu_available(), reason="CuPy GPU runtime not available"
)


class TestCupyModuleClearMemory:

    @pytest.fixture(autouse=True)
    def setup(self):
        from lpf.array.module import CupyModule
        self.am = CupyModule("gpu", 0)

    def test_clear_memory_does_not_crash(self):
        """clear_memory() should execute without AttributeError."""
        arr = self.am.array(np.random.randn(100, 100).astype(np.float32))
        del arr

        self.am.clear_memory()
