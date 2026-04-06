"""Mixin that provides CuPy-accelerated pdefunc evaluation for adaptive solvers."""


class CupyPdefuncMixin:
    """Shared CuPy kernel acceleration for adaptive solvers.

    Provides _init_cuda() and _call_pdefunc() that transparently use
    fused CUDA pdefunc kernels when the model runs on a CuPy device,
    falling back to model.pdefunc().copy() otherwise.

    Requires self._km and self._fast_math to be set by the subclass.
    """

    def _init_cuda(self, model):
        import numpy as np
        from lpf.kernels.compiler import get_kernel_manager
        if (self._km is None
                or self._km._model_name != model.name
                or self._km._dtype != np.dtype(model.dtype)):
            self._km = get_kernel_manager(model.name, model.dtype, self._fast_math)

    def _call_pdefunc(self, model, t, y_mesh):
        from lpf.array.module import CupyModule
        if self._km is not None and isinstance(model.am, CupyModule):
            import cupy as cp
            out = cp.empty_like(y_mesh)
            self._km.launch_pdefunc(
                y_mesh, out, model.params,
                model.batch_size, model.height, model.width,
                1.0 / (model.dx ** 2))
            return out
        # pdefunc() returns an internal buffer that is overwritten on the
        # next call, so the result must be copied. Use clone() for torch
        # tensors and copy() for numpy/cupy arrays.
        out = model.pdefunc(t, y_mesh)
        return out.clone() if hasattr(out, "clone") else out.copy()

    def _maybe_init_cuda(self, model):
        """Init CUDA kernels only for CuPy models."""
        from lpf.array.module import CupyModule
        if isinstance(model.am, CupyModule):
            self._init_cuda(model)
