from lpf.solvers.solver import Solver


class RK23Solver(Solver):
    """Bogacki-Shampine RK23 (fixed-step 3rd-order) solver.

    Automatically uses fused CUDA kernels when the model is on a CuPy
    device (``device="cuda:*"``).
    """

    def __init__(self, *args, fast_math=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "RK23Solver"
        self._fast_math = fast_math
        self._cuda_impl = None

    def _get_cuda(self):
        if self._cuda_impl is None:
            from lpf.solvers.curk23solver import CuRK23Solver
            self._cuda_impl = CuRK23Solver(fast_math=self._fast_math)
        return self._cuda_impl

    @staticmethod
    def _is_cuda(model):
        from lpf.array.module import CupyModule, TorchModule
        if model is None:
            return False
        if isinstance(model.am, CupyModule):
            return True
        if isinstance(model.am, TorchModule):
            return hasattr(model.am, "_device") and "cuda" in str(model.am._device)
        return False

    def _forward_to_cuda(self, model, dt, n_iters, rtol, period_output, kwargs):
        """Forward stored params to the Cu* solver."""
        return self._get_cuda().solve(
            model=model,
            dt=dt if dt is not None else self._dt,
            n_iters=n_iters if n_iters is not None else self._n_iters,
            rtol=rtol if rtol is not None else self._rtol,
            period_output=(period_output if period_output is not None
                           else self._period_output),
            **kwargs)

    def solve(self, model=None, dt=None, n_iters=None, rtol=None,
              period_output=None, **kwargs):
        if model is None:
            model = self._model
        if self._is_cuda(model):
            return self._forward_to_cuda(
                model, dt, n_iters, rtol, period_output, kwargs)
        return super().solve(model=model, dt=dt, n_iters=n_iters,
                             rtol=rtol, period_output=period_output, **kwargs)

    def step(self, model, t, dt, y_mesh):
        k1 = model.pdefunc(t, y_mesh)
        k2 = model.pdefunc(t + 0.5 * dt, y_mesh + 0.5 * dt * k1)
        k3 = model.pdefunc(t + 0.75 * dt, y_mesh + 0.75 * dt * k2)
        return dt * ((2.0 / 9.0) * k1 + (1.0 / 3.0) * k2 + (4.0 / 9.0) * k3)
