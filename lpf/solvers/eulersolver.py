from lpf.solvers.solver import Solver


class EulerSolver(Solver):
    """Euler method solver.

    Automatically uses fused CUDA kernels when the model is on a CuPy
    device (``device="cuda:*"``).  No manual backend selection needed.
    """

    def __init__(self, *args, fast_math=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "EulerSolver"
        self._fast_math = fast_math
        self._cuda_impl = None

    def _get_cuda(self):
        if self._cuda_impl is None:
            from lpf.solvers.cueulersolver import CuEulerSolver
            self._cuda_impl = CuEulerSolver(fast_math=self._fast_math)
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
        dydt = model.pdefunc(t, y_mesh)
        return dydt * dt
