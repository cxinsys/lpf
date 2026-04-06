from lpf.solvers.solver import Solver
from lpf.solvers._cupy_pdefunc_mixin import CupyPdefuncMixin
import numpy as np


class AdaptiveRKF45Solver(CupyPdefuncMixin, Solver):
    """Adaptive Runge-Kutta-Fehlberg 4(5) method with automatic step size control.

    Uses embedded RK formulas to estimate local error and automatically
    adjusts step size.  The 4th-order solution advances the state while
    the 5th-order solution provides error estimation.

    Automatically uses fused CUDA pdefunc kernels when the model is on
    a CuPy device (``device="cuda:*"``).
    """

    def __init__(self, *args, fast_math=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "AdaptiveRKF45Solver"

        # RKF45 Butcher tableau
        self._a = [
            [0, 0, 0, 0, 0, 0],
            [1/4, 0, 0, 0, 0, 0],
            [3/32, 9/32, 0, 0, 0, 0],
            [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
            [439/216, -8, 3680/513, -845/4104, 0, 0],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]
        ]
        self._b4 = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
        self._b5 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
        self._c = [0, 1/4, 3/8, 12/13, 1, 1/2]

        # Adaptive control
        self.safety_factor = 0.9
        self.min_factor = 0.1
        self.max_factor = 5.0
        self.tolerance = 1e-6

        self._dt_current = None
        self._step_count = 0
        self._rejected_steps = 0

        self._fast_math = fast_math
        self._km = None

    def _make_jax_solver(self):
        from lpf.solvers._jax.solvers import JaxAdaptiveRKF45Solver
        return JaxAdaptiveRKF45Solver(
            tolerance=self.tolerance,
            safety_factor=self.safety_factor,
            min_factor=self.min_factor,
            max_factor=self.max_factor,
        )

    def solve(self, model=None, **kwargs):
        self.reset_adaptation()
        if model is None:
            model = self._model
        self._maybe_init_cuda(model)
        return super().solve(model=model, **kwargs)

    def step(self, model, t, dt, y_mesh):
        if self._dt_current is None:
            self._dt_current = dt

        t_local = 0.0
        delta_y_total = model.am.zeros(y_mesh.shape, dtype=y_mesh.dtype)
        y_current = y_mesh

        while t_local < dt:
            h = min(self._dt_current, dt - t_local)
            max_attempts = 10

            for attempt in range(max_attempts):
                k = self._compute_rk_stages(model, t + t_local, h, y_current)

                # 4th and 5th order solutions
                y4 = y_current
                y5 = y_current
                for s in range(6):
                    if self._b4[s] != 0:
                        y4 = y4 + h * self._b4[s] * k[s]
                    if self._b5[s] != 0:
                        y5 = y5 + h * self._b5[s] * k[s]

                # Error estimate
                error = model.am.abs(y5 - y4)
                max_error = float(model.am.get(error).max())

                # Optimal step size (p=4 -> exponent 1/5)
                if max_error > 0:
                    dt_opt = h * self.safety_factor * (self.tolerance / max_error) ** 0.2
                    dt_opt = max(h * self.min_factor, min(h * self.max_factor, dt_opt))
                else:
                    dt_opt = h * self.max_factor

                if max_error <= self.tolerance or h <= dt * 1e-10:
                    delta_y_total = delta_y_total + (y4 - y_current)
                    y_current = y4
                    t_local += h
                    self._dt_current = dt_opt
                    self._step_count += 1
                    break
                else:
                    self._dt_current = dt_opt
                    h = min(dt_opt, dt - t_local)
                    self._rejected_steps += 1
            else:
                import warnings
                warnings.warn(f"AdaptiveRKF45: max attempts at t={t + t_local:.6e}")
                delta_y_total = delta_y_total + (y4 - y_current)
                y_current = y4
                t_local += h

        return delta_y_total

    def _compute_rk_stages(self, model, t, dt, y_mesh):
        k = [None] * 6
        k[0] = self._call_pdefunc(model, t, y_mesh)

        for i in range(1, 6):
            y_temp = y_mesh
            for j in range(i):
                if self._a[i][j] != 0:
                    y_temp = y_temp + dt * self._a[i][j] * k[j]
            k[i] = self._call_pdefunc(model, t + self._c[i] * dt, y_temp)

        return k

    def get_step_statistics(self):
        return {
            'total_steps': self._step_count,
            'rejected_steps': self._rejected_steps,
            'current_dt': self._dt_current,
            'rejection_rate': self._rejected_steps / max(1, self._step_count + self._rejected_steps)
        }

    def reset_adaptation(self):
        self._dt_current = None
        self._step_count = 0
        self._rejected_steps = 0
