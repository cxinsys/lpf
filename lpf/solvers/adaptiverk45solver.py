from lpf.solvers.solver import Solver
import numpy as np


class AdaptiveRKF45Solver(Solver):
    """Adaptive Runge-Kutta-Fehlberg 4(5) method with automatic step size control.

    This solver uses embedded Runge-Kutta formulas to estimate local error
    and automatically adjusts step size for optimal efficiency and stability.
    The 4th order solution is used for stepping, while the 5th order solution
    provides error estimation.

    Features:
    - Automatic step size adaptation
    - Faster convergence than fixed-step methods
    - Built-in error control
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "AdaptiveRKF45Solver"

        # RKF45 Butcher tableau coefficients (stored as plain floats for backend compatibility)
        self._a = [
            [0, 0, 0, 0, 0, 0],
            [1/4, 0, 0, 0, 0, 0],
            [3/32, 9/32, 0, 0, 0, 0],
            [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
            [439/216, -8, 3680/513, -845/4104, 0, 0],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]
        ]

        self._b4 = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]  # 4th order
        self._b5 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]  # 5th order

        self._c = [0, 1/4, 3/8, 12/13, 1, 1/2]

        # Adaptive parameters
        self.safety_factor = 0.9
        self.min_factor = 0.1
        self.max_factor = 5.0
        self.tolerance = 1e-6

        # Step size history for stability
        self._dt_current = None
        self._step_count = 0
        self._rejected_steps = 0

    def step(self, model, t, dt, y_mesh):
        """Perform one adaptive step, sub-stepping to cover the full dt interval."""
        if self._dt_current is None:
            self._dt_current = dt

        t_local = 0.0
        delta_y_total = model.am.zeros(y_mesh.shape, dtype=y_mesh.dtype)
        y_current = y_mesh

        while t_local < dt:
            # Don't overshoot the target time
            h = min(self._dt_current, dt - t_local)

            max_attempts = 10
            attempt = 0

            while attempt < max_attempts:
                # Compute RK stages
                k = self._compute_rk_stages(model, t + t_local, h, y_current)

                # Compute 4th and 5th order solutions
                y4 = y_current
                y5 = y_current
                for s in range(6):
                    y4 = y4 + h * self._b4[s] * k[s]
                    y5 = y5 + h * self._b5[s] * k[s]

                # Estimate local truncation error
                error = model.am.abs(y5 - y4)
                max_error = float(model.am.get(error).max())

                # Compute optimal step size
                if max_error > 0:
                    dt_optimal = h * self.safety_factor * (self.tolerance / max_error) ** 0.2
                    dt_optimal = max(h * self.min_factor, min(h * self.max_factor, dt_optimal))
                else:
                    dt_optimal = h * self.max_factor

                # Accept or reject step
                if max_error <= self.tolerance or h <= dt * 1e-10:
                    # Accept step
                    delta_y_total = delta_y_total + (y4 - y_current)
                    y_current = y4
                    t_local += h
                    self._dt_current = dt_optimal
                    self._step_count += 1
                    break
                else:
                    # Reject step and try smaller step size
                    self._dt_current = dt_optimal
                    h = min(dt_optimal, dt - t_local)
                    self._rejected_steps += 1
                    attempt += 1
            else:
                # Max attempts reached, accept the last computed step
                import warnings
                warnings.warn(f"AdaptiveRKF45: max attempts reached at t={t + t_local}")
                delta_y_total = delta_y_total + (y4 - y_current)
                y_current = y4
                t_local += h

        return delta_y_total

    def _compute_rk_stages(self, model, t, dt, y_mesh):
        """Compute the 6 RK stages for RKF45."""
        k = [None] * 6

        k[0] = model.pdefunc(t, y_mesh)

        for i in range(1, 6):
            y_temp = y_mesh
            for j in range(i):
                if self._a[i][j] != 0:
                    y_temp = y_temp + dt * self._a[i][j] * k[j]
            k[i] = model.pdefunc(t + self._c[i] * dt, y_temp)

        return k

    def get_step_statistics(self):
        """Return statistics about step size adaptation."""
        return {
            'total_steps': self._step_count,
            'rejected_steps': self._rejected_steps,
            'current_dt': self._dt_current,
            'rejection_rate': self._rejected_steps / max(1, self._step_count + self._rejected_steps)
        }

    def reset_adaptation(self):
        """Reset adaptive parameters for new solve."""
        self._dt_current = None
        self._step_count = 0
        self._rejected_steps = 0
