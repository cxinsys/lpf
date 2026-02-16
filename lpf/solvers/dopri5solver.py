import warnings
from lpf.solvers.solver import Solver


class DOPRI5Solver(Solver):
    """Dormand-Prince 5th order solver with embedded error estimation.

    DOPRI5 is widely considered one of the most efficient general-purpose
    ODE/PDE solvers. It provides:
    - Excellent stability properties
    - Built-in error control
    - Optimal balance between accuracy and computational cost
    - FSAL (First Same As Last) property for efficiency
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "DOPRI5Solver"

        # DOPRI5 Butcher tableau
        self.a21 = 1/5
        self.a31 = 3/40
        self.a32 = 9/40
        self.a41 = 44/45
        self.a42 = -56/15
        self.a43 = 32/9
        self.a51 = 19372/6561
        self.a52 = -25360/2187
        self.a53 = 64448/6561
        self.a54 = -212/729
        self.a61 = 9017/3168
        self.a62 = -355/33
        self.a63 = 46732/5247
        self.a64 = 49/176
        self.a65 = -5103/18656
        self.a71 = 35/384
        self.a72 = 0
        self.a73 = 500/1113
        self.a74 = 125/192
        self.a75 = -2187/6784
        self.a76 = 11/84

        # 5th order solution coefficients (same as a7*)
        self.b1 = 35/384
        self.b2 = 0
        self.b3 = 500/1113
        self.b4 = 125/192
        self.b5 = -2187/6784
        self.b6 = 11/84
        self.b7 = 0

        # 4th order solution coefficients for error estimation
        self.e1 = 35/384 - 5179/57600
        self.e2 = 0
        self.e3 = 500/1113 - 7571/16695
        self.e4 = 125/192 - 393/640
        self.e5 = -2187/6784 + 92097/339200
        self.e6 = 11/84 - 187/2100
        self.e7 = -1/40

        # Adaptive control parameters
        self.tolerance = 1e-6
        self.safety = 0.9
        self.max_factor = 5.0
        self.min_factor = 0.2

        # FSAL property: store last function evaluation
        self._k1_next = None
        self._adaptive_dt = None

    def step(self, model, t, dt, y_mesh):
        """Perform adaptive DOPRI5 step, sub-stepping to cover the full dt interval."""
        if self._adaptive_dt is None:
            self._adaptive_dt = dt

        t_local = 0.0
        delta_y_total = model.am.zeros(y_mesh.shape, dtype=y_mesh.dtype)
        y_current = y_mesh

        while t_local < dt:
            # Don't overshoot the target time
            h = min(self._adaptive_dt, dt - t_local)

            # Use FSAL property if available
            if self._k1_next is not None:
                k1 = self._k1_next
            else:
                k1 = model.pdefunc(t + t_local, y_current)

            max_attempts = 8
            accepted = False

            for attempt in range(max_attempts):
                # Compute intermediate stages
                k2 = model.pdefunc(t + t_local + h/5,
                                   y_current + h * self.a21 * k1)

                k3 = model.pdefunc(t + t_local + 3*h/10,
                                   y_current + h * (self.a31 * k1 + self.a32 * k2))

                k4 = model.pdefunc(t + t_local + 4*h/5,
                                   y_current + h * (self.a41 * k1 + self.a42 * k2 + self.a43 * k3))

                k5 = model.pdefunc(t + t_local + 8*h/9,
                                   y_current + h * (self.a51 * k1 + self.a52 * k2 +
                                                    self.a53 * k3 + self.a54 * k4))

                k6 = model.pdefunc(t + t_local + h,
                                   y_current + h * (self.a61 * k1 + self.a62 * k2 +
                                                    self.a63 * k3 + self.a64 * k4 + self.a65 * k5))

                # 5th order solution
                y_new = y_current + h * (self.b1 * k1 + self.b2 * k2 + self.b3 * k3 +
                                         self.b4 * k4 + self.b5 * k5 + self.b6 * k6)

                # For FSAL: k7 becomes k1 for next step
                k7 = model.pdefunc(t + t_local + h, y_new)

                # Error estimation using 4th order embedded solution
                error = h * (self.e1 * k1 + self.e2 * k2 + self.e3 * k3 +
                             self.e4 * k4 + self.e5 * k5 + self.e6 * k6 + self.e7 * k7)

                # Compute error norm (use model.am for backend compatibility)
                error_norm = float(model.am.get(
                    model.am.sqrt(model.am.mean(error**2))
                ))

                # Step size control
                if error_norm <= self.tolerance or h <= dt * 1e-12:
                    # Accept step
                    self._k1_next = k7  # FSAL property

                    # Update adaptive step size for next step
                    if error_norm > 0:
                        factor = self.safety * (self.tolerance / error_norm) ** 0.2
                        factor = max(self.min_factor, min(self.max_factor, factor))
                        self._adaptive_dt = h * factor

                    delta_y_total = delta_y_total + (y_new - y_current)
                    y_current = y_new
                    t_local += h
                    accepted = True
                    break
                else:
                    # Reject step, reduce step size
                    factor = max(self.min_factor,
                                 self.safety * (self.tolerance / error_norm) ** 0.25)
                    h = h * factor
                    h = min(h, dt - t_local)  # Don't overshoot
                    self._k1_next = None  # Invalidate FSAL after rejection

            if not accepted:
                # Fallback: accept the last computed step
                warnings.warn(f"DOPRI5: max attempts reached at t={t + t_local}")
                self._k1_next = k7
                delta_y_total = delta_y_total + (y_new - y_current)
                y_current = y_new
                t_local += h

        return delta_y_total

    def reset_fsal(self):
        """Reset FSAL state for new solve."""
        self._k1_next = None
        self._adaptive_dt = None
