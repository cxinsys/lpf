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
    - Superior stability for stiff problems
    - Faster convergence than fixed-step methods
    - Built-in error control
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "AdaptiveRKF45Solver"
        
        # RKF45 Butcher tableau coefficients
        self.a = np.array([
            [0, 0, 0, 0, 0, 0],
            [1/4, 0, 0, 0, 0, 0],
            [3/32, 9/32, 0, 0, 0, 0],
            [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
            [439/216, -8, 3680/513, -845/4104, 0, 0],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]
        ])
        
        self.b4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])  # 4th order
        self.b5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])  # 5th order
        
        self.c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
        
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
        """Perform one adaptive step using RKF45 method."""
        if self._dt_current is None:
            self._dt_current = dt
            
        max_attempts = 10  # Prevent infinite loops
        attempt = 0
        
        while attempt < max_attempts:
            # Compute RK stages
            k = self._compute_rk_stages(model, t, self._dt_current, y_mesh)
            
            # Compute 4th and 5th order solutions
            y4 = y_mesh + self._dt_current * np.sum(self.b4[:, np.newaxis, ...] * k, axis=0)
            y5 = y_mesh + self._dt_current * np.sum(self.b5[:, np.newaxis, ...] * k, axis=0)
            
            # Estimate local truncation error
            error = np.abs(y5 - y4)
            max_error = np.max(error)
            
            # Compute optimal step size
            if max_error > 0:
                dt_optimal = self._dt_current * self.safety_factor * (self.tolerance / max_error) ** 0.2
                dt_optimal = np.clip(dt_optimal, 
                                     self._dt_current * self.min_factor,
                                     self._dt_current * self.max_factor)
            else:
                dt_optimal = self._dt_current * self.max_factor
            
            # Accept or reject step
            if max_error <= self.tolerance or self._dt_current <= dt * 1e-10:
                # Accept step - return 4th order solution
                self._dt_current = min(dt_optimal, dt)  # Don't exceed original dt
                self._step_count += 1
                return y4 - y_mesh
            else:
                # Reject step and try smaller step size
                self._dt_current = dt_optimal
                self._rejected_steps += 1
                attempt += 1
                
        # If we reach here, use the last computed solution anyway
        print(f"Warning: Maximum step size reduction attempts reached at t={t}")
        return y4 - y_mesh
    
    def _compute_rk_stages(self, model, t, dt, y_mesh):
        """Compute the 6 RK stages for RKF45."""
        k = np.zeros((6, *y_mesh.shape))
        
        k[0] = model.pdefunc(t, y_mesh)
        
        for i in range(1, 6):
            y_temp = y_mesh + dt * np.sum(self.a[i, :i, np.newaxis, ...] * k[:i], axis=0)
            k[i] = model.pdefunc(t + self.c[i] * dt, y_temp)
            
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