# from lpf.solvers import EulerSolver
# from lpf.solvers import HeunSolver
# from lpf.solvers import RungeKuttaSolver


# class SolverFactory:
    
#     @staticmethod
#     def create(name, *args, **kwargs):
#         _name = name.lower()

#         if "euler" in _name:
#             return EulerSolver(*args, **kwargs)
#         elif "heun" in _name:
#             return HeunSolver(*args, **kwargs)
#         elif "rk45" in _name or "rungekutta" in _name:
#             return RungeKuttaSolver(*args, **kwargs)
        
#         raise ValueError("%s is not a supported solver."%(name))


from lpf.solvers import EulerSolver
from lpf.solvers import HeunSolver  
from lpf.solvers import RungeKuttaSolver
from lpf.solvers.adaptiverk45solver import AdaptiveRKF45Solver
from lpf.solvers.dopri5solver import DOPRI5Solver
from lpf.solvers.adamsbashforth2solver import AdamsBashforth2Solver
from lpf.solvers.rk23solver import RK23Solver


class SolverFactory:
    
    @staticmethod
    def create(name, *args, **kwargs):
        _name = name.lower().replace('-', '').replace('_', '')

        # Basic solvers
        if "euler" in _name:
            return EulerSolver(*args, **kwargs)
        elif "heun" in _name:
            return HeunSolver(*args, **kwargs)

        # Advanced adaptive solvers (must be checked before "rk45" to avoid substring collision)
        elif "dopri5" in _name or "dormandprince" in _name:
            return DOPRI5Solver(*args, **kwargs)
        elif "adaptiverk45" in _name or "rkf45" in _name or "fehlberg" in _name:
            return AdaptiveRKF45Solver(*args, **kwargs)

        # Classic RK4 (checked after adaptive variants)
        elif "rk45" in _name or "rungekutta" in _name:
            return RungeKuttaSolver(*args, **kwargs)
        
        # Multi-step solvers
        elif "adamsbashforth2" in _name or "ab2" in _name:
            return AdamsBashforth2Solver(*args, **kwargs)
        elif "rk23" in _name or "bogackishampine" in _name:
            return RK23Solver(*args, **kwargs)
        
        raise ValueError(f"{name} is not a supported solver. Available solvers: "
                        "euler, heun, rk45, dopri5, adaptive_rk45, adams_bashforth2, rk23")
    
    @staticmethod
    def get_recommended_solver(problem_type="general"):
        """Get recommended solver based on problem characteristics.
        
        Args:
            problem_type: "general", "stiff", "smooth", "oscillatory"
        
        Returns:
            String name of recommended solver
        """
        recommendations = {
            "general": "dopri5",
            "stiff": "adaptive_rk45", 
            "smooth": "dopri5",
            "oscillatory": "rk23",
            "fast": "dopri5"
        }
        
        return recommendations.get(problem_type, "dopri5")
    
    @staticmethod
    def list_solvers():
        """Return list of available solvers with descriptions."""
        return {
            "euler": "Simple 1st order explicit method",
            "heun": "2nd order predictor-corrector method", 
            "rk45": "Classic 4th order Runge-Kutta",
            "dopri5": "5th order Dormand-Prince (RECOMMENDED)",
            "adaptive_rk45": "Adaptive RKF45 with error control",
            "adams_bashforth2": "2nd order multi-step method",
            "rk23": "3rd order Bogacki-Shampine method"
        }