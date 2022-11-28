from lpf.solvers import EulerSolver
from lpf.solvers import RungeKuttaSolver


class SolverFactory:
    
    @staticmethod
    def create(name, *args, **kwargs):
        _name = name.lower()

        if "euler" in _name:
            return EulerSolver(*args, **kwargs)
        elif _name in ["rungekutta", "rk45"]:
            return RungeKuttaSolver(*args, **kwargs)
        
        raise ValueError("%s is not a supported solver."%(name))