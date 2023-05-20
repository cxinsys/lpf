from lpf.solvers import EulerSolver
from lpf.solvers import HeunSolver
from lpf.solvers import RungeKuttaSolver


class SolverFactory:
    
    @staticmethod
    def create(name, *args, **kwargs):
        _name = name.lower()

        if "euler" in _name:
            return EulerSolver(*args, **kwargs)
        elif "heun" in _name:
            return HeunSolver(*args, **kwargs)
        elif "rk45" in _name or "rungekutta" in _name:
            return RungeKuttaSolver(*args, **kwargs)
        
        raise ValueError("%s is not a supported solver."%(name))