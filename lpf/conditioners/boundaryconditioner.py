from abc import ABC, abstractmethod


class BoundaryConditioner(ABC):
    """
    Abstract base class (interface) for PDE boundary conditions.
    """

    @abstractmethod
    def apply(self, model=None, y_mesh=None, dydt_mesh=None):
        """
        Apply the boundary condition by modifying model's dydt_mesh or state variables (u, v).
        """
        raise NotImplementedError()
