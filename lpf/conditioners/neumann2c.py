from lpf.conditioners import TwoComponentBoundaryConditioner


class Neumann2C(TwoComponentBoundaryConditioner):
    """
    Conditioner for Neumann boundary conditions.

    Typically, it implies 'no flux (derivative = 0)'. In this example, we set
    the boundary cells of model._dydt_mesh to certain values for each state (u, v).
    """

    def __init__(self, val_u=0.0, val_v=0.0):
        """
        val_u: The Neumann boundary condition value for du/dt at the grid boundaries.
        val_v: The Neumann boundary condition value for dv/dt at the grid boundaries.
        """
        super().__init__()
        self._val_u = val_u
        self._val_v = val_v

    def apply(self, model=None, y_mesh=None, dydt_mesh=None):
        """
        Set the top/bottom/left/right edges of dydt_mesh for u and v separately.

        model._dydt_mesh has shape: (n_states, batch_size, height, width).
        For a 2-state system (u, v):
            - index 0 -> u
            - index 1 -> v
        """
        # Set the boundary of u.
        model.am.set(dydt_mesh, (0, slice(None), 0, slice(None)), self.val_u)
        model.am.set(dydt_mesh, (0, slice(None), -1, slice(None)), self.val_u)
        model.am.set(dydt_mesh, (0, slice(None), slice(None), 0), self.val_u)
        model.am.set(dydt_mesh, (0, slice(None), slice(None), -1), self.val_u)

        # Set the boundary of v.
        model.am.set(dydt_mesh, (1, slice(None), 0, slice(None)), self.val_v)
        model.am.set(dydt_mesh, (1, slice(None), -1, slice(None)), self.val_v)
        model.am.set(dydt_mesh, (1, slice(None), slice(None), 0), self.val_v)
        model.am.set(dydt_mesh, (1, slice(None), slice(None), -1), self.val_v)

        return dydt_mesh