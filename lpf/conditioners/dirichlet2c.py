from lpf.conditioners import TwoComponentBoundaryConditioner


class Dirichlet2C(TwoComponentBoundaryConditioner):
    """
    Conditioner for Dirichlet boundary conditions.

    This means directly fixing the value of the state variables themselves.
    If necessary, you can directly modify model.u, model.v, etc.
    """

    def __init__(self, val_u=0.0, val_v=0.0):
        """
        Example Dirichlet boundary values for u and v, respectively.
        """
        super().__init__()
        self._val_u = val_u
        self._val_v = val_v

    def apply(self, model, y_mesh=None, dydt_mesh=None):
        """
        Fix the top/bottom/left/right edges of model.u, model.v to _val_u, _val_v respectively.
        """
        with model.am:
            # # Top edge
            # y_mesh[:, 0, :] = self._val_u
            # y_mesh[:, 0, :] = self._val_v
            #
            # # Bottom edge
            # y_mesh[:, -1, :] = self._val_u
            # y_mesh[:, -1, :] = self._val_v
            #
            # # Left edge
            # y_mesh[:, :, 0] = self._val_u
            # y_mesh[:, :, 0] = self._val_v
            #
            # # Right edge
            # y_mesh[:, :, -1] = self._val_u
            # y_mesh[:, :, -1] = self._val_v

            # Set the boundary of u.
            model.am.set(y_mesh, (0, slice(None), 0, slice(None)), self.val_u)
            model.am.set(y_mesh, (0, slice(None), -1, slice(None)), self.val_u)
            model.am.set(y_mesh, (0, slice(None), slice(None), 0), self.val_u)
            model.am.set(y_mesh, (0, slice(None), slice(None), -1), self.val_u)

            # Set the boundary of v.
            model.am.set(y_mesh, (1, slice(None), 0, slice(None)), self.val_v)
            model.am.set(y_mesh, (1, slice(None), -1, slice(None)), self.val_v)
            model.am.set(y_mesh, (1, slice(None), slice(None), 0), self.val_v)
            model.am.set(y_mesh, (1, slice(None), slice(None), -1), self.val_v)

            return y_mesh
