from lpf.conditioners import BoundaryConditioner


class TwoComponentBoundaryConditioner(BoundaryConditioner):
    def __init__(self, val_u=None, val_v=None):

        self._val_u = val_u
        self._val_v = val_v

    @property
    def val_u(self):
        return self._val_u

    @val_u.setter
    def val_u(self, val):
        self._val_u = val

    @property
    def val_v(self):
        return self._val_v

    @val_v.setter
    def val_v(self, val):
        self._val_v = val

