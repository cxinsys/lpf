
import numpy as np


class LiawConverter:

    def to_params(self, x, params=None):
        """
        Args:
            x: Decision vector of PyGMO
        """
        if params is None:
            params = np.zeros((10,), dtype=np.float64)

        Du = 10 ** x[0]
        Dv = 10 ** x[1]
        ru = 10 ** x[2]
        rv = 10 ** x[3]
        k = 10 ** x[4]
        su = 10 ** x[5]
        sv = 10 ** x[6]
        mu = 10 ** x[7]

        params[0] = Du
        params[1] = Dv
        params[2] = ru
        params[3] = rv
        params[4] = k
        params[5] = su
        params[6] = sv
        params[7] = mu

        return params

    def to_init_states(self, x, init_states=None):
        if init_states is None:
            init_states = np.zeros((2,), dtype=np.float64)

        init_states[0] = 10 ** x[8]  # u0
        init_states[1] = 10 ** x[9]  # v0
        return init_states

    def to_init_pts(self, x):
        ir = np.zeros(num_init_pts, dtype=np.int32)
        ic = np.zeros(num_init_pts, dtype=np.int32)

        for i, coord in enumerate(zip(x[10::2], x[11::2])):
            ir[i] = int(coord[0])
            ic[i] = int(coord[1])

        return ir, ic

    def to_initializer(self, x):
        ir_init, ic_init = self.to_init_pts(x)
        init = LiawInitializer(ir_init=ir_init, ic_init=ic_init)
        return init
