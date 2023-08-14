from collections.abc import Sequence

import numpy as np

from lpf.models import TwoComponentModel


class GiererMeinhardtModel(TwoComponentModel):
    """
    Gierer-Meinhardt model
    - Activator-Inhibitor model with different sources.
    - The constant production term in the activator is removed for simplicity.
    """

    def __init__(self, *args, **kwargs):

        # Set the device.
        super().__init__(*args, **kwargs)

        # Set constant members.
        self._name = "GiererMeinhardtModel"

    def reactions(self, t, u_c, v_c):
        batch_size = self.params.shape[0]

        ru = self.params[:, 2].reshape(batch_size, 1, 1)
        rv = self.params[:, 3].reshape(batch_size, 1, 1)
        mu = self.params[:, 4].reshape(batch_size, 1, 1)
        nu = self.params[:, 5].reshape(batch_size, 1, 1)

        usq = u_c ** 2
        f = ru * usq / v_c - mu * u_c
        g = rv * usq - nu * v_c

        return f, g

    def to_dict(self,
                index=0,
                params=None,
                initializer=None,
                solver=None,
                generation=None,
                fitness=None):

        if params is None:
            if self._params is None:
                raise ValueError("params should be given.")

            params = self._params

        # Get the dict from the parent class.
        n2v = super().to_dict(index=index,
                              initializer=initializer,
                              params=params,
                              solver=solver,
                              generation=generation,
                              fitness=fitness)

        # Add the kinetic parameters to the parent dict.
        n2v["Du"] = float(params[index, 0])
        n2v["Dv"] = float(params[index, 1])
        n2v["ru"] = float(params[index, 2])  # ρ_u (rho_u)
        n2v["rv"] = float(params[index, 3])  # ρ_u (rho_v)
        n2v["mu"] = float(params[index, 4])  # μ
        n2v["nu"] = float(params[index, 5])  # ν
        
        return n2v

    @classmethod
    def parse_params(self, model_dicts, dtype=None):
        if not dtype:
            dtype = np.float64
            
        model_dicts = super().parse_params(model_dicts)
        batch_size = len(model_dicts)
        params = np.zeros((batch_size, 6), dtype=dtype)

        for index, n2v in enumerate(model_dicts):
            params[index, 0] = n2v["Du"]
            params[index, 1] = n2v["Dv"]
            params[index, 2] = n2v["ru"]
            params[index, 3] = n2v["rv"]
            params[index, 4] = n2v["mu"]
            params[index, 5] = n2v["nu"]

        return params

    def get_param_bounds(self):
        n_init_pts = self._n_init_pts

        if not hasattr(self, "bounds_min"):
            self.bounds_min = self.am.zeros((8 + 2 * n_init_pts), dtype=np.float64)

        if not hasattr(self, "bounds_max"):
            self.bounds_max = self.am.zeros((8 + 2 * n_init_pts), dtype=np.float64)

        # Du
        self.bounds_min[0] = -4
        self.bounds_max[0] = 0

        # Dv
        self.bounds_min[1] = -4
        self.bounds_max[1] = 0

        # ru
        self.bounds_min[2] = -3
        self.bounds_max[2] = 3

        # rv
        self.bounds_min[3] = -3
        self.bounds_max[3] = 3

        # mu
        self.bounds_min[4] = -3
        self.bounds_max[4] = 3

        # nu
        self.bounds_min[5] = -3
        self.bounds_max[5] = 3

        # u0
        self.bounds_min[6] = 0
        self.bounds_max[6] = 1.5

        # v0
        self.bounds_min[7] = 0
        self.bounds_max[7] = 1.5

        # Initial points (initializing positions).
        for index in range(8, 2 * n_init_pts, 2):
            self.bounds_min[index] = 0
            self.bounds_max[index] = self._height - 1
        # end of for

        for index in range(9, 2 * n_init_pts, 2):
            self.bounds_min[index] = 0
            self.bounds_max[index] = self._width - 1
        # end of for

        return self.bounds_min, self.bounds_max

    def len_decision_vector(self):  # The length of the decision vector in PyGMO
        return 8 + 2 * self._n_init_pts

# end of class GrayScottModel
