import numpy as np

from lpf.models import TwoComponentModel


class ThomasModel(TwoComponentModel):
    """Thomas model (enzyme kinetics).
    f = a - u - rho*u*v / (1 + u + K*u^2)
    g = alpha*(b - v) - rho*u*v / (1 + u + K*u^2)
    params: [Du, Dv, a, b, rho, alpha, K]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "ThomasModel"

    def reactions(self, t, u_c, v_c):
        batch_size = self.params.shape[0]

        a     = self.params[:, 2].reshape(batch_size, 1, 1)
        b     = self.params[:, 3].reshape(batch_size, 1, 1)
        rho   = self.params[:, 4].reshape(batch_size, 1, 1)
        alpha = self.params[:, 5].reshape(batch_size, 1, 1)
        K     = self.params[:, 6].reshape(batch_size, 1, 1)

        uv_term = rho * u_c * v_c / (1 + u_c + K * u_c ** 2)
        f = a - u_c - uv_term
        g = alpha * (b - v_c) - uv_term

        return f, g

    def to_dict(self, index=0, params=None, initializer=None,
                solver=None, generation=None, fitness=None):
        if params is None:
            if self._params is None:
                raise ValueError("params should be given.")
            params = self._params

        n2v = super().to_dict(index=index, initializer=initializer,
                              params=params, solver=solver,
                              generation=generation, fitness=fitness)

        n2v["Du"] = float(params[index, 0])
        n2v["Dv"] = float(params[index, 1])
        n2v["a"] = float(params[index, 2])
        n2v["b"] = float(params[index, 3])
        n2v["rho"] = float(params[index, 4])
        n2v["alpha"] = float(params[index, 5])
        n2v["K"] = float(params[index, 6])
        return n2v

    @classmethod
    def parse_params(cls, model_dicts, dtype=None):
        if not dtype:
            dtype = np.float32
        model_dicts = super().parse_params(model_dicts)
        batch_size = len(model_dicts)
        params = np.zeros((batch_size, 7), dtype=dtype)
        for i, n2v in enumerate(model_dicts):
            params[i, 0] = n2v["Du"]
            params[i, 1] = n2v["Dv"]
            params[i, 2] = n2v["a"]
            params[i, 3] = n2v["b"]
            params[i, 4] = n2v["rho"]
            params[i, 5] = n2v["alpha"]
            params[i, 6] = n2v["K"]
        return params

    def get_param_bounds(self):
        n_init_pts = self._n_init_pts
        if not hasattr(self, "bounds_min"):
            self.bounds_min = self.am.zeros((9 + 2 * n_init_pts), dtype=np.float32)
        if not hasattr(self, "bounds_max"):
            self.bounds_max = self.am.zeros((9 + 2 * n_init_pts), dtype=np.float32)
        self.bounds_min[0] = -4; self.bounds_max[0] = 0    # Du
        self.bounds_min[1] = -4; self.bounds_max[1] = 0    # Dv
        self.bounds_min[2] = -2; self.bounds_max[2] = 2    # a
        self.bounds_min[3] = -2; self.bounds_max[3] = 2    # b
        self.bounds_min[4] = -2; self.bounds_max[4] = 2    # rho
        self.bounds_min[5] = -2; self.bounds_max[5] = 2    # alpha
        self.bounds_min[6] = -3; self.bounds_max[6] = 1    # K
        self.bounds_min[7] = 0;  self.bounds_max[7] = 1.5  # u0
        self.bounds_min[8] = 0;  self.bounds_max[8] = 1.5  # v0
        for i in range(9, 2 * n_init_pts, 2):
            self.bounds_min[i] = 0; self.bounds_max[i] = self._height - 1
        for i in range(10, 2 * n_init_pts, 2):
            self.bounds_min[i] = 0; self.bounds_max[i] = self._width - 1
        return self.bounds_min, self.bounds_max

    def len_decision_vector(self):
        return 9 + 2 * self._n_init_pts
