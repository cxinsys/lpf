from itertools import product

import numpy as np
from lpf.converters import Converter
from lpf.initializers import LiawInitializer


class LiawConverter(Converter):
    """Helper object that supports the conversion between the decision vector of PyGMO and model parameters.
    """

    def to_dv(self,
              model_dict,
              n_init_pts=None):

        if not isinstance(model_dict, dict):
            raise TypeError("model_dict should be a dictionary.")

        if not n_init_pts:
            n_init_pts = -1
        elif n_init_pts and n_init_pts < 0:
            raise TypeError("n_init_pts should be greater than 0.")

        dv = [
            np.log10(model_dict["Du"]),
            np.log10(model_dict["Dv"]),
            np.log10(model_dict["ru"]),
            np.log10(model_dict["rv"]),
            np.log10(model_dict["k"]),
            np.log10(model_dict["su"]),
            np.log10(model_dict["sv"]),
            np.log10(model_dict["mu"]),
            np.log10(model_dict["u0"]),
            np.log10(model_dict["v0"])
        ]

        cnt_init_pts = 0
        for name, val in model_dict.items():
            if cnt_init_pts == n_init_pts:
                break

            if "init_pts" in name:
                dv.append(int(val[0]))
                dv.append(int(val[1]))
                cnt_init_pts += 1
        # end of for

        if cnt_init_pts < n_init_pts:
            n_init_pts_remained = n_init_pts - cnt_init_pts
            for i in range(n_init_pts_remained):
                dv.append(0)
                dv.append(0)

        return np.array(dv, dtype=np.float64)

    def to_params(self, dv, params=None):
        if params is None:
            params = np.zeros((1, 8), dtype=np.float64)

        params[0, 0] = 10 ** dv[0, 0]  # Du
        params[0, 1] = 10 ** dv[0, 1]  # Dv
        params[0, 2] = 10 ** dv[0, 2]  # ru
        params[0, 3] = 10 ** dv[0, 3]  # rv
        params[0, 4] = 10 ** dv[0, 4]  # k
        params[0, 5] = 10 ** dv[0, 5]  # su
        params[0, 6] = 10 ** dv[0, 6]  # sv
        params[0, 7] = 10 ** dv[0, 7]  # mu

        return params

    def to_init_states(self, dv, init_states=None):
        if init_states is None:
            init_states = np.zeros((1, 2), dtype=np.float64)

        init_states[0, 0] = 10 ** dv[0, 8]  # u0
        init_states[0, 1] = 10 ** dv[0, 9]  # v0
        return init_states

    def to_init_pts(self, dv):
        coords = []
        for coord in zip(dv[0, 10::2], dv[0, 11::2]):
            coords.append((int(coord[0]), int(coord[1])))

        return np.array([coords], dtype=np.uint32)

    def to_initializer(self, dv):
        init_states = self.to_init_states(dv)
        init_pts = self.to_init_pts(dv)
        initializer = LiawInitializer(init_states=init_states, init_pts=init_pts)
        return initializer
