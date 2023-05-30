from abc import ABC

import numpy as np
from lpf.initializers import LiawInitializer


class Converter(ABC):
    """Helper object that supports the conversion between the decision vector of PyGMO and model parameters.
    """

    @property
    def name(self):
        return self._name

    def to_dv(self,
              model_dict,
              n_init_pts=None):

        if not isinstance(model_dict, dict):
            raise TypeError("model_dict should be a dictionary.")

        if not n_init_pts:
            n_init_pts = -1
        elif n_init_pts and n_init_pts < 0:
            raise TypeError("n_init_pts should be greater than 0.")

        dv = []
        names = self.get_param_names()
        for name in names:
            dv.append(np.log10(model_dict[name]))


        cnt_init_pts = 0
        for name, val in model_dict.items():
            if cnt_init_pts == n_init_pts:
                break

            if name.startswith("init_pts"):
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
        raise NotImplementedError()

    def to_init_states(self, dv, init_states=None):
        raise NotImplementedError()

    def to_init_pts(self, dv):
        raise NotImplementedError()

    def to_initializer(self, dv, initializer_class=None):
        init_states = self.to_init_states(dv)
        init_pts = self.to_init_pts(dv)

        if initializer_class:
            initializer = initializer_class(init_states=init_states, init_pts=init_pts)
        else:
            initializer = LiawInitializer(init_states=init_states, init_pts=init_pts)

        return initializer