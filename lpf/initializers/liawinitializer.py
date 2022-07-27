import numpy as np

from lpf.initializers import Initializer
from lpf.models import ReactionDiffusionModel


class LiawInitializer(Initializer):
    
    def __init__(self, ind_init=None, dtype=None,):
        super().__init__(name="LiawInitializer", 
                         ind_init=ind_init,
                         dtype=dtype)

    def update(self, model_dicts):
        ind_init = []

        for i, n2v in enumerate(model_dicts):
            num_init_pts = 0
            dict_init_pts = {}
            for name, val in n2v.items():
                if "init_pts" in name:
                    # print(name, val)
                    dict_init_pts[name] = (int(val[0]), int(val[1]))
                    num_init_pts += 1
            # end of for

            for j, (name, val) in enumerate(dict_init_pts.items()):
                ind_init.append((i, val[0], val[1]))
            # end of for
        # end of for

        self._ind_init = np.array(ind_init, dtype=np.uint32)

    def initialize(self, model, init_states, ind_init=None):

        if not isinstance(model, ReactionDiffusionModel):
            err_msg = "model should be a subclass of ReactionDiffusionModel."
            raise TypeError(err_msg)

        if ind_init is None:
            if self._ind_init is not None:
                ind_init = self._ind_init
            else:  # Both ind_init and self._ind_init are not given
                raise ValueError("ind_init should be given!")

        with model.am:
            ind_init = model.am.array(ind_init, dtype=ind_init.dtype)

            model.t = 0.0
            u0 = model.am.array(init_states[:, 0], dtype=init_states.dtype)
            v0 = model.am.array(init_states[:, 1], dtype=init_states.dtype)

            batch_size = init_states.shape[0]
            shape = (batch_size, model.height, model.width)
            if not hasattr(model, "u"):
                model.u = model.am.zeros(shape, dtype=self.dtype)
            else:
                model.u[:, 0] = 0.0  # model.u.fill(0.0)

            for ix_batch in ind_init[:, 0]:
                model.u[ix_batch, ind_init[:, 1], ind_init[:, 2]] = u0[ix_batch]

            if not hasattr(model, "v"):
                tmp_v0 = v0.reshape(batch_size, 1, 1)
                model.v = tmp_v0 * model.am.ones(shape, dtype=self.dtype)
            else:
                model.v[:, 1] = v0  # model.v.fill(v0)
