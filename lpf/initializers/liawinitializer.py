from os.path import join as pjoin

import numpy as np
from PIL import Image

from lpf.initializers import Initializer
from lpf.models import ReactionDiffusionModel
from lpf.utils import get_module_dpath


class LiawInitializer(Initializer):
    
    def __init__(self, ind_init=None, dtype=None):
        super().__init__(name="liaw", ind_init=ind_init, dtype=dtype)

    def initialize(self, model, init_states, param_dicts, ind_init=None):

        if not isinstance(model, ReactionDiffusionModel):
            err_msg = "model should be a subclass of %s."%(model)
            raise TypeError(err_msg)

        if not ind_init:
            if self._ind_init is not None:
                ind_init = self._ind_init
        else:
            raise ValueError("ind_init should be given!")
        
        model.t = 0.0       
        u0 = init_states[:, 0]
        v0 = init_states[:, 1]

        batch_size = init_states.shape[0]
        shape = (batch_size, model.height, model.width)
        if not hasattr(model, "u"):
            model.u = np.zeros(shape, dtype=self.dtype)
        else:
            model.u[:, 0] = 0.0  # model.u.fill(0.0)
            
            
        for ix_batch in ind_init[:, 0]:
            model.u[ix_batch, ind_init[:, 1], ind_init[:, 2]] = u0[ix_batch]
        
        if not hasattr(model, "v"):
            # model.v = np.full(shape, v0, dtype=self.dtype)
            tmp_v0 = v0.reshape(batch_size, 1, 1)
            model.v = tmp_v0 * np.ones(shape, dtype=self.dtype)
        else:
            model.v[:, 1] = v0  # model.v.fill(v0)

    def from_param_dicts(self, param_dicts):
        ind_init = []

        for i, n2v in enumerate(param_dicts):
            num_init_pts = 0
            init_pts = {}
            for name, val in n2v.items():
                if "init_pts" in name:
                    # print(name, val)
                    init_pts[name] = (int(val[0]), int(val[1]))
                    num_init_pts += 1
            # end of for

            for j, (name, val) in enumerate(init_pts.items()):
                ind_init.append((i, val[0], val[1]))
        # end of for

        self._ind_init = np.array(ind_init, dtype=np.uint32)