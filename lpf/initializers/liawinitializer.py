import numpy as np

from lpf.initializers import Initializer
from lpf.models import ReactionDiffusionModel


class LiawInitializer(Initializer):
    
    def __init__(self, init_states=None, init_pts=None, dtype=None,):
        super().__init__(name="LiawInitializer",
                         init_states=init_states,
                         init_pts=init_pts,
                         dtype=dtype)

    def update(self, model_dicts):
        # Update init_states

        self._init_states =

        # Update init_pts
        init_pts = []

        for i, n2v in enumerate(model_dicts):
            num_init_pts = 0
            dict_init_pts = {}
            for name, val in n2v.items():
                if "init_pts" in name:
                    # print(name, val)
                    dict_init_pts[name] = (int(val[0]), int(val[1]))
                    num_init_pts += 1
            # end of for

            # for j, (name, val) in enumerate(dict_init_pts.items()):
            #     init_pts.append((val[0], val[1]))
            # # end of for

            coords = []
            for j, (name, coord) in enumerate(dict_init_pts.items()):
                coords.append((coord[0], coord[1]))
            # end of for
            init_pts.append(coords)
        # end of for

        self._init_pts = np.array(init_pts, dtype=np.uint32)

    def initialize(self, model, init_states=None, init_pts=None):

        if not isinstance(model, ReactionDiffusionModel):
            err_msg = "model should be a subclass of ReactionDiffusionModel."
            raise TypeError(err_msg)

        if init_states is None:
            if self._init_states is not None:
                init_states = self._init_states
            else:  # Both init_states and self._init_states are not given
                raise ValueError("init_states should be given!")

        if init_pts is None:
            if self._init_pts is not None:
                init_pts = self._init_pts
            else:  # Both init_pts and self._init_pts are not given
                raise ValueError("init_pts should be given!")

        with model.am:
            init_pts = model.am.array(init_pts, dtype=init_pts.dtype)

            model.t = 0.0

            batch_size = init_states.shape[0]

            u0 = model.am.array(init_states[:, 0], dtype=init_states.dtype)

            v0 = model.am.array(init_states[:, 1], dtype=init_states.dtype)
            v0 = v0.reshape(batch_size, 1, 1)

            shape = (batch_size, model.height, model.width)
            model.u = model.am.zeros(shape, dtype=self.dtype)

            for i in range(batch_size):
                model.u[i, init_pts[i, :, 0], init_pts[i, :, 1]] = u0[i]

            model.v = v0 * model.am.ones(shape, dtype=self.dtype)
