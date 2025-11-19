import numpy as np

from lpf.initializers import TwoComponentInitializer
import lpf.models


class LiawInitializer(TwoComponentInitializer):
    
    def __init__(self, init_states=None, init_pts=None, dtype=None):
        super().__init__(name="LiawInitializer",
                         init_states=init_states,
                         init_pts=init_pts,
                         dtype=dtype)

    def update(self, model_dicts):
        """Parse the initial states and points from the model dictionaries.
        """

        batch_size = len(model_dicts)
        
        # Use NumPy for initial creation since we don't have ArrayModule context yet
        # The arrays will be converted to appropriate backend arrays in initialize()
        self._init_states = np.zeros((batch_size, 2), dtype=self._dtype)
        init_pts = []

        for i, n2v in enumerate(model_dicts):
            self._init_states[i, 0] = n2v["u0"]
            self._init_states[i, 1] = n2v["v0"]

            n_init_pts = 0
            dict_init_pts = {}
            for name, val in n2v.items():
                if name.startswith("init_pts"):
                    dict_init_pts[name] = (int(val[0]), int(val[1]))
                    n_init_pts += 1
            # end of for

            coords = []
            for j, (name, coord) in enumerate(dict_init_pts.items()):
                coords.append((coord[0], coord[1]))
            # end of for
            init_pts.append(coords)
        # end of for

        self._init_pts = np.array(init_pts, dtype=np.uint32)

    def initialize(self, model, init_states=None, init_pts=None):

        if not isinstance(model, lpf.models.TwoComponentModel):
            err_msg = "model should be a subclass of TwoComponentModel."
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
            # Convert NumPy arrays to appropriate backend arrays
            init_pts = model.am.array(init_pts, dtype=init_pts.dtype)
            init_states = model.am.array(init_states, dtype=init_states.dtype)

            batch_size = model.batch_size  # init_states.shape[0]

            u0 = model.am.array(init_states[:, 0], dtype=init_states.dtype)
            v0 = model.am.array(init_states[:, 1], dtype=init_states.dtype)
            v0 = v0.reshape(batch_size, 1, 1)

            for i in range(batch_size):
                # model._u[i, init_pts[i, :, 0], init_pts[i, :, 1]] = u0[i]
                model.am.set(model._u, (i, init_pts[i, :, 0], init_pts[i, :, 1]), u0[i])

            # model._y_mesh[1, :, :, :] = v0            
            model.am.set(model._y_mesh, (1, ...), v0)
        # end of with

    def to_dict(self, index):
        n2v = {}  # Mapping variable names to values.

        n2v["initializer"] = self.name

        n2v["u0"] = float(self.init_states[index, 0])
        n2v["v0"] = float(self.init_states[index, 1])

        # Save init points
        n2v["n_init_pts"] = self.init_pts[index].shape[0]

        for i, (ir, ic) in enumerate(self.init_pts[index, :]):
            # Convert int to str due to JSON format.
            n2v["init_pts_%d"%(i)] = [int(ir), int(ic)]
        # end of for

        return n2v
