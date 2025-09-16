import numpy as np

from lpf.initializers import TwoComponentInitializer
import lpf.models


class TwoComponentConstantInitializer(TwoComponentInitializer):

    def __init__(self, init_states=None, init_pts=None, dtype=None):
        super().__init__(name="TwoComponentConstantInitializer",
                         init_states=init_states,
                         init_pts=init_pts,
                         dtype=dtype)

    def update(self, model_dicts, array_module=None):
        """Parse the initial states from the model dictionaries.
        """
        batch_size = len(model_dicts)
        # Use NumPy for initial creation since we don't have ArrayModule context yet
        # The arrays will be converted to appropriate backend arrays in initialize()
        self._init_states = np.zeros((batch_size, 2), dtype=self._dtype)

        for i, n2v in enumerate(model_dicts):
            self._init_states[i, 0] = n2v["u0"]
            self._init_states[i, 1] = n2v["v0"]
        # end of for

        self._init_pts = None

    def initialize(self, model, init_states=None, init_pts=None):

        if not isinstance(model, lpf.models.TwoComponentModel):
            err_msg = "model should be a subclass of TwoComponentModel."
            raise TypeError(err_msg)

        if init_states is None:
            if self._init_states is not None:
                init_states = self._init_states
            else:  # Both init_states and self._init_states are not given
                raise ValueError("init_states should be given!")

        with model.am:
            # Convert NumPy arrays to appropriate backend arrays
            init_states = model.am.array(init_states, dtype=init_states.dtype)
            
            batch_size = model.batch_size  # init_states.shape[0]

            u0 = init_states[:, 0]
            v0 = init_states[:, 1]

            u0 = u0.reshape(batch_size, 1, 1)
            v0 = v0.reshape(batch_size, 1, 1)

            # model._y_mesh[0, :, :, :] = u0
            # model._y_mesh[1, :, :, :] = v0
            model.am.set(model._y_mesh, (0, ...), u0)
            model.am.set(model._y_mesh, (1, ...), v0)
        # end of with

    def to_dict(self, index):
        n2v = {}  # Mapping variable names to values.

        n2v["initializer"] = self.name

        n2v["u0"] = float(self.init_states[index, 0])
        n2v["v0"] = float(self.init_states[index, 1])

        # Save init points
        n2v["n_init_pts"] = 0

        return n2v
