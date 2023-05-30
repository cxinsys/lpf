import numpy as np
from lpf.converters import Converter

class SchnakenbergModel(Converter):

    def __init__(self):
        self._name = "SchnakenbergConverter"

    def get_param_names(self):
        return [
            "Du",
            "Dv",
            "rho",
            "su",
            "sv",
            "mu",
            "u0",
            "v0"
        ]

    def to_params(self, dv, params=None):
        if params is None:
            params = np.zeros((1, 8), dtype=np.float64)

        params[0, 0] = 10 ** dv[0, 0]  # Du
        params[0, 1] = 10 ** dv[0, 1]  # Dv
        params[0, 2] = 10 ** dv[0, 2]  # rho
        params[0, 3] = 10 ** dv[0, 3]  # su
        params[0, 4] = 10 ** dv[0, 4]  # sv
        params[0, 5] = 10 ** dv[0, 5]  # mu

        return params

    def to_init_states(self, dv, init_states=None):
        if init_states is None:
            init_states = np.zeros((1, 2), dtype=np.float64)

        init_states[0, 0] = 10 ** dv[0, 6]  # u0
        init_states[0, 1] = 10 ** dv[0, 7]  # v0
        return init_states

    def to_init_pts(self, dv):
        coords = []
        for coord in zip(dv[0, 8::2], dv[0, 9::2]):
            coords.append((int(coord[0]), int(coord[1])))

        return np.array([coords], dtype=np.uint32)
