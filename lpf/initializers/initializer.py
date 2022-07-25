import numpy as np
from PIL import Image

from lpf.array import get_array_module


class Initializer:
    def __init__(self,
                 name=None,
                 ind_init=None,
                 dtype=None,
                 device=None):
        
        self._name = name
        
        if not dtype:
            dtype = np.float32
        
        self._dtype = dtype

        self._am = get_array_module(device)

        if ind_init is not None:
            self._ind_init = ind_init

    def initialize(self, model, init_states, params):
        raise NotImplementedError()

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def am(self):
        return self._am