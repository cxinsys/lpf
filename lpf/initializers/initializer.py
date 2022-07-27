import numpy as np
from PIL import Image


class Initializer:
    def __init__(self,
                 name=None,
                 ind_init=None,
                 dtype=None):
        
        self._name = name
        
        if not dtype:
            dtype = np.float32
        
        self._dtype = dtype
        self._ind_init = ind_init

    def initialize(self, model, init_states, params):
        raise NotImplementedError()

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype
