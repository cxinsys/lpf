from abc import ABC
import numpy as np


class Initializer(ABC):
    def __init__(self,
                 name=None,
                 init_states=None,
                 init_pts=None,
                 dtype=None):
        
        self._name = name
        
        if not dtype:
            dtype = np.float32
        
        self._dtype = dtype
        self._init_pts = init_pts
        self._init_states = init_states

    def initialize(self, model, init_states=None, init_pts=None):
        raise NotImplementedError()

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def init_pts(self):
        return self._init_pts
    
    @property
    def init_states(self):
        return self._init_states