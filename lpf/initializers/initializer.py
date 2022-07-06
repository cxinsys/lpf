import numpy as np
from PIL import Image


class Initializer:
    
    def __init__(self,
                 name=None,
                 fpath_init=None,
                 ir_init=None,
                 ic_init=None,
                 dtype=None):  
        
        self._name = name
        
        if not dtype:
            dtype = np.float32
        
        self._dtype = dtype
        
        if fpath_init:
            img_init = Image.open(fpath_init)
            arr_init = np.array(img_init, dtype=self.dtype)
            self._ir_init, self._ic_init = arr_init[:, :, -1].nonzero()
            
        if ir_init is not None and ic_init is not None:
            self._ir_init = ir_init
            self._ic_init = ic_init
        
        
    def initialize(self, model, init_state, params):
        raise NotImplementedError()


    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype