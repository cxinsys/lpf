from os.path import join as pjoin

import numpy as np
from PIL import Image

from lpf.initializers import Initializer
from lpf.models import ReactionDiffusionModel
from lpf.utils import get_module_dpath


class LiawInitializer(Initializer):
    
    def __init__(self,
                 name=None,
                 ir_init=None,
                 ic_init=None):
        if name:                        
            dpath_data = pjoin(get_module_dpath("data"), "haxyridis")
            fpath_init = pjoin(dpath_data, "init", "%s.png"%(name))
            super().__init__(name, fpath_init)        
            return
        
        if ir_init is not None and ic_init is not None:            
            super().__init__(name="liaw",
                             ir_init=ir_init,
                             ic_init=ic_init)
            return
        
        raise ValueError("name or both ir_init and ic_init should be given.")
        
    def initialize(self, model, init_states, params):
        if not isinstance(model, ReactionDiffusionModel):
            err_msg = "model should be a subclass of %s."%(Model)
            raise TypeError(err_msg)
        
        model.t = 0.0       
        u0, v0 = init_states
        
        shape = (model.height, model.width)
        if not hasattr(model, "u"):
            model.u = np.zeros(shape, dtype=np.float64)
        else:
            model.u.fill(0.0)
            
        model.u[self._ir_init, self._ic_init] = u0
        
        if not hasattr(model, "v"):
            model.v = np.full(shape, v0, dtype=np.float64)
        else:
            model.v.fill(v0)

