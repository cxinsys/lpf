import numpy as np
import scipy as sp
import scipy.stats
import cv2

from lpf.objectives import Objective


class EachColorProportion(Objective):
        
    def __init__(self, targets=None, coeff=None, lower=None, upper=None):
        if not coeff:
            coeff = 10.0
            
        self._coeff = coeff
        
        if targets:
            self._target_colpros = self.get_target_colpros(targets)
        else:
            self._target_colpros = None            
                    
        if not lower:
            lower = np.array((200, 0, 0), dtype=np.uint8)
            
        if not upper:
            upper = np.array((255, 89, 40), dtype=np.uint8)
            
        self._lower = lower
        self._upper = upper
        
    def get_colpros(self, arr):
        
        mask = cv2.inRange(arr, self._lower, self._upper)
        num_col_pts = (mask == 255).sum()
        num_tot_pts = mask.size
        
        colpro = num_col_pts / num_tot_pts
        
        return colpro
                
    def get_target_colpros(self, targets):
        colpros = []
        for target in targets:    
            # from image to array
            arr_target = np.array(target, dtype=np.uint8)  
            colpro = self.get_colpros(arr_target)
            colpros.append(colpro)
            
        return colpros
    
    def compute(self, x, targets=None, coeff=None):
        
        if not self._target_colpros:
            if not targets:
                err_msg = "targets should be given for compute() " \
                          "if targets are not given for __init__()."
                raise AttributeError(err_msg)
                
            target_colpros = self.get_target_colpros(targets)
        else:
            target_colpros = self._target_colpros
            
                    
        if not coeff:
            coeff = self._coeff
                    
        
        arr_colpro = np.zeros((len(target_colpros),), dtype=np.float64)
        for i, colpro_trg in enumerate(target_colpros):
            colpro_src = self.get_colpros(np.array(x))
            
            # loc is the mean and scale the standard deviation.
            rv = sp.stats.norm(loc=colpro_trg, scale=0.1)
            arr_colpro[i] = 1 / rv.pdf(colpro_src)
            
            
        return coeff * arr_colpro
    

class SumColorProportion(EachColorProportion):
    
    def compute(self, *args, **kwargs):
        arr_colpro = super().compute(*args, **kwargs)
        return arr_colpro.sum()


class MeanColorProportion(EachColorProportion):
    
    def compute(self, *args, **kwargs):
        arr_colpro = super().compute(*args, **kwargs)
        return np.mean(arr_colpro)
    
    
class MinColorProportion(EachColorProportion):

    def compute(self, *args, **kwargs):
        arr_colpro = super().compute(*args, **kwargs)
        return arr_colpro.min()


class MaxColorProportion(Objective):
    
    def compute(self, *args, **kwargs):
        arr_colpro = super().compute(*args, **kwargs)
        return arr_colpro.max()

