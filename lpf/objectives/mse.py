import numpy as np

from lpf.objectives import Objective


class EachMeanSquareError(Objective):
    
    def __init__(self, coeff=None):
        if not coeff:
            coeff = 1e-1
            
        self._coeff = coeff
                        
    def compute(self, x, targets, coeff=None):        
        if not coeff:
            coeff = self._coeff
        
        arr_mse = np.zeros((len(targets),), dtype=np.float64)
        for i, target in enumerate(targets):
            arr_mse[i] = np.mean((np.array(x) - np.array(target))**2)
          
        return coeff * arr_mse
    

class SumMeanSquareError(EachMeanSquareError):
    
    def compute(self, x, targets):
        arr_mse = super().compute(x, targets)
        return arr_mse.sum()


class MeanMeanSquareError(EachMeanSquareError):
    
    def compute(self, x, targets):
        arr_mse = super().compute(x, targets)
        return np.mean(arr_mse)
    
    
class MinMeanSquareError(EachMeanSquareError):

    def compute(self, x, targets):
        arr_mse = super().compute(x, targets)
        return arr_mse.min()


class MaxMeanSquareError(Objective):
    
    def compute(self, x, targets):
        arr_mse = super().compute(x, targets)
        return arr_mse.max()

