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

        arr_mse = np.zeros((len(targets), len(x)), dtype=np.float64)
        for i, target in enumerate(targets):
            for j, img in enumerate(x):
                arr_mse[i, j] = np.mean((np.array(img) - np.array(target))**2)
          
        return coeff * arr_mse
    

class SumMeanSquareError(EachMeanSquareError):
    
    def compute(self, x, targets):
        arr_mse = super().compute(x, targets)
        return arr_mse.sum(axis=0)


class MeanMeanSquareError(EachMeanSquareError):
    
    def compute(self, x, targets):
        arr_mse = super().compute(x, targets)
        return np.mean(arr_mse, axis=0)
    
    
class MinMeanSquareError(EachMeanSquareError):

    def compute(self, x, targets):
        arr_mse = super().compute(x, targets)
        return arr_mse.min(axis=0)


class MaxMeanSquareError(EachMeanSquareError):
    
    def compute(self, x, targets):
        arr_mse = super().compute(x, targets)
        return arr_mse.max(axis=0)

