import numpy as np
import cv2

from lpf.objectives import Objective



class EachHistogramRootMeanSquareError(Objective):
        
    def __init__(self, targets=None, coeff=None):        
        if not coeff:
            coeff = 1e-02
            
        self._coeff = coeff
        
        if targets:
            self._target_hists = self.get_target_histograms(targets)
        else:
            self._target_hists = None            
        
    def get_histogram(self, arr):
        hist_r = cv2.calcHist([arr], [2], None, [256], [0, 256])
        hist_g = cv2.calcHist([arr], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([arr], [0], None, [256], [0, 256])
    
        return np.array([hist_r.ravel(), hist_g.ravel(), hist_b.ravel()])
    
            
    def get_target_histograms(self, targets):
        hists = []
        for target in targets:    
            arr_target = np.array(target, dtype=np.uint8)
            hist = self.get_histogram(arr_target)
            hists.append(hist)
            
        return hists
    
    def compute(self, x, targets=None, coeff=None):
        
        if not self._target_hists:
            if not targets:
                err_msg = "targets should be given for compute() " \
                          "if targets are not given for __init__()."
                raise AttributeError(err_msg)
                
            target_hists = self.get_target_histograms(targets)
        else:
            target_hists = self._target_hists
            
                    
        if not coeff:
            coeff = self._coeff
                    
        
        arr_rmse = np.zeros((len(target_hists),), dtype=np.float64)
        for i, hist_target in enumerate(target_hists):
            hist_source = self.get_histogram(np.array(x))
            arr_rmse[i] = np.sqrt(np.mean((hist_target - hist_source)**2))
          
        return coeff * arr_rmse
    

class SumHistogramRootMeanSquareError(EachHistogramRootMeanSquareError):
    
    def compute(self, *args, **kwargs):
        arr_rmse = super().compute(*args, **kwargs)
        return arr_rmse.sum()


class MeanHistogramRootMeanSquareError(EachHistogramRootMeanSquareError):
    
    def compute(self, *args, **kwargs):
        arr_rmse = super().compute(*args, **kwargs)
        return np.mean(arr_rmse)
    
    
class MinHistogramRootMeanSquareError(EachHistogramRootMeanSquareError):

    def compute(self, *args, **kwargs):
        arr_rmse = super().compute(*args, **kwargs)
        return arr_rmse.min()


class MaxHistogramRootMeanSquareError(Objective):
    
    def compute(self, *args, **kwargs):
        arr_rmse = super().compute(*args, **kwargs)
        return arr_rmse.max()

