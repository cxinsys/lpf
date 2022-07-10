from lpf.objectives.mse import SumMeanSquareError
from lpf.objectives.mse import MeanMeanSquareError
from lpf.objectives.mse import MinMeanSquareError
from lpf.objectives.mse import MaxMeanSquareError

from lpf.objectives.colorproportion import SumColorProportion
from lpf.objectives.colorproportion import MeanColorProportion
from lpf.objectives.colorproportion import MinColorProportion
from lpf.objectives.colorproportion import MaxColorProportion

from lpf.objectives.histrmse import SumHistogramRootMeanSquareError
from lpf.objectives.histrmse import MeanHistogramRootMeanSquareError
from lpf.objectives.histrmse import MinHistogramRootMeanSquareError
from lpf.objectives.histrmse import MaxHistogramRootMeanSquareError

from lpf.objectives.vggperceptualloss import SumVgg16PerceptualLoss
from lpf.objectives.vggperceptualloss import MeanVgg16PerceptualLoss
from lpf.objectives.vggperceptualloss import MinVgg16PerceptualLoss
from lpf.objectives.vggperceptualloss import MaxVgg16PerceptualLoss

class ObjectiveFactory:
    
    @staticmethod
    def create(name, coeff=None, device=None):
        _name = name.lower()
        
        if _name == "summeansquareerror":
            return SumMeanSquareError(coeff=coeff) 
        elif _name == "meanmeansquareerror":
            return MeanMeanSquareError(coeff=coeff) 
        elif _name == "minmeansquareerror":
            return MinMeanSquareError(coeff=coeff)
        elif _name  == "maxmeansquareerror":
            return MaxMeanSquareError(coeff=coeff)        
        
        elif _name == "sumcolorproportion":
        	return SumColorProportion(coeff=coeff) 
        elif _name == "meancolorproportion":
        	return MeanColorProportion(coeff=coeff) 
        elif _name == "mincolorproportion":
        	return MinColorProportion(coeff=coeff)
        elif _name  == "maxcolorproportion":
        	return MaxColorProportion(coeff=coeff)              
        
        elif _name == "sumhistogramrootmeansquareerror":
            return SumHistogramRootMeanSquareError(coeff=coeff)
        elif _name == "meanhistogramrootmeansquareerror":
            return MeanHistogramRootMeanSquareError(coeff=coeff)
        elif _name == "minhistogramrootmeansquareerror":
            return MinHistogramRootMeanSquareError(coeff=coeff)
        elif _name == "maxhistogramrootmeansquareerror":
            return MaxHistogramRootMeanSquareError(coeff=coeff)        
        
        elif _name == "sumvgg16perceptualloss":
            return SumVgg16PerceptualLoss(coeff=coeff, device=device) 
        elif _name == "meanvgg16perceptualloss":
            return MeanVgg16PerceptualLoss(coeff=coeff, device=device) 
        elif _name == "minvgg16perceptualloss":
            return MinVgg16PerceptualLoss(coeff=coeff, device=device)
        elif _name  == "maxvgg16perceptualloss":
            return MaxVgg16PerceptualLoss(coeff=coeff, device=device)
        
        
        raise ValueError("%s is not a supported objective."%(name))