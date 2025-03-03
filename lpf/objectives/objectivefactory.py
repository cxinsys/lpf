from collections.abc import Sequence

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

from lpf.objectives.ssim import SumStructuralSimilarityIndexMeasure
from lpf.objectives.ssim import MeanStructuralSimilarityIndexMeasure
from lpf.objectives.ssim import MinStructuralSimilarityIndexMeasure
from lpf.objectives.ssim import MaxStructuralSimilarityIndexMeasure

from lpf.objectives.perceptualsimilarity import SumLearnedPerceptualImagePatchSimilarity
from lpf.objectives.perceptualsimilarity import MeanLearnedPerceptualImagePatchSimilarity
from lpf.objectives.perceptualsimilarity import MinLearnedPerceptualImagePatchSimilarity
from lpf.objectives.perceptualsimilarity import MaxLearnedPerceptualImagePatchSimilarity

class ObjectiveFactory:

    @staticmethod
    def create_single(name, coeff=None, device=None, **kwargs):
        _name = name.lower()

        if _name == "summeansquareerror":
            return SumMeanSquareError(coeff=coeff, **kwargs)
        elif _name == "meanmeansquareerror":
            return MeanMeanSquareError(coeff=coeff, **kwargs)
        elif _name == "minmeansquareerror":
            return MinMeanSquareError(coeff=coeff, **kwargs)
        elif _name == "maxmeansquareerror":
            return MaxMeanSquareError(coeff=coeff, **kwargs)

        elif _name == "sumcolorproportion":
            return SumColorProportion(coeff=coeff, **kwargs)
        elif _name == "meancolorproportion":
            return MeanColorProportion(coeff=coeff, **kwargs)
        elif _name == "mincolorproportion":
            return MinColorProportion(coeff=coeff, **kwargs)
        elif _name == "maxcolorproportion":
            return MaxColorProportion(coeff=coeff, **kwargs)

        elif _name == "sumhistogramrootmeansquareerror":
            return SumHistogramRootMeanSquareError(coeff=coeff, **kwargs)
        elif _name == "meanhistogramrootmeansquareerror":
            return MeanHistogramRootMeanSquareError(coeff=coeff, **kwargs)
        elif _name == "minhistogramrootmeansquareerror":
            return MinHistogramRootMeanSquareError(coeff=coeff, **kwargs)
        elif _name == "maxhistogramrootmeansquareerror":
            return MaxHistogramRootMeanSquareError(coeff=coeff, **kwargs)

        elif _name == "sumvgg16perceptualloss":
            return SumVgg16PerceptualLoss(coeff=coeff, device=device, **kwargs)
        elif _name == "meanvgg16perceptualloss":
            return MeanVgg16PerceptualLoss(coeff=coeff, device=device, **kwargs)
        elif _name == "minvgg16perceptualloss":
            return MinVgg16PerceptualLoss(coeff=coeff, device=device, **kwargs)
        elif _name == "maxvgg16perceptualloss":
            return MaxVgg16PerceptualLoss(coeff=coeff, device=device, **kwargs)

        elif _name == "sumstructuralsimilarityindexmeasure":
            return SumStructuralSimilarityIndexMeasure(coeff=coeff, device=device, **kwargs)
        elif _name == "meanstructuralsimilarityindexmeasure":
            return MeanStructuralSimilarityIndexMeasure(coeff=coeff, device=device, **kwargs)
        elif _name == "minstructuralsimilarityindexmeasure":
            return MinStructuralSimilarityIndexMeasure(coeff=coeff, device=device, **kwargs)
        elif _name == "maxstructuralsimilarityindexmeasure":
            return MaxStructuralSimilarityIndexMeasure(coeff=coeff, device=device, **kwargs)

        elif "sumlearnedperceptualimagepatchsimilarity" in _name:
            _, net_type = _name.split(":")
            return SumLearnedPerceptualImagePatchSimilarity(net_type=net_type, coeff=coeff, device=device, **kwargs)
        elif "meanlearnedperceptualimagepatchsimilarity" in _name:
            _, net_type = _name.split(":")
            return MeanLearnedPerceptualImagePatchSimilarity(net_type=net_type, coeff=coeff, device=device, **kwargs)
        elif "minlearnedperceptualimagepatchsimilarity" in _name:
            _, net_type = _name.split(":")
            return MinLearnedPerceptualImagePatchSimilarity(net_type=net_type, coeff=coeff, device=device, **kwargs)
        elif "maxlearnedperceptualimagepatchsimilarity" in _name:
            _, net_type = _name.split(":")
            return MaxLearnedPerceptualImagePatchSimilarity(net_type=net_type, coeff=coeff, device=device, **kwargs)

        raise ValueError("%s is not a supported objective."%(name))


    @staticmethod
    def create(obj, coeff=None, device=None, **kwargs):
        
        if isinstance(obj, str):
            return ObjectiveFactory.create_single(obj, coeff, device, **kwargs)

        elif isinstance(obj, Sequence):
            objectives = []
            for cfg in obj:
                name = cfg[0]
                coeff = float(cfg[1])
                device = cfg[2]

                print("[OBJECTIVE DEVICE] %s: %s" % (name, device))
                objectives.append(ObjectiveFactory.create_single(name, coeff, device, **kwargs))
            # end of for

            return objectives
        # end of if
        else:
            return ObjectiveFactory.create_single(obj, coeff, device, **kwargs)