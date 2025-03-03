from lpf.objectives.objective import Objective

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

from lpf.objectives.objectivefactory import ObjectiveFactory



__all__ = [
    "SumMeanSquareError",
    "MeanMeanSquareError",
    "MinMeanSquareError",
    "MaxMeanSquareError",
    
    "SumColorProportion",
    "MeanColorProportion",
    "MinColorProportion",
    "MaxColorProportion",

    "SumHistogramRootMeanSquareError",
    "MeanHistogramRootMeanSquareError",
    "MinHistogramRootMeanSquareError",
    "MaxHistogramRootMeanSquareError",
    
    "SumVgg16PerceptualLoss",
    "MeanVgg16PerceptualLoss",
    "MinVgg16PerceptualLoss",
    "MaxVgg16PerceptualLoss",

    "SumStructuralSimilarityIndexMeasure",
    "MeanStructuralSimilarityIndexMeasure",
    "MinStructuralSimilarityIndexMeasure",
    "MaxStructuralSimilarityIndexMeasure",

    "SumLearnedPerceptualImagePatchSimilarity",
    "MeanLearnedPerceptualImagePatchSimilarity",
    "MinLearnedPerceptualImagePatchSimilarity",
    "MaxLearnedPerceptualImagePatchSimilarity",

    "ObjectiveFactory"
]