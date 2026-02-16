import gc
import numpy as np

try:
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torchmetrics.image import StructuralSimilarityIndexMeasure
except (ImportError, ModuleNotFoundError) as err:
    err_msg = "Cannot use StructuralSimilarityIndexMeasure objectives, " \
              "since it fails to import torch, torchvision, or torchmetrics."
    print(err)
    print(err_msg)
    raise

from lpf.objectives import Objective

class EachStructuralSimilarityIndexMeasure(Objective):

    def __init__(self, coeff=None, device=None):

        if coeff is None:
            coeff = 1.0

        self._coeff = coeff

        super().__init__(device=device)
        self.model = StructuralSimilarityIndexMeasure().to(self.device)
        self.to_tensor = transforms.ToTensor()

    def compute(self, x, targets, coeff=None):

        if coeff is None:
            coeff = self._coeff

        x = self.to_tensor(x).to(self.device)

        arr_loss = np.zeros((len(targets),), dtype=np.float64)
        with torch.no_grad():
            for i, target in enumerate(targets):
                target = self.to_tensor(target).to(self.device)
                arr_loss[i] = self.model(x[None, ...], target[None, ...]).item()

        if "cuda" in self.device:  # [!] self.device is not torch.device
            torch.cuda.empty_cache()

        del target
        del x
        gc.collect()
        return coeff * arr_loss


class SumStructuralSimilarityIndexMeasure(EachStructuralSimilarityIndexMeasure):

    def compute(self, x, targets):
        each_loss = super().compute(x, targets)
        return each_loss.sum()


class MeanStructuralSimilarityIndexMeasure(EachStructuralSimilarityIndexMeasure):

    def compute(self, x, targets):
        arr_loss = super().compute(x, targets)
        return np.mean(arr_loss)


class MinStructuralSimilarityIndexMeasure(EachStructuralSimilarityIndexMeasure):

    def compute(self, x, targets):
        arr_loss = super().compute(x, targets)
        return np.min(arr_loss)


class MaxStructuralSimilarityIndexMeasure(EachStructuralSimilarityIndexMeasure):

    def compute(self, x, targets):
        arr_loss = super().compute(x, targets)
        return np.max(arr_loss)


