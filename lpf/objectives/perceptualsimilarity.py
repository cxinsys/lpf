import gc
import numpy as np

try:
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import lpips
except (ImportError, ModuleNotFoundError) as err:
    err_msg = "Cannot use FrechetInceptionDistance objectives, " \
              "since it fails to import torch, torchvision, or torchmetrics."
    print(err)
    print(err_msg)

from lpf.objectives import Objective


class EachLearnedPerceptualImagePatchSimilarity(Objective):

    def __init__(self, net_type=None, coeff=None, device=None):

        if coeff is None:
            coeff = 1.0

        self._coeff = coeff

        super().__init__(device=device)

        if net_type is None:
            net_type = 'vgg'

        self.model = lpips.LPIPS(net=net_type).to(self.device)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def compute(self, x, targets, coeff=None):

        if coeff is None:
            coeff = self._coeff

        arr_img = []
        for img in x:
            tmp = self.to_tensor(img).to(self.device)
            arr_img.append(tmp)

        x = torch.stack(arr_img)

        arr_loss = []
        with torch.no_grad():
            for i, target in enumerate(targets):
                target = self.to_tensor(target).to(self.device)
                arr_loss.append(self.model.forward(x, target).reshape(-1).detach().cpu())

            arr_loss = np.array(arr_loss)

        if "cuda" in self.device:  # [!] self.device is not torch.device
            torch.cuda.empty_cache()

        del target
        del x
        del arr_img
        del tmp

        gc.collect()
        return coeff * arr_loss


class SumLearnedPerceptualImagePatchSimilarity(EachLearnedPerceptualImagePatchSimilarity):

    def compute(self, x, targets):
        arr_loss = super().compute(x, targets)
        return arr_loss.sum(axis=0)


class MeanLearnedPerceptualImagePatchSimilarity(EachLearnedPerceptualImagePatchSimilarity):

    def compute(self, x, targets):
        arr_loss = super().compute(x, targets)
        return arr_loss.mean(axis=0)


class MinLearnedPerceptualImagePatchSimilarity(EachLearnedPerceptualImagePatchSimilarity):

    def compute(self, x, targets):
        arr_loss = super().compute(x, targets)
        return np.min(arr_loss, axis=0)


class MaxLearnedPerceptualImagePatchSimilarity(EachLearnedPerceptualImagePatchSimilarity):

    def compute(self, x, targets):
        arr_loss = super().compute(x, targets)
        return np.max(arr_loss, axis=0)
