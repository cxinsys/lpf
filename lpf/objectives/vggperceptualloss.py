"""
References
- https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
- https://gist.github.com/brucemuller/37906a86526f53ec7f50af4e77d025c9
"""

import gc
import numpy as np

try:
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.models import VGG16_Weights
except (ImportError, ModuleNotFoundError) as err:

    err_msg = "Cannot use Vgg16PerceptualLoss objectives, " \
              "since it fails to import torch or torchvision."
    print(err)
    print(err_msg)

from lpf.objectives import Objective


class Vgg16PerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(Vgg16PerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=None, style_layers=None):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        if feature_layers is None:
            # feature_layers = [0, 1, 2, 3]
            feature_layers = [2]

        if style_layers is None:
            style_layers = [0, 1, 2, 3]

        loss = torch.zeros((len(input))).to(device=input.device)
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                gram_y = gram_y.expand_as(gram_x)
                elementwise_loss = torch.nn.functional.l1_loss(gram_x, gram_y, reduction='none')
                batch_loss = elementwise_loss.mean(dim=(1, 2))
                loss += batch_loss
        return loss


class EachVgg16PerceptualLoss(Objective):

    def __init__(self, coeff=None, device=None):

        if coeff is None:
            coeff = 1.0

        self._coeff = coeff

        super().__init__(device=device)
        self.model = Vgg16PerceptualLoss(resize=False).to(self.device)
        self.to_tensor = transforms.ToTensor()

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
                arr_loss.append(self.model(x, target).detach().cpu().numpy())

        if "cuda" in self.device:  # [!] self.device is not torch.device
            torch.cuda.empty_cache()

        arr_loss = np.array(arr_loss)
        del target
        del x
        del arr_img
        del tmp
        gc.collect()
        return coeff * arr_loss


class SumVgg16PerceptualLoss(EachVgg16PerceptualLoss):

    def compute(self, x, targets):
        arr_loss = super().compute(x, targets)
        return arr_loss.sum(axis=0)


class MeanVgg16PerceptualLoss(EachVgg16PerceptualLoss):

    def compute(self, x, targets):
        arr_loss = super().compute(x, targets)
        return np.mean(arr_loss, axis=0)


class MinVgg16PerceptualLoss(EachVgg16PerceptualLoss):

    def compute(self, x, targets):
        arr_loss = super().compute(x, targets)
        return np.min(arr_loss, axis=0)


class MaxVgg16PerceptualLoss(EachVgg16PerceptualLoss):

    def compute(self, x, targets):
        arr_loss = super().compute(x, targets)
        return np.max(arr_loss, axis=0)