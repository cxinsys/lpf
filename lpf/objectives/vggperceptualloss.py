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
except (ImportError, ModuleNotFoundError):
    err_msg = "Cannot use Vgg16PerceptualLoss objectives, " \
              "since it fails to import torch or tortchvision."
    print(err_msg)

from lpf.objectives import Objective


# class Vgg16PerceptualLoss(torch.nn.Module):
#     def __init__(self, resize=True):
#         super(Vgg16PerceptualLoss, self).__init__()
#         blocks = []
#         blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
#         blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
#         blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
#         blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
#         for bl in blocks:
#             for p in bl.parameters():
#                 p.requires_grad = False
#
#         self.blocks = torch.nn.ModuleList(blocks)
#         self.transform = torch.nn.functional.interpolate
#
#         self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
#         self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
#
#         self.resize = resize
#
#     def forward(self, input, target):
#         if input.shape[1] != 3:
#             input = input.repeat(1, 3, 1, 1)
#             target = target.repeat(1, 3, 1, 1)
#         input = (input-self.mean) / self.std
#         target = (target-self.mean) / self.std
#         if self.resize:
#             input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
#             target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
#         loss = 0.0
#         x = input
#         y = target
#         for block in self.blocks:
#             x = block(x)
#             y = block(y)
#             loss += torch.nn.functional.l1_loss(x, y)
#         return loss

class Vgg16PerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(Vgg16PerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
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
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        if not feature_layers:
            feature_layers = [0, 1, 2, 3]

        loss = 0.0
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
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class EachVgg16PerceptualLoss(Objective):
    
    def __init__(self, coeff=None, device=None):
        
        if not coeff:
            coeff = 1.0
        
        self._coeff = coeff
        
        super().__init__(device=device)
        self.model = Vgg16PerceptualLoss(resize=False).to(self.device)
        self.to_tensor = transforms.ToTensor()

    def compute(self, x, targets, coeff=None):
        
        if not coeff:
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

        
class SumVgg16PerceptualLoss(EachVgg16PerceptualLoss):
        
    def compute(self, x, targets):
        each_loss = super().compute(x, targets)    
        return each_loss.sum()
    
    
class MeanVgg16PerceptualLoss(EachVgg16PerceptualLoss):
        
    def compute(self, x, targets):
        arr_loss = super().compute(x, targets)    
        return np.mean(arr_loss)
    
    
class MinVgg16PerceptualLoss(EachVgg16PerceptualLoss):
        
    def compute(self, x, targets):
        arr_loss = super().compute(x, targets)    
        return np.min(arr_loss)
    
    
class MaxVgg16PerceptualLoss(EachVgg16PerceptualLoss):
        
    def compute(self, x, targets):
        arr_loss = super().compute(x, targets)    
        return np.max(arr_loss)
    
