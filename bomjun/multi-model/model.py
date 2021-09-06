import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet
import timm

class EfficientNetModel(nn.Module):
    """
    Mask Model - output dimension: 3
    Gender Model - output dimension: 2
    Age Model = output dimension: 3
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b7', in_channels=3, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


