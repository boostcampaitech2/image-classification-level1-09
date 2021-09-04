import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from efficientnet_pytorch import EfficientNet

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
# CNN (EfficientNet)
class EFF00(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes = num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class EFF05(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5', num_classes = num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class EFF07(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7', num_classes = num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

# Transformer
class SWIN_LARGE(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("swin_large_patch4_window7_224", pretrained=True, num_classes = num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class CAIT_S24(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("cait_s24_224", pretrained=True, num_classes = num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
