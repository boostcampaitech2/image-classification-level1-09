import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import math
import timm #'swin_large_patch4_window7_224' - in_features = 1536


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
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

# Resnet 50
class resnet50(nn.Module):
    def __init__(self, num_classes, freeze):
        super().__init__()
        self.freeze = freeze
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        self.init_param()

    def init_param(self):
        torch.nn.init.kaiming_uniform_(self.resnet50.fc.weight)
        stdv = 1./math.sqrt(self.resnet50.fc.weight.size(1))
        self.resnet50.fc.bias.data.uniform_(-stdv, stdv)
        # if self.freeze:
        #     for param in self.resnet50.parameters():
        #         param.requies_grad = False

    def forward(self, x):
        return self.resnet50(x)

# Resnet 152
class resnet152(nn.Module):
    def __init__(self, num_classes, freeze):
        super().__init__()
        self.freeze = freeze
        self.resnet152 = torchvision.models.resnet152(pretrained=True)
        self.resnet152.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        self.init_param()

    def init_param(self):
        torch.nn.init.kaiming_uniform_(self.resnet152.fc.weight)
        stdv = 1./math.sqrt(self.resnet152.fc.weight.size(1))
        self.resnet152.fc.bias.data.uniform_(-stdv, stdv)
        # if self.freeze:
        #     for param in self.resnet152.parameters():
        #         param.requies_grad = False

    def forward(self, x):
        return self.resnet152(x)


# dm_nfnet_f3
class dm_nfnet_f3(nn.Module):
    def __init__(self, num_classes, freeze):
        super().__init__()
        self.freeze = freeze
        self.dm_nfnet_f3 = timm.create_model('dm_nfnet_f3', pretrained=True)
        self.dm_nfnet_f3.head.fc = torch.nn.Linear(in_features=3072, out_features=num_classes, bias=True)
        self.init_param()

    def init_param(self):
        torch.nn.init.kaiming_uniform_(self.dm_nfnet_f3.head.fc.weight)
        stdv = 1./math.sqrt(self.dm_nfnet_f3.head.fc.weight.size(1))
        self.dm_nfnet_f3.head.fc.bias.data.uniform_(-stdv, stdv)
        # if self.freeze:
        #     for param in self.dm_nfnet_f3.parameters():
        #         param.requies_grad = False

    def forward(self, x):
        return self.dm_nfnet_f3(x)

# swin_large_patch4_window7_224
class swin_large_patch4_window7_224(nn.Module):
    def __init__(self, num_classes, freeze):
        super().__init__()
        self.freeze = freeze
        self.model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
        self.model.head = torch.nn.Linear(in_features=1536, out_features=num_classes, bias=True)
        self.init_param()

    def init_param(self):
        torch.nn.init.kaiming_uniform_(self.model.head.weight)
        stdv = 1./math.sqrt(self.model.head.weight.size(1))
        self.model.head.bias.data.uniform_(-stdv, stdv)
        if self.freeze:
            for param in self.model.parameters():
                param.requies_grad = False

    def forward(self, x):
        return self.model(x)