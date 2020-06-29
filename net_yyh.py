import torch.nn as nn
import torchvision
from config_cifar10 import *
# from config import *
import torch.nn.functional as F




class ResNet50_minist(nn.Module):
    def __init__(self, class_num):
        super(ResNet50_minist, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.just_for_mnist = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu
        )
        self.base = nn.Sequential(*list(resnet50.children())[4:-2])
        # print(self.base)
        self.classifier = nn.Linear(2048, class_num)

    def forward(self, x):
        x = self.just_for_mnist(x)
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        y = self.classifier(f)
        return y


class ResNet50_original(nn.Module):
    def __init__(self, class_num):
        super(ResNet50_original, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, class_num)

    def forward(self, x):
        x = self.base(x)    # (1,3,224,224) -> (1,2048,7,7)
        x = F.avg_pool2d(x, x.size()[2:])  # (1,2048,7,7) ->  (1,2048,1,1)
        f = x.view(x.size(0), -1)   #   (1,2048,1,1) ->  (1,2048)
        y = self.classifier(f)   # (1,2048) -> (num_of_class)
        return y

