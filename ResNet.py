import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data as data_utils


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=15, stride=stride, padding=7, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=15, stride=1, padding=7, bias=False),
            nn.BatchNorm1d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, numb_blocks=16, num_classes=2):
        super(ResNet, self).__init__()

        self.inchannel = 9

        #         self.conv1 = nn.Sequential(
        #             nn.Conv1d(51, 96, kernel_size = 15, stride=1, padding = 7, bias=False),
        #             nn.BatchNorm1d(96),
        #             nn.ReLU(),
        #         )

        self.layer1 = self.make_layer(ResidualBlock, 64, numb_blocks, stride=1)

        #         self.gru = nn.GRU(100, 1, num_layers=2, bidirectional = False, batch_first=True)

        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        #         out = self.conv1(x)

        out = self.layer1(x)

        #         out, h = self.gru(out)
        out = F.avg_pool1d(out, out.shape[2])

        out = out.view(out.size(0), -1)
        #         out = F.dropout(out, 0.5, training=self.training)

        out = self.fc(out)
        return out


def Resnet(numb_blocks):
    return ResNet(ResidualBlock, numb_blocks)