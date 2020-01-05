import torch.nn.functional as F
import torch.nn as nn
import torch

class AlexNet1D(nn.Module):
    def __init__(self, num_classes=3):
        super(AlexNet1D, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(6)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True))

    def up_type(self, name):
        if name == 'reg':
            self.out = nn.Linear(512, 1)
            self.type = name
        elif name == 'cls':
            self.out = nn.Linear(512, self.num_classes)
            self.type = name
        else:
            self.out_reg = nn.Linear(512, 1)
            self.out_cls = nn.Linear(512, self.num_classes)
            self.type = name

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.type == 'both':
            out1 = self.out_reg(x)
            out2 = self.out_cls(x)
            return out1, out2
        else:
            out = self.out(x)
            return out

class ResNet1D(nn.Module):
    def __init__(self, layers=18):
        super(ResNet1D, self).__init__()

    def forward(self, input):
        pass

class VGG1D(nn.Module):
    def __init__(self):
        super(VGG1D, self).__init__()

    def forward(self, input):

        pass