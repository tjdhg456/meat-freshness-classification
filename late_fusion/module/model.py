import torch.nn.functional as F
import torch.nn as nn
import torch

##################################################
############## AlexNet Module #####################
##################################################
class AlexNet1D(nn.Module):
    def __init__(self, in_channel, train_rule, init_weights=True):
        super(AlexNet1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(6)

        self.train_rule = train_rule

        in_fc = 256 * 6
        if self.train_rule == 'mid':
            self.conv1 = nn.Conv1d(1, 256, kernel_size=1, padding=0)
            in_fc = 256 * 10
            out_fc = 512
            out_fc2 = 512
        elif self.train_rule == 'late':
            in_fc = 256 * 6
            out_fc = 32
            out_fc2 = 36
        else:
            pass

        self.embedding = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_fc, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, out_fc),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(out_fc2, 3)

        if init_weights:
            self._initialize_weights()



    def forward(self, x, aux):
        x = self.features(x)
        x = self.avgpool(x)

        if self.train_rule == 'mid':
            aux = self.conv1(aux)
            x = torch.cat([x, aux], dim=2)

        x = torch.flatten(x, 1)
        emb = self.embedding(x)
        if self.train_rule == 'late':
            aux = torch.squeeze(aux)
            emb = torch.cat([emb, aux], dim=1)
        out = self.classifier(emb)
        return out


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
