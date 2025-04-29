import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width):
        super(ResNeXtBottleneck, self).__init__()
        D = int(math.floor(out_channels * (base_width / 64.0))) * cardinality

        self.conv1 = nn.Conv2d(in_channels, D, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D)
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(D)
        self.conv3 = nn.Conv2d(D, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNeXt(nn.Module):
    def __init__(self, layers, cardinality=32, base_width=4, num_classes=1000):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.base_width = base_width

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResNeXtBottleneck, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(ResNeXtBottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(ResNeXtBottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(ResNeXtBottleneck, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResNeXtBottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.cardinality, self.base_width))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, 1, self.cardinality, self.base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnext50_32x4d(num_classes=1000):
    return ResNeXt(layers=[3, 4, 6, 3], cardinality=32, base_width=4, num_classes=num_classes)
