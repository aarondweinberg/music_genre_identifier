# A version of ResNet18 that takes TWO greyscale images as inputs
# For images of smaller height (<=32 pixels, or <=64 pixels) stride and maxpool are adjusted
# to prevent premature flattening

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet18GrayBackbone(nn.Module):
    def __init__(self, input_height=128):
        super(ResNet18GrayBackbone, self).__init__()
        self.in_channels = 64

        if input_height <= 32:
            conv1_stride = 1
            self.use_maxpool = False
        elif input_height <= 64:
            conv1_stride = 2
            self.use_maxpool = False
        else:
            conv1_stride = 2
            self.use_maxpool = True

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=conv1_stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if self.use_maxpool:
            out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # shape: (batch_size, 512)
        return out


class DualBranchResNet18Gray(nn.Module):
    def __init__(self, num_classes, input_height1=128, input_height2=128):
        super(DualBranchResNet18Gray, self).__init__()
        self.branch1 = ResNet18GrayBackbone(input_height=input_height1)
        self.branch2 = ResNet18GrayBackbone(input_height=input_height2)
        self.fc = nn.Linear(512 * 2, num_classes)

    def forward(self, x):
        x1, x2 = x  # Each is (batch_size, 1, H, W)
        f1 = self.branch1(x1)  # shape: (batch_size, 512)
        f2 = self.branch2(x2)
        combined = torch.cat([f1, f2], dim=1)  # shape: (batch_size, 1024)
        return self.fc(combined)
