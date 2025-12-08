import torch
import torch.nn as nn

from torchvision.models.resnet import resnet18
from ..modules.builder import SEG_ENCODER

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1) #相当与通道维度上连接，以弥补因为使用mb导致的卷积信息丢失。
        return self.conv(x1)

@SEG_ENCODER.register_module()
class SegEncode(nn.Module):
    def __init__(self, inC, outC, size):
        super(SegEncode, self).__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
        self.up_sampler = nn.Upsample(size=size, mode='bilinear', align_corners=True)
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x): #torch.Size([2, 256, 200, 400])
        x = self.up_sampler(x)
        x = self.conv1(x) #torch.Size([2, 64, 200, 400])
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x) #torch.Size([2, 64, 100, 200])
        x = self.layer2(x1) #torch.Size([2, 128, 50, 100])
        x2 = self.layer3(x) #torch.Size([2, 256, 25, 50])

        x = self.up1(x2, x1) #torch.Size([2, 256, 100, 200])
        x = self.up2(x) #torch.Size([2, 4, 200, 400]) 语义分割预测特征图

        return x
