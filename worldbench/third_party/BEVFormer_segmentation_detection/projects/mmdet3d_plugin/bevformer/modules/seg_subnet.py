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
    def __init__(self, inC, outC):
        super(SegEncode, self).__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
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
        x = self.conv1(x) #torch.Size([2, 64, 200, 400])
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x) #torch.Size([2, 64, 100, 200])
        x = self.layer2(x1) #torch.Size([2, 128, 50, 100])
        x2 = self.layer3(x) #torch.Size([2, 256, 25, 50])

        x = self.up1(x2, x1) #torch.Size([2, 256, 100, 200])
        x = self.up2(x) #torch.Size([2, 4, 200, 400]) 语义分割预测特征图

        return x

@SEG_ENCODER.register_module()
class SegEncode_v1(nn.Module):

    def __init__(self, inC, outC):
        super(SegEncode_v1, self).__init__()
        self.seg_head = nn.Sequential(
            nn.Conv2d(inC, inC, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, outC , kernel_size=1))

    def forward(self, x):

        return self.seg_head(x)


import math
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16


@SEG_ENCODER.register_module()
class DeconvEncode(BaseModule):
    """The neck used in `CenterNet <https://arxiv.org/abs/1904.07850>`_ for
    object classification and box regression.

    Args:
         in_channel (int): Number of input channels.
         num_deconv_filters (tuple[int]): Number of filters per stage.
         num_deconv_kernels (tuple[int]): Number of kernels per stage.
         use_dcn (bool): If True, use DCNv2. Default: True.
         init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channel,
                 num_deconv_filters,
                 num_deconv_kernels,
                 outC=4,
                 use_dcn=True,
                 init_cfg=None):
        super(DeconvEncode, self).__init__(init_cfg)
        assert len(num_deconv_filters) == len(num_deconv_kernels)
        self.fp16_enabled = False
        self.use_dcn = use_dcn
        self.in_channel = in_channel
        self.deconv_layers = self._make_deconv_layer(num_deconv_filters,
                                                     num_deconv_kernels)

        self.seg_head = nn.Sequential(
            nn.Conv2d(num_deconv_filters[-1], num_deconv_filters[-1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_deconv_filters[-1], outC , kernel_size=1))

    def _make_deconv_layer(self, num_deconv_filters, num_deconv_kernels):
        """use deconv layers to upsample backbone's output."""
        layers = []
        for i in range(len(num_deconv_filters)):
            feat_channel = num_deconv_filters[i]
            conv_module = ConvModule(
                self.in_channel,
                feat_channel,
                3,
                padding=1,
                conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
                norm_cfg=dict(type='BN'))
            layers.append(conv_module)
            upsample_module = ConvModule(
                feat_channel,
                feat_channel,
                num_deconv_kernels[i],
                stride=2,
                padding=1,
                conv_cfg=dict(type='deconv'),
                norm_cfg=dict(type='BN'))
            layers.append(upsample_module)
            self.in_channel = feat_channel

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (
                                    1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # self.use_dcn is False
            elif not self.use_dcn and isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    @auto_fp16()
    def forward(self, inputs):
        outs = self.deconv_layers(inputs)
        outs = self.seg_head(outs)
        return outs

"""

dict(
type='DeconvEncode',
in_channel=256,
num_deconv_filters=(256, 128, 64),
num_deconv_kernels=(4, 4, 4),
use_dcn=True),
        
"""
