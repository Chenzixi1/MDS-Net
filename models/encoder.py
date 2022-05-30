from typing import List

import torch
import torch.nn as nn


def get_norm(norm, out_channels, **kwargs):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
        kwargs: Additional parameters in normalization layers,
            such as, eps, momentum

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
        }[norm]
    return norm(out_channels, **kwargs)

def get_activation(activation):
    """
    Only support `ReLU` and `LeakyReLU` now.

    Args:
        activation (str or callable):

    Returns:
        nn.Module: the activation layer
    """

    act = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
    }[activation]
    return act(inplace=True)

def c2_xavier_fill(module: nn.Module):
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


class DilatedEncoder(nn.Module):
    """
    Dilated Encoder for YOLOF.

    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
          which are 1x1 conv + 3x3 conv
        - the dilated residual block
    """

    def __init__(self, in_channels):
        super(DilatedEncoder, self).__init__()
        # fmt: off
        self.in_channels = in_channels
        self.encoder_channels = 512
        self.block_mid_channels = 128
        self.num_residual_blocks = 4
        self.block_dilations = [2, 4, 6, 8]
        # self.num_residual_blocks = 3
        # self.block_dilations = [2, 4, 6]
        self.norm_type = "BN"
        self.act_type = "ReLU"
        self.out_channels = 512
        # fmt: on
        assert len(self.block_dilations) == self.num_residual_blocks

        # init
        self._init_layers()
        self._init_weight()

    def _init_layers(self):
        self.lateral_conv = nn.Conv2d(self.in_channels,
                                      self.encoder_channels,
                                      kernel_size=1)
        self.lateral_norm = get_norm(self.norm_type, self.encoder_channels)
        self.fpn_conv = nn.Conv2d(self.encoder_channels,
                                  self.encoder_channels,
                                  kernel_size=3,
                                  padding=1)
        self.fpn_norm = get_norm(self.norm_type, self.encoder_channels)
        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            encoder_blocks.append(
                Bottleneck(
                    self.encoder_channels,
                    self.block_mid_channels,
                    dilation=dilation,
                    norm_type=self.norm_type,
                    act_type=self.act_type
                )
            )
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def _init_weight(self):
        c2_xavier_fill(self.lateral_conv)
        c2_xavier_fill(self.fpn_conv)

        for m in [self.lateral_norm, self.fpn_norm]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        for m in self.dilated_encoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        out = self.lateral_norm(self.lateral_conv(feature))
        out = self.fpn_norm(self.fpn_conv(out))
        return self.dilated_encoder_blocks(out)


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels: int = 512,
                 mid_channels: int = 128,
                 dilation: int = 1,
                 norm_type: str = 'BN',
                 act_type: str = 'ReLU'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            get_norm(norm_type, mid_channels),
            get_activation(act_type)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            get_norm(norm_type, mid_channels),
            get_activation(act_type)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            get_norm(norm_type, in_channels),
            get_activation(act_type)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out
