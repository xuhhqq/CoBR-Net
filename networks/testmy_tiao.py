import os
import setting
os.environ["CUDA_VISIBLE_DEVICES"] = setting.setting['gpu']
import torch.nn as nn
import torch

from networks.over import overlock_t
from networks.gaosi import EGAPlusPlus,Myfusion1,DBlock
from functools import partial
import torch.nn.functional as F
from networks.LEGNet import Stem
nonlinearity = partial(F.relu, inplace=True)


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

#上采样模块 用于解码器的上采样阶段
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x











class CoBR(nn.Module):
    def __init__(self, num_classes=1):
        super(CoBR, self).__init__()

        filters = [64, 128, 384, 640]
        self.backboon = overlock_t(pretrained=True)
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[1], filters[0])
        self.stem = Stem(
            in_chans=3,
            stem_dim=64,
            act_layer=nn.ReLU,
            norm_layer=dict(type='BN', requires_grad=True)
        )
        self.db = DBlock(c=64)
        self.fu=Myfusion1(dim=128, dim0=64,num_heads=4, bias=True)

        self.ega_e1 = EGAPlusPlus(dim=filters[0], norm_layer=dict(type='BN', requires_grad=True), act_layer=nn.ReLU)
        self.ega_e2 = EGAPlusPlus(dim=filters[1], norm_layer=dict(type='BN', requires_grad=True), act_layer=nn.ReLU)
        self.ega_e3 = EGAPlusPlus(dim=filters[2], norm_layer=dict(type='BN', requires_grad=True), act_layer=nn.ReLU)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x, return_aux=False):
        # Encoder
        xedge = self.stem(x)
        xedge = self.db(xedge)

        x4 = self.backboon(x)
        e4 = x4[3]
        e3 = x4[2]
        e22 = x4[1]
        e11 = x4[0]

        # GAIM
        e1, e2 = self.fu(e11, e22, e3, e4)

        # Decoder
        d4 = self.decoder4(e4) + e3 + self.ega_e3(e3)
        d3 = self.decoder3(d4) + e2 + self.ega_e2(e2)
        d2 = self.decoder2(d3) + e1 + self.ega_e1(e1)

        dfusion = torch.cat([d2, xedge], dim=1)
        d1 = self.decoder1(dfusion)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        if return_aux:
            aux = {
                # GAIM 输入特征
                'e11_before_gaim': e11,
                'e22_before_gaim': e22,

                # GAIM 输出特征
                'e1_after_gaim': e1,
                'e2_after_gaim': e2,
                # 新增别名，兼容grad_gaim.py脚本
                "e11": e11,
                "e22": e22,
                "e1": e1,
                "e2": e2,
                # 解码特征
                'd2': d2,
                'd1': d1,
            }

            return out, aux

        return torch.sigmoid(out)









