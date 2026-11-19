import torch
import torch.nn as nn


class CoordAttMeanMax(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAttMeanMax, self).__init__()
        self.pool_h_mean = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_mean = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1_mean = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1_mean = nn.BatchNorm2d(mip)
        self.conv2_mean = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.conv1_max = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1_max = nn.BatchNorm2d(mip)
        self.conv2_max = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Mean pooling branch
        x_h_mean = self.pool_h_mean(x)
        x_w_mean = self.pool_w_mean(x).permute(0, 1, 3, 2)
        y_mean = torch.cat([x_h_mean, x_w_mean], dim=2)
        y_mean = self.conv1_mean(y_mean)
        y_mean = self.bn1_mean(y_mean)
        y_mean = self.relu(y_mean)
        x_h_mean, x_w_mean = torch.split(y_mean, [h, w], dim=2)
        x_w_mean = x_w_mean.permute(0, 1, 3, 2)

        # Max pooling branch
        x_h_max = self.pool_h_max(x)
        x_w_max = self.pool_w_max(x).permute(0, 1, 3, 2)
        y_max = torch.cat([x_h_max, x_w_max], dim=2)
        y_max = self.conv1_max(y_max)
        y_max = self.bn1_max(y_max)
        y_max = self.relu(y_max)
        x_h_max, x_w_max = torch.split(y_max, [h, w], dim=2)
        x_w_max = x_w_max.permute(0, 1, 3, 2)

        # Apply attention
        x_h_mean = self.conv2_mean(x_h_mean).sigmoid()
        x_w_mean = self.conv2_mean(x_w_mean).sigmoid()
        x_h_max = self.conv2_max(x_h_max).sigmoid()
        x_w_max = self.conv2_max(x_w_max).sigmoid()

        # Expand to original shape
        x_h_mean = x_h_mean.expand(-1, -1, h, w)
        x_w_mean = x_w_mean.expand(-1, -1, h, w)
        x_h_max = x_h_max.expand(-1, -1, h, w)
        x_w_max = x_w_max.expand(-1, -1, h, w)

        # Combine outputs
        attention_mean = identity * x_w_mean * x_h_mean
        attention_max = identity * x_w_max * x_h_max

        # Sum the attention outputs
        return attention_mean + attention_max





import torch
import torch.nn as nn


# 哔哩哔哩CV缝合救星：2025.05.22B站视频
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class FreMLP(nn.Module):

    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq) #分离出频域数据的 幅度（magnitude）
        pha = torch.angle(x_freq) #分离出频域数据的 相位（phase）
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out


class Branch(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''

    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c

        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation,
                      stride=1, groups=self.dw_channel,
                      bias=True, dilation=dilation)  # the dconv
        )

    def forward(self, input):
        return self.branch(input)
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class DBlock(nn.Module):
    '''
    Change this block using Branch
    '''

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations=[1, 4, 9], extra_depth_wise=True):
        super().__init__()
        # we define the 2 branches
        self.dw_channel = DW_Expand * c

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1, groups=c,
                                    bias=True, dilation=1) if extra_depth_wise else nn.Identity()  # optional extra dw
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(self.dw_channel, DW_Expand=1, dilation=dilation))

        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0,
                      stride=1,
                      groups=1, bias=True, dilation=1),
        )
        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, adapter=None):

        y = inp
        x = self.norm1(inp)
        # x = self.conv1(self.extra_conv(x))
        x = self.extra_conv(self.conv1(x))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        # second step
        x = self.conv4(self.norm2(y))  # size [B, 2*C, H, W]
        x = self.sg2(x)  # size [B, C, H, W]
        x = self.conv5(x)  # size [B, C, H, W]
        x = y + x * self.gamma

        return x






class DualCoordAttentionGate(nn.Module):
    def __init__(self, inp, oup, groups=32, reduction=4, use_residual=True):
        super(DualCoordAttentionGate, self).__init__()
        self.use_residual = use_residual

        # 水平与垂直方向的平均/最大池化
        self.pool_h_mean = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_mean = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        # 压缩中间维度 mip
        mip = max(8, inp // groups)

        # 平均和最大通路共享的卷积模块（降低通道维度）
        self.shared_conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.shared_bn = nn.BatchNorm2d(mip)
        self.relu = nn.ReLU(inplace=True)

        # 分别用于水平和垂直方向的卷积恢复通道维度
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1)

        # 可学习门控机制，用于自适应融合 mean 与 max 注意力
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inp, inp // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp // reduction, 2, 1),  # 输出两个权重：mean 和 max
            nn.Softmax(dim=1)  # 对两个权重进行归一化
        )

        # 通道注意力模块（SE 类似）增强融合后的特征响应
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(oup, oup // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(oup // reduction, oup, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 平均池化路径
        x_h_mean = self.pool_h_mean(x)
        x_w_mean = self.pool_w_mean(x).permute(0, 1, 3, 2)
        y_mean = torch.cat([x_h_mean, x_w_mean], dim=2)
        y_mean = self.shared_conv1(y_mean)
        y_mean = self.shared_bn(y_mean)
        y_mean = self.relu(y_mean)
        x_h_mean, x_w_mean = torch.split(y_mean, [h, w], dim=2)
        x_w_mean = x_w_mean.permute(0, 1, 3, 2)
        x_h_mean = self.conv_h(x_h_mean).sigmoid()
        x_w_mean = self.conv_w(x_w_mean).sigmoid()
        attn_mean = x_h_mean * x_w_mean

        # 最大池化路径
        x_h_max = self.pool_h_max(x)
        x_w_max = self.pool_w_max(x).permute(0, 1, 3, 2)
        y_max = torch.cat([x_h_max, x_w_max], dim=2)
        y_max = self.shared_conv1(y_max)
        y_max = self.shared_bn(y_max)
        y_max = self.relu(y_max)
        x_h_max, x_w_max = torch.split(y_max, [h, w], dim=2)
        x_w_max = x_w_max.permute(0, 1, 3, 2)
        x_h_max = self.conv_h(x_h_max).sigmoid()
        x_w_max = self.conv_w(x_w_max).sigmoid()
        attn_max = x_h_max * x_w_max

        # 可学习权重融合 mean 和 max 路径
        gate_weights = self.gate(identity)  # 输出维度为 [B, 2, 1, 1]
        mean_weight = gate_weights[:, 0:1]
        max_weight = gate_weights[:, 1:2]
        attn = attn_mean * mean_weight + attn_max * max_weight

        # 应用注意力
        out = identity * attn

        # 通道注意力增强输出特征
        scale = self.channel_att(out)
        out = out * scale

        # 可选残差连接
        if self.use_residual:
            out = out + identity

        return out



import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
import math


class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   build_norm_layer(norm_layer, channel)[1])

    def forward(self, x):
        out = self.block(x)
        return out


class EdgeGaussianAggregation(nn.Module):
    def __init__(self, dim, size, sigma, norm_layer, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, norm_layer, act_layer)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out

    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
            for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()





import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
import math


class MultiScaleGaussian(nn.Module):
    def __init__(self, dim, sizes, sigmas):
        super().__init__()
        self.filters = nn.ModuleList()
        for size, sigma in zip(sizes, sigmas):
            kernel = self.build_kernel(size, sigma)
            conv = nn.Conv2d(dim, dim, kernel_size=size, padding=size // 2, groups=dim, bias=False)
            conv.weight.data = kernel.repeat(dim, 1, 1, 1)
            conv.weight.requires_grad = False
            self.filters.append(conv)

    def forward(self, x):
        return sum(f(x) for f in self.filters) / len(self.filters)

    def build_kernel(self, size, sigma):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
            for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()


class ScharrEdge(nn.Module):
    def __init__(self, dim):
        super().__init__()
        scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32)
        scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32)
        self.weight_x = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.weight_y = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.weight_x.weight.data = scharr_x.view(1, 1, 3, 3).repeat(dim, 1, 1, 1)
        self.weight_y.weight.data = scharr_y.view(1, 1, 3, 3).repeat(dim, 1, 1, 1)
        self.weight_x.weight.requires_grad = False
        self.weight_y.weight.requires_grad = False

    def forward(self, x):
        gx = self.weight_x(x)
        gy = self.weight_y(x)
        edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return edge


class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # (B,C,1,1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # (B,1,C)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # (B,C,1,1)
        return x * y.expand_as(x)


class EGAPlusPlus(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.gaussian = MultiScaleGaussian(dim, sizes=[3, 5], sigmas=[0.8, 1.2])
        self.scharr = ScharrEdge(dim)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()
        self.eca = ECA(dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            build_norm_layer(norm_layer, dim)[1],
            act_layer()
        )

    def forward(self, x):
        g = self.gaussian(x)
        e = self.scharr(x)
        fuse = self.norm(self.act(x + g + e))
        out = self.eca(fuse)
        return self.fuse(out)



import torch
import torch.nn as nn

# 哔哩哔哩：CV缝合救星
class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x
def get_norm_layer(norm_type, channels, num_groups):
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.InstanceNorm3d(channels)
class GBConv(nn.Module):
    def __init__(self, in_channels, norm_type='GN'):
        super(GBConv, self).__init__()

        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x

        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)
        x = x1 * x2
        x = self.block4(x)

        return x + residual


# 定义GBConv模块
class SpatialGuidedConvolution(nn.Module):
    def __init__(self, in_channels, norm_type='GN'):
        super(SpatialGuidedConvolution, self).__init__()

        # 自适应卷积核大小，增强对不同图像尺寸的适应性
        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU()
        )

        # 简单的几何先验引导模块
        self.sgc_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        residual = x

        # 常规卷积块
        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)

        # 添加空间引导卷积（SGC）
        sgc_attention = self.sgc_attention(x1)
        x = x1 * x2 * sgc_attention
        x = self.block4(x)

        return x + residual


import torch
import torch.nn as nn
from einops import rearrange


# CV缝合救星2025.06.08视频
# 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
class ACFMAttention(nn.Module):
    def __init__(self, dim, num_heads=2, bias=False):
        super(ACFMAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)

    # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv

        return output

class Myfusion(nn.Module):
    def __init__(self, dim=128,dim0=64, num_heads=2, bias=False):
        super(Myfusion, self).__init__()
        filters = [64, 128, 384, 640]

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature0 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.temperature1 = nn.Parameter(torch.ones(filters[1], 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(filters[1], 1, 1))

        self.conv1 = nn.Conv2d(filters[1]*2, filters[1], kernel_size=1)
        self.conv2 = nn.Conv2d(filters[1], filters[0], kernel_size=1)
        self.conv3 = nn.Conv2d(filters[0] * 2, filters[0], kernel_size=1)
        self.block1 = nn.Sequential(
            BottConv(filters[3], filters[1], filters[3] // 4, 3, 1, 1),
            get_norm_layer(norm_type='GN', channels=filters[1],num_groups=filters[1] // 16),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            BottConv(filters[2], filters[1], filters[2] // 4, 3, 1, 1),
            get_norm_layer(norm_type='GN', channels=filters[1],num_groups=filters[1] // 16),
            nn.ReLU()
        )

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)
#############################################
        self.qkv0 = nn.Conv3d(dim0, dim0 * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv0 = nn.Conv3d(dim0 * 3, dim0 * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim0 * 3,
                                     bias=bias)
        self.project_out0 = nn.Conv3d(dim0, dim0, kernel_size=(1, 1, 1), bias=bias)
        self.fc0 = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv0 = nn.Conv3d(9 * dim0 // self.num_heads, dim0, kernel_size=(3, 3, 3), bias=True,
                                   groups=dim0 // self.num_heads, padding=1)

        # 使用 nn.UpsamplingBilinear2d 进行上采样
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)  # 上采样2倍
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=4)  # 上采样2倍
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)  # 上采样2倍
    # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
    def forward(self, x1,x2,x3,x4):

        x33=self.upsample1(x3)
        x44 = self.upsample2(x4)

        xguide=self.block2(x33)*self.temperature1+ self.block1(x44)*self.temperature2  #128

        xguide0= self.conv2(xguide)
        xguide0 =self.upsample3(xguide0)

        # 在通道维度上拼接
        x_concat = torch.cat((x2, xguide), dim=1)
        x_concat = self.conv1(x_concat)
        b, c, h, w = x_concat.shape
        x = x_concat.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv
#############################################
        # 在通道维度上拼接
        x_concat0 = torch.cat((x1, xguide0), dim=1)
        x_concat0 = self.conv3(x_concat0)
        b0, c0, h0, w0 = x_concat0.shape
        x0 = x_concat0.unsqueeze(2)
        qkv0 = self.qkv_dwconv0(self.qkv0(x0))
        qkv0 = qkv0.squeeze(2)
        f_conv0 = qkv0.permute(0, 2, 3, 1)
        f_all0 = qkv0.reshape(f_conv0.shape[0], h0 * w0, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all0 = self.fc0(f_all0.unsqueeze(2))
        f_all0 = f_all0.squeeze(2)

        # local conv
        f_conv0 = f_all0.permute(0, 3, 1, 2).reshape(x0.shape[0], 9 * x0.shape[1] // self.num_heads, h0, w0)
        f_conv0 = f_conv0.unsqueeze(2)
        out_conv0 = self.dep_conv0(f_conv0)  # B, C, H, W
        out_conv0 = out_conv0.squeeze(2)

        # global SA
        q0, k0, v0 = qkv0.chunk(3, dim=1)

        q0 = rearrange(q0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k0 = rearrange(k0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v0 = rearrange(v0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q0 = torch.nn.functional.normalize(q0, dim=-1)
        k0 = torch.nn.functional.normalize(k0, dim=-1)

        attn0 = (q0 @ k0.transpose(-2, -1)) * self.temperature0
        attn0 = attn0.softmax(dim=-1)

        out0 = (attn0 @ v0)
        # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
        out0 = rearrange(out0, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h0, w=w0)
        out0 = out0.unsqueeze(2)
        out0 = self.project_out0(out0)
        out0 = out0.squeeze(2)
        output0 = out0 + out_conv0


        return output0,output

class Myfusion1(nn.Module):
    def __init__(self, dim=128,dim0=64, num_heads=2, bias=False):
        super(Myfusion1, self).__init__()
        filters = [64, 128, 384, 640]

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature0 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.temperature1 = nn.Parameter(torch.ones(filters[1], 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(filters[1], 1, 1))

        self.cordatt0= DualCoordAttentionGate(inp=filters[1], oup=filters[1])
        self.cordatt1= DualCoordAttentionGate(inp=filters[0], oup=filters[0])
        self.conv1 = nn.Conv2d(filters[1]*2, filters[1], kernel_size=1)
        self.conv2 = nn.Conv2d(filters[1], filters[0], kernel_size=1)
        self.conv3 = nn.Conv2d(filters[0] * 2, filters[0], kernel_size=1)
        self.block1 = nn.Sequential(
            BottConv(filters[3], filters[1], filters[3] // 4, 3, 1, 1),
            get_norm_layer(norm_type='GN', channels=filters[1],num_groups=filters[1] // 16),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            BottConv(filters[2], filters[1], filters[2] // 4, 3, 1, 1),
            get_norm_layer(norm_type='GN', channels=filters[1],num_groups=filters[1] // 16),
            nn.ReLU()
        )

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)
#############################################
        self.qkv0 = nn.Conv3d(dim0, dim0 * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv0 = nn.Conv3d(dim0 * 3, dim0 * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim0 * 3,
                                     bias=bias)
        self.project_out0 = nn.Conv3d(dim0, dim0, kernel_size=(1, 1, 1), bias=bias)
        self.fc0 = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv0 = nn.Conv3d(9 * dim0 // self.num_heads, dim0, kernel_size=(3, 3, 3), bias=True,
                                   groups=dim0 // self.num_heads, padding=1)

        # 使用 nn.UpsamplingBilinear2d 进行上采样
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)  # 上采样2倍
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=4)  # 上采样2倍
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)  # 上采样2倍
    # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
    def forward(self, x1,x2,x3,x4):

        x33=self.upsample1(x3)
        x44 = self.upsample2(x4)

        xguide=self.block2(x33)*self.temperature1+ self.block1(x44)*self.temperature2  #128

        xguide0= self.conv2(xguide)
        xguide0 =self.upsample3(xguide0)

        # 在通道维度上拼接
        x_concat = torch.cat((x2, xguide), dim=1)
        x_concat = self.conv1(x_concat)
        b, c, h, w = x_concat.shape
        x = x_concat.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv
#############################################
        # 在通道维度上拼接
        x_concat0 = torch.cat((x1, xguide0), dim=1)
        x_concat0 = self.conv3(x_concat0)
        b0, c0, h0, w0 = x_concat0.shape
        x0 = x_concat0.unsqueeze(2)
        qkv0 = self.qkv_dwconv0(self.qkv0(x0))
        qkv0 = qkv0.squeeze(2)
        f_conv0 = qkv0.permute(0, 2, 3, 1)
        f_all0 = qkv0.reshape(f_conv0.shape[0], h0 * w0, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all0 = self.fc0(f_all0.unsqueeze(2))
        f_all0 = f_all0.squeeze(2)

        # local conv
        f_conv0 = f_all0.permute(0, 3, 1, 2).reshape(x0.shape[0], 9 * x0.shape[1] // self.num_heads, h0, w0)
        f_conv0 = f_conv0.unsqueeze(2)
        out_conv0 = self.dep_conv0(f_conv0)  # B, C, H, W
        out_conv0 = out_conv0.squeeze(2)

        # global SA
        q0, k0, v0 = qkv0.chunk(3, dim=1)

        q0 = rearrange(q0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k0 = rearrange(k0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v0 = rearrange(v0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q0 = torch.nn.functional.normalize(q0, dim=-1)
        k0 = torch.nn.functional.normalize(k0, dim=-1)

        attn0 = (q0 @ k0.transpose(-2, -1)) * self.temperature0
        attn0 = attn0.softmax(dim=-1)

        out0 = (attn0 @ v0)
        out0 = rearrange(out0, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h0, w=w0)
        out0 = out0.unsqueeze(2)
        out0 = self.project_out0(out0)
        out0 = out0.squeeze(2)
        output0 = out0 + out_conv0

        output0= self.cordatt1(output0)
        output = self.cordatt0(output)
        return output0,output


class Myfusion2(nn.Module):
    def __init__(self, dim=128,dim0=64, num_heads=2, bias=False):
        super(Myfusion2, self).__init__()
        filters = [64, 128, 384, 640]

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature0 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.temperature1 = nn.Parameter(torch.ones(filters[1], 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(filters[1], 1, 1))

        self.cordatt0= CoordAttMeanMax(inp=filters[1], oup=filters[1])
        self.cordatt1= CoordAttMeanMax(inp=filters[0], oup=filters[0])
        self.conv1 = nn.Conv2d(filters[1]*2, filters[1], kernel_size=1)
        self.conv2 = nn.Conv2d(filters[1], filters[0], kernel_size=1)
        self.conv3 = nn.Conv2d(filters[0] * 2, filters[0], kernel_size=1)
        self.block1 = nn.Sequential(
            BottConv(filters[3], filters[1], filters[3] // 4, 3, 1, 1),
            get_norm_layer(norm_type='GN', channels=filters[1],num_groups=filters[1] // 16),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            BottConv(filters[2], filters[1], filters[2] // 4, 3, 1, 1),
            get_norm_layer(norm_type='GN', channels=filters[1],num_groups=filters[1] // 16),
            nn.ReLU()
        )

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)
#############################################
        self.qkv0 = nn.Conv3d(dim0, dim0 * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv0 = nn.Conv3d(dim0 * 3, dim0 * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim0 * 3,
                                     bias=bias)
        self.project_out0 = nn.Conv3d(dim0, dim0, kernel_size=(1, 1, 1), bias=bias)
        self.fc0 = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv0 = nn.Conv3d(9 * dim0 // self.num_heads, dim0, kernel_size=(3, 3, 3), bias=True,
                                   groups=dim0 // self.num_heads, padding=1)

        # 使用 nn.UpsamplingBilinear2d 进行上采样
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)  # 上采样2倍
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=4)  # 上采样2倍
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)  # 上采样2倍
    # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
    def forward(self, x1,x2,x3,x4):

        x33=self.upsample1(x3)
        x44 = self.upsample2(x4)

        xguide=self.block2(x33)*self.temperature1+ self.block1(x44)*self.temperature2  #128

        xguide0= self.conv2(xguide)
        xguide0 =self.upsample3(xguide0)

        # 在通道维度上拼接
        x_concat = torch.cat((x2, xguide), dim=1)
        x_concat = self.conv1(x_concat)
        b, c, h, w = x_concat.shape
        x = x_concat.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv
#############################################
        # 在通道维度上拼接
        x_concat0 = torch.cat((x1, xguide0), dim=1)
        x_concat0 = self.conv3(x_concat0)
        b0, c0, h0, w0 = x_concat0.shape
        x0 = x_concat0.unsqueeze(2)
        qkv0 = self.qkv_dwconv0(self.qkv0(x0))
        qkv0 = qkv0.squeeze(2)
        f_conv0 = qkv0.permute(0, 2, 3, 1)
        f_all0 = qkv0.reshape(f_conv0.shape[0], h0 * w0, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all0 = self.fc0(f_all0.unsqueeze(2))
        f_all0 = f_all0.squeeze(2)

        # local conv
        f_conv0 = f_all0.permute(0, 3, 1, 2).reshape(x0.shape[0], 9 * x0.shape[1] // self.num_heads, h0, w0)
        f_conv0 = f_conv0.unsqueeze(2)
        out_conv0 = self.dep_conv0(f_conv0)  # B, C, H, W
        out_conv0 = out_conv0.squeeze(2)

        # global SA
        q0, k0, v0 = qkv0.chunk(3, dim=1)

        q0 = rearrange(q0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k0 = rearrange(k0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v0 = rearrange(v0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q0 = torch.nn.functional.normalize(q0, dim=-1)
        k0 = torch.nn.functional.normalize(k0, dim=-1)

        attn0 = (q0 @ k0.transpose(-2, -1)) * self.temperature0
        attn0 = attn0.softmax(dim=-1)

        out0 = (attn0 @ v0)
        # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
        out0 = rearrange(out0, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h0, w=w0)
        out0 = out0.unsqueeze(2)
        out0 = self.project_out0(out0)
        out0 = out0.squeeze(2)
        output0 = out0 + out_conv0

        output0= self.cordatt1(output0)
        output = self.cordatt0(output)
        return output0,output


class Myfusionl(nn.Module):
    def __init__(self, dim=128,dim0=64, num_heads=2, bias=False):
        super(Myfusionl, self).__init__()
        filters = [256, 512, 1024, 2048]

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature0 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.temperature1 = nn.Parameter(torch.ones(filters[1], 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(filters[1], 1, 1))

        self.cordatt0= DualCoordAttentionGate(inp=filters[1], oup=filters[1])
        self.cordatt1= DualCoordAttentionGate(inp=filters[0], oup=filters[0])
        self.conv1 = nn.Conv2d(filters[1]*2, filters[1], kernel_size=1)
        self.conv2 = nn.Conv2d(filters[1], filters[0], kernel_size=1)
        self.conv3 = nn.Conv2d(filters[0] * 2, filters[0], kernel_size=1)
        self.block1 = nn.Sequential(
            BottConv(filters[3], filters[1], filters[3] // 4, 3, 1, 1),
            get_norm_layer(norm_type='GN', channels=filters[1],num_groups=filters[1] // 16),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            BottConv(filters[2], filters[1], filters[2] // 4, 3, 1, 1),
            get_norm_layer(norm_type='GN', channels=filters[1],num_groups=filters[1] // 16),
            nn.ReLU()
        )

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)
#############################################
        self.qkv0 = nn.Conv3d(dim0, dim0 * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv0 = nn.Conv3d(dim0 * 3, dim0 * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim0 * 3,
                                     bias=bias)
        self.project_out0 = nn.Conv3d(dim0, dim0, kernel_size=(1, 1, 1), bias=bias)
        self.fc0 = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv0 = nn.Conv3d(9 * dim0 // self.num_heads, dim0, kernel_size=(3, 3, 3), bias=True,
                                   groups=dim0 // self.num_heads, padding=1)

        # 使用 nn.UpsamplingBilinear2d 进行上采样
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)  # 上采样2倍
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=4)  # 上采样2倍
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)  # 上采样2倍
    # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
    def forward(self, x1,x2,x3,x4):

        x33=self.upsample1(x3)
        x44 = self.upsample2(x4)

        xguide=self.block2(x33)*self.temperature1+ self.block1(x44)*self.temperature2  #128

        xguide0= self.conv2(xguide)
        xguide0 =self.upsample3(xguide0)

        # 在通道维度上拼接
        x_concat = torch.cat((x2, xguide), dim=1)
        x_concat = self.conv1(x_concat)
        b, c, h, w = x_concat.shape
        x = x_concat.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv
#############################################
        # 在通道维度上拼接
        x_concat0 = torch.cat((x1, xguide0), dim=1)
        x_concat0 = self.conv3(x_concat0)
        b0, c0, h0, w0 = x_concat0.shape
        x0 = x_concat0.unsqueeze(2)
        qkv0 = self.qkv_dwconv0(self.qkv0(x0))
        qkv0 = qkv0.squeeze(2)
        f_conv0 = qkv0.permute(0, 2, 3, 1)
        f_all0 = qkv0.reshape(f_conv0.shape[0], h0 * w0, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all0 = self.fc0(f_all0.unsqueeze(2))
        f_all0 = f_all0.squeeze(2)

        # local conv
        f_conv0 = f_all0.permute(0, 3, 1, 2).reshape(x0.shape[0], 9 * x0.shape[1] // self.num_heads, h0, w0)
        f_conv0 = f_conv0.unsqueeze(2)
        out_conv0 = self.dep_conv0(f_conv0)  # B, C, H, W
        out_conv0 = out_conv0.squeeze(2)

        # global SA
        q0, k0, v0 = qkv0.chunk(3, dim=1)

        q0 = rearrange(q0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k0 = rearrange(k0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v0 = rearrange(v0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q0 = torch.nn.functional.normalize(q0, dim=-1)
        k0 = torch.nn.functional.normalize(k0, dim=-1)

        attn0 = (q0 @ k0.transpose(-2, -1)) * self.temperature0
        attn0 = attn0.softmax(dim=-1)

        out0 = (attn0 @ v0)
        # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
        out0 = rearrange(out0, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h0, w=w0)
        out0 = out0.unsqueeze(2)
        out0 = self.project_out0(out0)
        out0 = out0.squeeze(2)
        output0 = out0 + out_conv0

        output0= self.cordatt1(output0)
        output = self.cordatt0(output)
        return output0,output

if __name__ == "__main__":
    # 模拟输入参数
    batch_size = 1
    channels = 32
    height = 256
    width = 256
    num_heads = 4

    # 创建输入张量：形状 [B, C, H, W]
    x = torch.randn(batch_size, channels, height, width)

    x1 = torch.randn(1, 256, 128, 128)
    x2 = torch.randn(1, 512, 64, 64)
    x3 = torch.randn(1, 1024, 32, 32)
    x4 = torch.randn(1, 2048, 16, 16)

    model = Myfusionl(dim=512, dim0=256,num_heads=num_heads, bias=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x1 = x1.to(device)
    x2 = x2.to(device)
    x3 = x3.to(device)
    x4 = x4.to(device)
    model = model.to(device)

    # 前向传播
    x22,x33 = model(x1,x2,x3,x4)
    print(model)
    print("哔哩哔哩:CV缝合救星")

    # 打印输入输出形状
    print("输入形状:", x.shape)  # [B, C, H, W]
    print("输出形状:", x22.shape)  # [B, C, H, W]
    print("输出形状:", x33.shape)  # [B, C, H, W]


