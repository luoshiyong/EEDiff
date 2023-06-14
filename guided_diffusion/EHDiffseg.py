import torch
import torch.nn as nn
import torch.nn.functional as F
# from net.Res2Net_v1b import res2net50_v1b_26w_4s
import torchvision
import numpy as np
from torch import Tensor
from typing import Tuple, Optional
from einops import rearrange
from guided_diffusion.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    layer_norm,
)
class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        channels = channels//3
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh()
        )

    def forward_single_frame(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h

    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h

    def forward(self, x, h: Optional[Tensor]):
        if h is None:
            # print("x size = ",x.size())
            h = torch.zeros((x.size(0), x.size(-3)//3, x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        # x->  [4, 120, 168, 168]->[4,3,40,168,168]
        bb,cc,hh,ww = x.shape
        x = x.reshape(bb,3,cc//3,hh,ww)
        if x.ndim == 5:
            x,ps = self.forward_time_series(x, h)
            # print("ndim 5 out shape = ",x.shape)
            return x.reshape(bb,cc,hh,ww),ps
        else:
            return self.forward_single_frame(x, h)

class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gru = ConvGRU(channels)

    def forward(self, x, r: Optional[Tensor] = None):
        b, r = self.gru(x, r)
        return b, r
# 判断变量是否为None
def exists(x):
    return x is not None
class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch,group = 1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, groups=group),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, groups=group),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):

        x = self.conv(x)
        return x
from einops import rearrange
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch,emb_dim):
        super(up_conv, self).__init__()
        self.emb_dim = 256
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(emb_dim, out_ch))
            if exists(emb_dim)
            else None
        )
        self.newup = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x,emb=None):
        x = self.up(x)
        if exists(self.mlp) and exists(self.emb_dim):
            time_emb = self.mlp(emb)
            x = rearrange(time_emb, "b c -> b c 1 1") + x
        x = self.newup(x)
        return x


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # print(y.shape)
        y = self.fc(y)
        # print(y.shape)
        y = y.view(b, c, 1, 1).expand_as(x)
        # print("y shape = ",y.shape)
        # print("x shape = ",x.shape)
        return y



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch,kernel = 3,padding = 1):
        super(DEPTHWISECONV, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel,
                                    stride=1,
                                    padding=padding,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
from guided_diffusion.densenet121 import EdgeEncoder

class DFusion(nn.Module):
    def __init__(self, edge_dim,seg_dim,mid_dim,time_emb = False):
        super(DFusion, self).__init__()
        self.time_emb = time_emb
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(256, mid_dim))
            if self.time_emb
            else None
        )
        self.c1 = nn.Conv2d(edge_dim, mid_dim, 1)
        self.c11 = nn.Conv2d(mid_dim,edge_dim, 1)
        self.c3 = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, 3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True))
        self.c5 = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, 3, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True))
        self.c7 = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, 3, padding=7, dilation=7, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True))
        self.s1 = nn.Conv2d(seg_dim, mid_dim, 1)
        self.s11 = nn.Conv2d(mid_dim,seg_dim, 1)
        self.s3 = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1,bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True))
        self.se = SE_Block(mid_dim)
    def forward(self,edge,seg,emb = None):
        edge1 =self.c1(edge)
        seg1 = self.s1(seg)
        if self.time_emb and emb is not None:
            edge1 = edge1 + rearrange(self.mlp(emb),"b c -> b c 1 1")
        edge3 = self.c3(edge1)
        edge5 = self.c5(edge1)
        edge7 = self.c7(edge1)
        attn = self.se(edge3+edge5+edge7+seg1)
        seg_out = (edge3+edge5+edge7)*attn+seg1
        seg_out = self.s11(seg_out)+seg
        edge_out = self.s3(seg1)+edge1
        edge_out = self.c11(edge_out)+edge
        return edge_out,seg_out

"""
    edge encoder block 
"""

# 声明了一个残差网络
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# 上采样
def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


# 下采样
def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


# 基础的神经网络会用到的层，定义了层里面的两个基本操作，卷积和归一化以及激活函数
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


# 残差网络的构建
class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)


# 定义之前提到的ConvNeXt网络
class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)

class EHDiffseg(nn.Module):
    # res2net based encoder decoder
    def __init__(self, model_channels=64):
        super(EHDiffseg, self).__init__()
        # time emb
        time_embed_dim = 64 * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        # ---- x_e DenseNet Backbone ----
        self.densenet = EdgeEncoder(
                        dim=64,
                        channels=4,
                        dim_mults=(1, 2, 4,8),
                        use_convnext = True
        )
        # ---- x group gru encoder ----
        n1 = 60
        in_ch = 3
        group = 3
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(in_ch, filters[0], group)
        self.gru1 = BottleneckBlock(filters[0])
        self.Conv2 = conv_block(filters[0], filters[1], group)
        self.gru2 = BottleneckBlock(filters[1])
        self.Conv3 = conv_block(filters[1], filters[2], group)
        self.gru3 = BottleneckBlock(filters[2])
        self.Conv4 = conv_block(filters[2], filters[3], group)
        self.gru4 = BottleneckBlock(filters[3])
        self.Conv5 = conv_block(filters[3], filters[4], group)
        # seg head
        self.ffconv = nn.Sequential(
            nn.Conv2d(1472, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.ffconv2 = nn.Sequential(
            nn.Conv2d(752, 512, 3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 64, 1),
            nn.BatchNorm2d(64)
        )
        # edge upsample
        self.up1 = up_conv(512, 256,emb_dim=256)
        self.trans1 = nn.Sequential(
            # nn.Conv2d(512, 128, 3, padding=1, bias=False),
            DEPTHWISECONV(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = up_conv(512, 256,emb_dim=256)
        self.trans2 = nn.Sequential(
            # nn.Conv2d(256, 128, 3, padding=1, bias=False),
            DEPTHWISECONV(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up3 = up_conv(384, 256,emb_dim=256)
        self.trans3 = nn.Sequential(
            # nn.Conv2d(64, 32, 3, padding=1, bias=False),
            DEPTHWISECONV(64, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.up4 = up_conv(288, 128,emb_dim=256)
        self.outconv = nn.Sequential(
            nn.Conv2d(128, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, 1)
        )
        self.outconv2 = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x,timesteps=None, y=None):
        x_e = x
        x = x[:, 0:3, :, :]
        emb = self.time_embed(timestep_embedding(timesteps, 64))   # [b,256]
        # x_e encoder
        out = self.densenet(x_e,emb)
        x1 = out[0]         # [2,64,128,128]
        x2 = out[1]         # [2,128,64,64]
        x3 = out[2]         # [2,256,32,32]
        x4 = out[3]         # [2,512,16,16]
        # x encoder
        e1 = self.Conv1(x)
        e1, _ = self.gru1(e1)  # [2, 60, 256, 256]
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e2, _ = self.gru2(e2)  # [2,120,128,128]
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e3, _ = self.gru3(e3)   # [2,240,64,64]
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e4, _ = self.gru4(e4)   # [2,480,32,32]
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)     # [2,960,16,16]
        e5 =self.ffconv(torch.cat([e5,x4],dim=1)) #  [2,512,16,16]
        # --->seg head
        seg = F.interpolate(e5, size=e3.shape[2:], mode='bilinear',align_corners=False)
        seg = self.ffconv2(torch.cat([seg,e3],dim=1))    # 512+240=752->[2,64,64,64]
        seg = F.interpolate(seg, size=x.shape[2:], mode='bilinear',align_corners=False)
        seg = self.outconv2(seg)
        # edge head
        e5 = self.up1(e5,emb)     # [2,256,32,32]
        e5 = torch.cat([e5,self.trans1(x3)],dim=1)  # [2,512,32,32]
        e5 = self.up2(e5,emb)      # [2,256,64,64]
        e5 = torch.cat([e5,self.trans2(x2)],dim=1)  # [2,384,64,64]
        e5 = self.up3(e5,emb)      # [2,256,128,128]
        e5 = torch.cat([e5, self.trans3(x1)], dim=1)  # [2,288,128,128]
        e5 = self.up4(e5,emb)  # [2,128,256,256]
        e5 = self.outconv(e5)

        return  seg,e5
# deformable transformer block
from .defconv import DefC
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return DefC(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, bias=False)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.toq = conv3x3(512,256,1)
        self.tok = conv3x3(512, 256, 1)
        self.tov = conv3x3(512, 256, 1)
        self.res = nn.Conv2d(256,512,3,padding=1)
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        BB, NN, CC = x.shape
        x = x.reshape(BB,NN,16,16)
        B,C,H,W = x.shape
        q = self.toq(x).flatten(2).reshape(B, C//2, self.num_heads, H*W // self.num_heads).permute(0, 2, 1, 3)
        k = self.tok(x).flatten(2).reshape(B, C//2, self.num_heads, H*W // self.num_heads).permute(0, 2, 1, 3)
        v = self.tov(x).flatten(2).reshape(B, C//2, self.num_heads, H*W // self.num_heads).permute(0, 2, 1, 3)
        # print("k shape = ",k.shape)
        # B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, C//2, H*W)
        # print("x shape = ",x.shape)   # [4, 480, 441]
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.res(x.reshape(B,C//2,H,W)).flatten(2)
        return x


class TBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.reshape(B,C,H*W)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.reshape(B,C,H,W)

#
from functools import partial
class EHDiffseg_fusion(nn.Module):
    # res2net based encoder decoder
    def __init__(self,
            model_channels=64,
            channels=4,
            dim=64,
            dim_mults=(1, 2, 4, 8),
            resnet_block_groups=8,
            use_convnext=True,
            convnext_mult=2):
        super(EHDiffseg_fusion, self).__init__()
        # time emb
        time_embed_dim = 64 * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        # ---- x_e DenseNet Backbone ----
        # determine dimensions
        self.channels = channels

        init_dim = 64
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # print("dims = ",dims)
        in_out = list(zip(dims[:-1], dims[1:]))
        # print("in_out = ",in_out)
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # layers
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=256),
                        block_klass(dim_out, dim_out, time_emb_dim=256),
                        # Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out),  # if not is_last else nn.Identity(),
                    ]
                )
            )
        # ---- x group gru encoder ----
        n1 =   60
        in_ch = 3
        group = 3
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(in_ch, filters[0], group)
        self.gru1 = BottleneckBlock(filters[0])
        self.Conv2 = conv_block(filters[0], filters[1], group)
        self.gru2 = BottleneckBlock(filters[1])
        self.Conv3 = conv_block(filters[1], filters[2], group)
        self.gru3 = BottleneckBlock(filters[2])
        self.Conv4 = conv_block(filters[2], filters[3], group)
        self.gru4 = BottleneckBlock(filters[3])
        self.Conv5 = conv_block(filters[3], filters[4], group)
        # seg head
        self.ffconv = nn.Sequential(
            nn.Conv2d(1472, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.ffconv2 = nn.Sequential(
            nn.Conv2d(752, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 64, 1),
            nn.BatchNorm2d(64)
        )
        # edge upsample
        self.up1 = up_conv(512, 256, emb_dim=256)
        self.trans1 = nn.Sequential(
            # nn.Conv2d(512, 128, 3, padding=1, bias=False),
            DEPTHWISECONV(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = up_conv(512, 256, emb_dim=256)
        self.trans2 = nn.Sequential(
            # nn.Conv2d(256, 128, 3, padding=1, bias=False),
            DEPTHWISECONV(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up3 = up_conv(384, 256, emb_dim=256)
        self.trans3 = nn.Sequential(
            # nn.Conv2d(64, 32, 3, padding=1, bias=False),
            DEPTHWISECONV(64, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.up4 = up_conv(288, 128, emb_dim=256)
        self.outconv = nn.Sequential(
            nn.Conv2d(128, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, 1)
        )
        self.outconv2 = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )
        self.fusion1 = DFusion(64,120, 32)
        self.fusion2 = DFusion(128,240, 64)
        # self.fusion3 = DFusion(256,480, 128)
        # global attention
        # self.mst = TBlock(dim=256, num_heads=8)
    def forward(self, x, timesteps=None, y=None):
        x_e = x
        x = x[:, 0:3, :, :]
        emb = self.time_embed(timestep_embedding(timesteps, 64))  # [b,256]
        # x_e encoder
        ex = self.init_conv(x_e)
        # downsample
        x1 = self.downs[0][0](ex,emb)
        x1 = self.downs[0][1](x1, emb)
        x1 = self.downs[0][2](x1)       # [2,64,128,128]
        # 
        e1 = self.Conv1(x)
        e1, _ = self.gru1(e1)  # [2, 60, 256, 256]
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e2, _ = self.gru2(e2)  # [2,120,128,128]
        x1, e2 = self.fusion1(x1, e2)       # 128 level fusion

        x2 = self.downs[1][0](x1, emb)
        x2 = self.downs[1][1](x2, emb)
        x2 = self.downs[1][2](x2)       # [2,128,64,64]
        #
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e3, _ = self.gru3(e3)  # [2,240,64,64]
        x2, e3 = self.fusion2(x2, e3)   # 64 level fusion

        x3 = self.downs[2][0](x2, emb)
        x3 = self.downs[2][1](x3, emb)
        x3 = self.downs[2][2](x3)       # [2,256,32,32]
        #
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e4, _ = self.gru4(e4)  # [2,480,32,32]
        # x3, e4 = self.fusion3(x3, e4)  # 32 level fusion

        x4 = self.downs[3][0](x3, emb)
        x4 = self.downs[3][1](x4, emb)
        x4 = self.downs[3][2](x4)       # [2,512,16,16]



        # x encoder



        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)  # [2,960,16,16]
        e5 = self.ffconv(torch.cat([e5, x4], dim=1))  # [2,512,16,16]
        #
        # e5 = self.mst(e5)
        # --->seg head
        seg = F.interpolate(e5, size=e3.shape[2:], mode='bilinear', align_corners=False)
        seg = self.ffconv2(torch.cat([seg, e3], dim=1))  # 512+240=752->[2,64,64,64]
        seg = F.interpolate(seg, size=x.shape[2:], mode='bilinear', align_corners=False)
        seg = self.outconv2(seg)
        # edge head
        e5 = self.up1(e5, emb)  # [2,256,32,32]
        e5 = torch.cat([e5, self.trans1(x3)], dim=1)  # [2,512,32,32]
        e5 = self.up2(e5, emb)  # [2,256,64,64]
        e5 = torch.cat([e5, self.trans2(x2)], dim=1)  # [2,384,64,64]
        e5 = self.up3(e5, emb)  # [2,256,128,128]
        e5 = torch.cat([e5, self.trans3(x1)], dim=1)  # [2,288,128,128]
        e5 = self.up4(e5, emb)  # [2,128,256,256]
        e5 = self.outconv(e5)

        return seg, e5
ras = EHDiffseg_fusion(64).cuda()
print("params = ",sum(p.numel() for p in ras.parameters() if p.requires_grad))  # 45780699
x = torch.randn(2,4,256,256).cuda()
t = torch.from_numpy(np.array([10,32])).long()
seg,edge = ras(x,t.cuda())
print("seg shape = ",seg.shape)
print("edge shape = ",edge.shape)