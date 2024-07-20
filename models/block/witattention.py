import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np
from .torch_wavelets import DWT_2D, IDWT_2D

class WaveAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.sr_ratio = sr_ratio

        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim//4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim//4),
            nn.ReLU(inplace=True),
        )
        self.filter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.kv_embed = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2)
        )
        self.proj = nn.Linear(dim+dim//4, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B,C,H,W= x.shape
        x = x.view(B, -1, H * W).permute(0, 2, 1)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        # print(x.shape)
        x_dwt = self.dwt(self.reduce(x))
        x_dwt = self.filter(x_dwt)

        x_idwt = self.idwt(x_dwt)
        x_idwt = x_idwt.view(B, -1, x_idwt.size(-2)*x_idwt.size(-1)).transpose(1, 2)

        kv = self.kv_embed(x_dwt).reshape(B, C, -1).permute(0, 2, 1)
        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(torch.cat([x, x_idwt], dim=-1))
        x = x.permute(0, 2, 1)
        x = x.view(B, C, H, W)
        
        return x

class WaveAttention2(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.sr_ratio = sr_ratio

        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim//4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim//4),
            nn.ReLU(inplace=True),
        )
        self.filter = nn.Sequential(
            nn.Conv2d(dim*4, dim*4, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim*4),
            nn.ReLU(inplace=True),
        )
        # self.kv_embed = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        # self.q = nn.Linear(dim, dim)
        # self.kv = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, dim * 2)
        # )
        # self.proj = nn.Linear(dim+dim//4, dim)
        # self.apply(self._init_weights)
        self.cat = nn.Sequential(
            nn.Conv2d(2*dim, dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.cat2 = nn.Sequential(
            nn.Conv2d(8*dim, 4*dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),
        )
        self.cat3 = nn.Sequential(
            nn.Conv2d(2*dim, dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.dim = dim
        self.query_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        # self.query_conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=4*dim, out_channels=dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=4*dim, out_channels=dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1) 
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x1 ,x2):
        x = self.cat(torch.cat([x1,x2],dim=1))
        m_batchsize,C,width,height= x.shape
        # B, N, C = x.shape
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        width2 = width//2
        height2 = height//2
        # x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        # print(x.shape)
        x1_dwt = self.dwt(x1)
        x1_dwt = self.filter(x1_dwt)
        x2_dwt = self.dwt(x2)
        x2_dwt = self.filter(x2_dwt)
        # print("x1_dwt:")
        # print(x1_dwt.shape)
        x_dwt = self.cat2(torch.cat([x1_dwt,x2_dwt],dim=1))
        # print("x_dwt:")
        # print(x_dwt.shape)
        x_idwt = self.idwt(x_dwt)
        # print("x_idwt:")
        # print(x_idwt.shape)
        # x_idwt = x_idwt.view(B, -1, x_idwt.size(-2)*x_idwt.size(-1)).transpose(1, 2)
        # print("x_idwt2:")
        # print(x_idwt.shape)
        proj_key = self.key_conv(x_dwt).view(m_batchsize, -1, width2 * height2)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = self.softmax(energy)
        # print("energy:")
        # print(energy.shape)
        proj_value = self.value_conv(x_dwt).view(m_batchsize, -1, width2 * height2)
        out = torch.bmm(proj_value, energy.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        # print("out:")
        # print(out.shape)
        outr = x = self.cat3(torch.cat([out,x_idwt],dim=1))

        
        return outr

class WaveAttention3(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.sr_ratio = sr_ratio

        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim//4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim//4),
            nn.ReLU(inplace=True),
        )
        self.filter = nn.Sequential(
            nn.Conv2d(dim*4, dim*4, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim*4),
            nn.ReLU(inplace=True),
        )
        # self.kv_embed = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        # self.q = nn.Linear(dim, dim)
        # self.kv = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, dim * 2)
        # )
        # self.proj = nn.Linear(dim+dim//4, dim)
        # self.apply(self._init_weights)
        self.cat = nn.Sequential(
            nn.Conv2d(2*dim, dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.cat2 = nn.Sequential(
            nn.Conv2d(8*dim, 4*dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),
        )
        self.cat3 = nn.Sequential(
            nn.Conv2d(2*dim, dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.dim = dim
        self.query_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        # self.query_conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1) 
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x1 ,x2):
        x = self.cat(torch.cat([x1,x2],dim=1))
        m_batchsize,C,width,height= x.shape
        # B, N, C = x.shape
        # proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        width2 = width//2
        height2 = height//2
        # x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        # print(x.shape)
        x1_dwt = self.dwt(x1)
        # x1_dwt = self.filter(x1_dwt)
        x2_dwt = self.dwt(x2)
        # x2_dwt = self.filter(x2_dwt)
        # print("x1_dwt:")
        # print(x1_dwt.shape)
        x_dwt = self.cat2(torch.cat([x1_dwt,x2_dwt],dim=1))
        # proj_query = self.query_conv(x_dwt).view(m_batchsize, -1, width2 * height2).permute(0, 2, 1)
        # print("x_dwt:")
        # print(x_dwt.shape)
        x_idwt = self.idwt(x_dwt)
        proj_query = self.query_conv(x_idwt).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # print("x_idwt:")
        # print(x_idwt.shape)
        # x_idwt = x_idwt.view(B, -1, x_idwt.size(-2)*x_idwt.size(-1)).transpose(1, 2)
        # print("x_idwt2:")
        # print(x_idwt.shape)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = self.softmax(energy)
        # print("energy:")
        # print(energy.shape)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, energy.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        # print("out:")
        # print(out.shape)
        outr = x = self.cat3(torch.cat([out,x],dim=1))

        
        return outr