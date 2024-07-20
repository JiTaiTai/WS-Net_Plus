import torch
import torch.nn as nn
import torch.nn.functional as F
from models.block.Base import Conv3Relu, Conv1Relu, Conv5Relu
class MS_CAM(nn.Module):


    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.query_conv = nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1) 
        self.sigmoid = nn.Sigmoid()
        self.Conv1 = Conv3Relu(channels * 2, channels)
    def forward(self, x, residual):
        xa = x + residual
        # xa = self.Conv1(torch.cat([x , residual],1))
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        # m_batchsize, C, width, height = xa.size()
        # proj_query = self.query_conv(xa).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        # proj_key = self.key_conv(xa).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        # proj_value = self.value_conv(xa).view(m_batchsize, -1, width * height).permute(0, 2, 1)   # B X C X N
        # # proj_key_att = self.softmax(proj_key)
        # energy = torch.bmm(proj_key, proj_value)  # transpose check
        # energy = self.softmax(energy) 
        
        # # print(proj_value.shape)
        # out = torch.bmm(proj_query, energy).permute(0, 2, 1)
        # out = out.view(m_batchsize, C, width, height)
        # wei = self.sigmoid(out)

        xo = 2 * residual * wei + 2 * x * (1 - wei)
        # xo = x * wei + residual * (1 - wei)
        return xo

class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

class absAFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels, r=4):
        super(absAFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.query_conv = nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1) 
        self.sigmoid = nn.Sigmoid()
        self.Conv1 = Conv3Relu(channels * 2, channels)
        self.Conv2 = Conv3Relu(channels, channels)
        self.Conv3 = Conv5Relu(channels, channels)
    def forward(self, x, residual):
        xabs = torch.abs(x-residual)
        x1 = self.Conv2(xabs)
        x2 = self.Conv3(xabs)
        xa = x1 + x2
        # xa = self.Conv1(torch.cat([x , residual],1))
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        # m_batchsize, C, width, height = xa.size()
        # proj_query = self.query_conv(xa).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        # proj_key = self.key_conv(xa).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        # proj_value = self.value_conv(xa).view(m_batchsize, -1, width * height).permute(0, 2, 1)   # B X C X N
        # # proj_key_att = self.softmax(proj_key)
        # energy = torch.bmm(proj_key, proj_value)  # transpose check
        # energy = self.softmax(energy) 
        
        # # print(proj_value.shape)
        # out = torch.bmm(proj_query, energy).permute(0, 2, 1)
        # out = out.view(m_batchsize, C, width, height)
        # wei = self.sigmoid(out)

        xo = 2 * x1 * wei + 2 * x2 * (1 - wei)
        # xo = x * wei + residual * (1 - wei)
        return xo

class multiAFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels, r=4):
        super(multiAFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att3 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att4 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.query_conv = nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1) 
        self.sigmoid = nn.Sigmoid()
        self.Conv1 = Conv3Relu(channels * 3, channels)
        self.Conv3 = Conv5Relu(channels * 3, channels)
        self.Conv2 = Conv3Relu(channels * 2, channels)
        self.maxpooling = nn.MaxPool2d(kernel_size=2,
                                        stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x, y ,z):
        # x1 = self.Conv1(torch.cat([x,y,z],1))
        # x2 = self.Conv3(torch.cat([x,y,z],1))
        
        # xa = x1 + x2
        fu = y + z
        # x = self.maxpooling(xa)
        # m_batchsize, C, width, height = x.size()
        # proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        # proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        # # proj_key_att = self.softmax(proj_key)
        # energy = torch.bmm(proj_query, proj_key)  # transpose check
        # energy = self.softmax(energy) 
        # proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        # # print(proj_value.shape)
        # out1_1 = torch.bmm(proj_value, energy.permute(0, 2, 1))

        # mul_query = self.query_conv2(x).view(m_batchsize, -1, width * height)
        # out2 = mul_query * proj_value
        # out = out1_1 + out2
        # out = out.view(m_batchsize, C, width, height)
        # out = self.upsample(out)
        # wei = self.sigmoid(out)
        xl = self.local_att(fu)
        xg = self.global_att(fu)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * y* wei + 2 * z * (1 - wei)

        fu2 = x + xo
        xl = self.local_att2(fu2)
        xg = self.global_att2(fu2)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        res = 2 * x* wei + 2 * xo * (1 - wei)
        # xa = x + z
        # xl = self.local_att3(xa)
        # xg = self.global_att3(xa)
        # xlg = xl + xg
        # wei = self.sigmoid(xlg)
        # x2 = 2 * x * wei + 2 * z * (1 - wei)

        # xyzcat = x1+x2
        # xl = self.local_att2(xyzcat)
        # xg = self.global_att2(xyzcat)
        # xlg = xl + xg
        # wei = self.sigmoid(xlg)
        # x4 = 2 * x1 * wei + 2 * x2 * (1 - wei)

        # xa = y + z
        # xl = self.local_att4(xa)
        # xg = self.global_att4(xa)
        # xlg = xl + xg
        # wei = self.sigmoid(xlg)
        # x3 = 2 * y * wei + 2 * z * (1 - wei)
        # xo = self.Conv2(torch.cat([x4,x3],1))
        # xo = x1 + x2 + x3
        # xo = x * wei + residual * (1 - wei)
        return res
