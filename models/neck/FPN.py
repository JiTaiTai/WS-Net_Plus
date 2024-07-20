import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.block.Base import Conv3Relu, Conv1Relu
from models.block.Drop import DropBlock
from models.block.Field import PPM, ASPP4, SPP, ASPP3
from models.block.PAM import PAM
from models.block.CAM import CAM
from models.block.BAM import BAM, CrossAttention,CrossAttention2,CrossAttention_DWT,CrossAttention_DWT_s,TripletAttention
from models.block.Attention_Module import HFAB,TH,TH2,TH3,TH4
from models.block.MSCAM import AFF,iAFF,absAFF,multiAFF
from models.block.witattention import WaveAttention
from models.block.dctatt import MultiSpectralAttentionLayer
from ..block.torch_wavelets import DWT_2D, IDWT_2D, DWT_2D3, IDWT_2D3
from models.block.dwtc import DWTC
def aug(input1,input2):
    dif1 = input1 - input2
    dif2 = input2 - input1
    att1 = (dif1+2) /2
    att2 = (dif2+2) /2
    input1 = input1 * att1
    input2 = input2 * att2
    return input1,input2

class FPNNeck(nn.Module):
    def __init__(self, inplanes, neck_name='fpn+ppm+fuse'):
        super().__init__()
        
        self.stage1_Conv1 = Conv3Relu(inplanes , inplanes)  # channel: 2*inplanes ---> inplanes
        self.stage2_Conv1 = Conv3Relu(inplanes * 2, inplanes * 2)  # channel: 4*inplanes ---> 2*inplanes
        self.stage3_Conv1 = Conv3Relu(inplanes * 4, inplanes * 4)  # channel: 8*inplanes ---> 4*inplanes
        self.stage4_Conv1 = Conv3Relu(inplanes * 8, inplanes * 8)  # channel: 16*inplanes ---> 8*inplanes

        self.stage1_Conv12 = Conv3Relu(inplanes , inplanes)  # channel: 2*inplanes ---> inplanes
        self.stage2_Conv12 = Conv3Relu(inplanes * 2, inplanes * 2)  # channel: 4*inplanes ---> 2*inplanes
        self.stage3_Conv12 = Conv3Relu(inplanes * 4, inplanes * 4)  # channel: 8*inplanes ---> 4*inplanes
        self.stage4_Conv12 = Conv3Relu(inplanes * 8, inplanes * 8)  # channel: 16*inplanes ---> 8*inplanes

        self.stage_Conv_after_up = Conv3Relu(inplanes , inplanes)
        self.stage2_Conv_after_up = Conv3Relu(inplanes * 2, inplanes)
        self.stage3_Conv_after_up = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage4_Conv_after_up = Conv3Relu(inplanes * 8, inplanes * 4)

        self.stage_Conv2 = Conv1Relu(inplanes , inplanes)
        self.stage1_Conv2 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv2 = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_Conv2 = Conv3Relu(inplanes * 8, inplanes * 4)

        self.stage1_Conv20diff = Conv3Relu(inplanes * 4, inplanes)
        self.stage2_Conv20diff = Conv3Relu(inplanes * 6, inplanes * 2)
        self.stage3_Conv20diff = Conv3Relu(inplanes * 8, inplanes * 4)

        self.stage4_Conv_after_updiff = Conv3Relu(inplanes * 8 , inplanes * 4)

        self.stage3_Conv_after_updiff = Conv3Relu(inplanes * 4 , inplanes * 2)
        self.stage3_Conv_after_up2diff = Conv3Relu(inplanes * 8 , inplanes * 2)

        self.stage2_Conv_after_updiff = Conv3Relu(inplanes * 2 , inplanes)
        self.stage2_Conv_after_up2diff = Conv3Relu(inplanes * 4 , inplanes)
        self.stage2_Conv_after_up3diff = Conv3Relu(inplanes * 8 , inplanes)


        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')

        self.dwtc1 = DWTC(inplanes*4)
        self.dwtc2 = DWTC(inplanes*8)
        self.dwtc3 = DWTC(inplanes*16)
        self.dwtc4 = DWTC(inplanes*32)

        self.stage1dwt_Conv1 = Conv3Relu(inplanes * 8, inplanes * 4)  # channel: 2*inplanes ---> inplanes
        self.stage2dwt_Conv1 = Conv3Relu(inplanes * 16, inplanes * 8)  # channel: 4*inplanes ---> 2*inplanes
        self.stage3dwt_Conv1 = Conv3Relu(inplanes * 32, inplanes * 16)  # channel: 8*inplanes ---> 4*inplanes
        self.stage4dwt_Conv1 = Conv3Relu(inplanes * 64, inplanes * 32)  # channel: 16*inplanes ---> 8*inplanes

        self.stage1fusion_Conv1 = Conv3Relu(inplanes * 2, inplanes)  # channel: 2*inplanes ---> inplanes
        self.stage2fusion_Conv1 = Conv3Relu(inplanes * 4, inplanes * 2)  # channel: 4*inplanes ---> 2*inplanes
        self.stage3fusion_Conv1 = Conv3Relu(inplanes * 8, inplanes * 4)  # channel: 8*inplanes ---> 4*inplanes
        self.stage4fusion_Conv1 = Conv3Relu(inplanes * 16, inplanes * 8)  # channel: 16*inplanes ---> 8*inplanes

        self.aspp4 = ASPP4(inplanes*8)
        
        self.Sigmoid = nn.Sigmoid()
        

        if "fuse" in neck_name:
            self.stage1_Conv3 = Conv3Relu(inplanes * 2, inplanes)
            self.stage2_Conv3 = Conv3Relu(inplanes * 2, inplanes)   
            self.stage3_Conv3 = Conv3Relu(inplanes * 4, inplanes)
            self.stage4_Conv3 = Conv3Relu(inplanes * 8, inplanes)

            self.final_Conv = Conv3Relu(inplanes * 4, inplanes)

            self.fuse = True
        else:
            self.fuse = False

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)


        if "drop" in neck_name:
            rate, size, step = (0.15, 7, 30)
            self.drop = DropBlock(rate=rate, size=size, step=step)
        else:
            self.drop = DropBlock(rate=0, size=0, step=0)

    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4 = ms_feats
        change1_h, change1_w = fa1.size(2), fa1.size(3)
        
        [fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4] = self.drop([fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4])  # dropblock
        
        fa4 = self.aspp4(fa4)
        fb4 = self.aspp4(fb4) 

        change3_2 = self.stage4_Conv_after_up(self.up(fa4))
        fa3 = self.stage3_Conv2(torch.cat([fa3, change3_2], 1))
        change3_2 = self.stage4_Conv_after_up(self.up(fb4))
        fb3 = self.stage3_Conv2(torch.cat([fb3, change3_2], 1))


        change2_2 = self.stage3_Conv_after_up(self.up(fa3))
        fa2 = self.stage2_Conv2(torch.cat([fa2, change2_2], 1))
        change2_2 = self.stage3_Conv_after_up(self.up(fb3))
        fb2 = self.stage2_Conv2(torch.cat([fb2, change2_2], 1))


        change1_2 = self.stage2_Conv_after_up(self.up(fa2))
        fa1 = self.stage1_Conv2(torch.cat([fa1, change1_2], 1))
        change1_2 = self.stage2_Conv_after_up(self.up(fb2))
        fb1 = self.stage1_Conv2(torch.cat([fb1, change1_2], 1))

        diff1 = torch.abs(fa1-fb1)
        compressed_map_upsampled1 = diff1

        diff2 = torch.abs(fa2-fb2)
        compressed_map2 = torch.mean(diff2, dim=1, keepdim=True)
        compressed_map_upsampled2 = F.interpolate(compressed_map2, size=[256,256], mode='bilinear', align_corners=True)
        compressed_map_upsampled2 = compressed_map_upsampled2.squeeze(1)

        diff3 = torch.abs(fa3-fb3)
        compressed_map3 = torch.mean(diff3, dim=1, keepdim=True)
        compressed_map_upsampled3 = F.interpolate(compressed_map3, size=[256,256], mode='bilinear', align_corners=True)
        compressed_map_upsampled3 = compressed_map_upsampled3.squeeze(1)

        diff4 = torch.abs(fa4-fb4)
        compressed_map4 = torch.mean(diff4, dim=1, keepdim=True)
        compressed_map_upsampled4 = F.interpolate(compressed_map4, size=[256,256], mode='bilinear', align_corners=True)
        compressed_map_upsampled4 = compressed_map_upsampled4.squeeze(1)

        change1a = self.stage1_Conv1(torch.abs(fa1-fb1))
        change2a = self.stage2_Conv1(torch.abs(fa2-fb2))
        change3a = self.stage3_Conv1(torch.abs(fa3-fb3))
        change4a = self.stage4_Conv1(torch.abs(fa4-fb4))

        fa1dwt = self.dwt(fa1)
        fa1dwt = self.dwtc1(fa1dwt)
        fb1dwt= self.dwt(fb1)
        fb1dwt = self.dwtc1(fb1dwt)
        f1dwt = self.stage1dwt_Conv1(torch.cat([fa1dwt, fb1dwt], 1))
        f1idwt = self.idwt(f1dwt)

        change1a = self.stage1fusion_Conv1(torch.cat([change1a, f1idwt], 1))


        fa2dwt= self.dwt(fa2)
        fa2dwt = self.dwtc2(fa2dwt)
        fb2dwt= self.dwt(fb2)
        fb2dwt = self.dwtc2(fb2dwt)
        f2dwt = self.stage2dwt_Conv1(torch.cat([fa2dwt, fb2dwt], 1))
        f2idwt = self.idwt(f2dwt)

        change2a = self.stage2fusion_Conv1(torch.cat([change2a, f2idwt], 1))


        fa3dwt = self.dwt(fa3)
        fa3dwt = self.dwtc3(fa3dwt)
        fb3dwt = self.dwt(fb3)
        fb3dwt = self.dwtc3(fb3dwt)
        f3dwt = self.stage3dwt_Conv1(torch.cat([fa3dwt, fb3dwt], 1))
        f3idwt = self.idwt(f3dwt)

        change3a = self.stage3fusion_Conv1(torch.cat([change3a, f3idwt], 1))


        fa4dwt = self.dwt(fa4)
        fa4dwt = self.dwtc4(fa4dwt)
        fb4dwt = self.dwt(fb4)
        fb4dwt = self.dwtc4(fb4dwt)
        f4dwt = self.stage4dwt_Conv1(torch.cat([fa4dwt, fb4dwt], 1))
        f4idwt = self.idwt(f4dwt)

        change4a = self.stage4fusion_Conv1(torch.cat([change4a, f4idwt], 1))


        change4 = self.aspp4(change4a)
        change3_2 = self.stage4_Conv_after_updiff(self.up(change4))
        change3 = self.stage3_Conv20diff(torch.cat([change3a, change3_2], 1))

        change2_2 = self.stage3_Conv_after_updiff(self.up(change3))
        change2_3 = self.stage3_Conv_after_up2diff(self.up2(change4))
        change2 = self.stage2_Conv20diff(torch.cat([change2a, change2_2, change2_3], 1))
 
        change1_2 = self.stage2_Conv_after_updiff(self.up(change2))
        change1_3 = self.stage2_Conv_after_up2diff(self.up2(change3))
        change1_4 = self.stage2_Conv_after_up3diff(self.up3(change4))
        change1 = self.stage1_Conv20diff(torch.cat([change1a, change1_2, change1_3, change1_4], 1))

        

        if self.fuse:
            change4 = self.stage4_Conv3(F.interpolate(change4, size=(change1_h, change1_w),
                                                      mode='bilinear', align_corners=True))
            change3 = self.stage3_Conv3(F.interpolate(change3, size=(change1_h, change1_w),
                                                      mode='bilinear', align_corners=True))
            change2 = self.stage2_Conv3(F.interpolate(change2, size=(change1_h, change1_w),
                                                      mode='bilinear', align_corners=True))

            [change1, change2, change3, change4] = self.drop([change1, change2, change3, change4]) 
            change = self.final_Conv(torch.cat([change1, change2, change3, change4], 1))

        else:
            change = change1
        
        return change,compressed_map_upsampled1,compressed_map_upsampled2,compressed_map_upsampled3,compressed_map_upsampled4

