import torch
import torch.nn.functional as F
from torch import nn
from models.block.Base import Conv3Relu, Conv1Relu
from models.block.torch_wavelets import DWT_2D, IDWT_2D, DWT_2D3, IDWT_2D3
import numpy as np
from models.block.torch_wavelets import DWT_2D, IDWT_2D, DWT_2D3, IDWT_2D3
from models.block.dwtc import DWTC
class BAM(nn.Module):
    """ Basic self-attention module
    """

    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(BAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in //8
        self.activation = activation
        self.ds = ds  #
        self.pool = nn.AvgPool2d(self.ds)
        #print('ds: ',ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, input):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        #x = self.pool(input)
        x = input
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel**-.5) * energy

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        # out = F.interpolate(out, [width*self.ds,height*self.ds])
        # out = out + input
        out = self.gamma * out + input

        return out

class MultiHeadBAM(nn.Module):
    def __init__(self, in_dim, num_heads=8, ds=8, activation=nn.ReLU):
        super(MultiHeadBAM, self).__init__()
        assert in_dim % num_heads == 0, "in_dim should be divisible by num_heads"
        
        self.chanel_in = in_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.ds = ds
        self.activation = activation
        self.pool = nn.AvgPool2d(self.ds)

        # Multi-head query, key, value convolutions
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # Output: in_dim for multi-head
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.final_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # Linear combination of multi-heads

    def forward(self, input):
        x = input
        m_batchsize, C, width, height = x.size()
        
        proj_query = self.query_conv(x).view(m_batchsize, self.num_heads, self.head_dim, width * height).permute(0, 2, 3, 1)
        proj_key = self.key_conv(x).view(m_batchsize, self.num_heads, self.head_dim, width * height).permute(0, 2, 1, 3)
        
        energy = torch.matmul(proj_query, proj_key)
        energy = (self.head_dim**-.5) * energy
        
        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(m_batchsize, self.num_heads, self.head_dim, width * height).permute(0, 2, 1, 3)
        out = torch.matmul(proj_value, attention.permute(0, 2, 1, 3))
        
        out = out.permute(0, 2, 1, 3).contiguous().view(m_batchsize, C, width, height)
        
        out = self.final_conv(out)
        out = self.gamma * out + input
        
        return out


class CrossAttention(nn.Module):
    """ Basic self-attention module
    """

    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(CrossAttention, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in //8
        self.activation = activation
        self.ds = ds  #
        self.pool = nn.AvgPool2d(self.ds)
        #print('ds: ',ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim*4, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1)
        self.query_conv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1)
        self.query_conv4 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1)
        # self.query_conv5 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv6 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv7 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv8 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv5 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv6 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv7 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv8 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        # self.concat_conv = Conv3Relu(in_dim * 2, in_dim)
        # self.concat2_conv = Conv3Relu(in_dim * 2, in_dim)
        # self.diff_conv = Conv3Relu(in_dim , in_dim)
        self.softmax = nn.Softmax(dim=-1)  #
        self.linear_1 = nn.Linear(in_dim, in_dim // 4)
        self.linear_2 = nn.Linear(in_dim // 4, in_dim)
        self.linear_3 = nn.Linear(in_dim, in_dim // 4)
        self.linear_4 = nn.Linear(in_dim // 4, in_dim)


    def forward(self, input1, input2):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        #x = self.pool(input)
        # concat_result = self.concat_conv(torch.cat([input1,input2],1))
        # abs_result = torch.abs(input1-input2)
        # new_tensor = self.generate_tensors(input1, input2)
        n_b, n_c, h, w = input1.size()

        # feats1 = F.adaptive_avg_pool2d(input1, (1, 1)).view((n_b, n_c))
        # feats1 = F.relu(self.linear_1(feats1))
        # feats1 = torch.sigmoid(self.linear_2(feats1))

        # feats2 = F.adaptive_avg_pool2d(input2, (1, 1)).view((n_b, n_c))
        # feats2 = F.relu(self.linear_3(feats2))
        # feats2 = torch.sigmoid(self.linear_4(feats2))

        # feats = (feats1 + feats2) / 2 

        # feats = feats.view((n_b, n_c, 1, 1))
        # feats = feats.expand_as(input1).clone()
        # outfeats = torch.mul(feats, input_)

        
        
        x = input1
        y = input2
        # q1 = self.query_conv1(new_tensor[0])
        # q2 = self.query_conv2(new_tensor[1])
        # q3 = self.query_conv3(new_tensor[2])
        # q4 = self.query_conv4(new_tensor[3])
        # q5 = self.query_conv5(new_tensor[4])
        # q6 = self.query_conv6(new_tensor[5])
        # q7 = self.query_conv7(new_tensor[6])
        # q8 = self.query_conv8(new_tensor[7])
        # q_in = torch.cat([q1,q2,q3,q4,q5,q6,q7,q8],1)
        # q_in2 = torch.cat([q4,q3],1)
        input3 = input1 + (input2 - input1) / 3
        input4 = input3 + (input2 - input1) / 3
        # input5 = input4 + (input2 - input1) / 7
        # input6 = input5 + (input2 - input1) / 7
        # input7 = input6 + (input2 - input1) / 7
        # input8 = input7 + (input2 - input1) / 7
        q1 = self.query_conv1(input1)
        q2 = self.query_conv2(input2)
        q3 = self.query_conv3(input3)
        q4 = self.query_conv4(input4)
        # q5 = self.query_conv5(input5)
        # q6 = self.query_conv6(input6)
        # q7 = self.query_conv7(input7)
        # q8 = self.query_conv8(input8)
        # q_in = self.query_conv(new_tensor)
        q_in1 = torch.cat([q1,q3,q4,q2],1)
        q_in2 = torch.cat([q2,q4,q3,q1],1)
        m_batchsize, C, width, height = x.size()
        proj_query = q_in1.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel**-.5) * energy

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out1 = out.view(m_batchsize, C, width, height)
        # out1 = torch.mul(feats, out1)
        out1 = self.gamma * out1 + input1
        
        m_batchsize, C, width, height = y.size()
        proj_query = q_in2.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv2(y).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel**-.5) * energy

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv2(y).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out2 = out.view(m_batchsize, C, width, height)
        # out2 = torch.mul(feats, out2)
        out2 = self.gamma * out2 + input2
        # diff = self.diff_conv(torch.abs(out1-out2))
        # out = self.concat2_conv(torch.cat([out1,out2],1))
        # out = F.interpolate(out, [width*self.ds,height*self.ds])
        # out = out + abs_result
        # out = self.gamma * out + concat_result

        return out1, out2

    def generate_tensors(self, tensor1, tensor2, steps=7):
        difference = tensor2 - tensor1
        step = difference / steps
        new_tensors = [tensor1 + step * i for i in range(1, steps)]
        all_tensors = [tensor1] + new_tensors + [tensor2]
        # result = torch.cat(all_tensors, dim=1) 
        return all_tensors

class CrossAttention2(nn.Module):
    """ Basic self-attention module
    """

    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(CrossAttention2, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in //8
        self.activation = activation
        self.ds = ds  #
        self.pool = nn.AvgPool2d(self.ds)
        #print('ds: ',ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.concat_conv = Conv3Relu(in_dim * 2, in_dim)
        self.concat2_conv = Conv3Relu(in_dim * 2, in_dim)
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, input1, input2):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        #x = self.pool(input)
        concat_result = self.concat_conv(torch.cat([input1,input2],1))

        x = input1
        y = input2

        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(y).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel**-.5) * energy
        
        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out1 = out.view(m_batchsize, C, width, height)
        out1 = self.gamma * out1 + input1
        
        m_batchsize, C, width, height = y.size()
        proj_query = self.query_conv2(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv2(y).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel**-.5) * energy

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv2(y).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out2 = out.view(m_batchsize, C, width, height)
        out2 = self.gamma * out2 + input2
        out = self.concat2_conv(torch.cat([out1,out2],1))
        # out = F.interpolate(out, [width*self.ds,height*self.ds])
        # out = out + concat_result
        # out = self.gamma * out + concat_result

        return out
    

class CrossAttention_DWT(nn.Module):
    """ Basic self-attention module
    """

    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(CrossAttention_DWT, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in //8
        self.activation = activation
        self.ds = ds  #
        self.pool = nn.AvgPool2d(self.ds)
        #print('ds: ',ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim*4, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim*4, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim*4, out_channels=in_dim*4, kernel_size=1)
        self.query_conv1 = nn.Conv2d(in_channels=in_dim*4, out_channels=in_dim // 2 , kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=in_dim*4, out_channels=in_dim // 2, kernel_size=1)
        # self.query_conv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1)
        # self.query_conv4 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1)
        # self.query_conv5 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv6 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv7 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv8 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv5 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv6 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv7 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv8 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim*4, out_channels=in_dim, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim*4, out_channels=in_dim*4, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        # self.concat_conv = Conv3Relu(in_dim * 2, in_dim)
        # self.concat2_conv = Conv3Relu(in_dim * 2, in_dim)
        # self.diff_conv = Conv3Relu(in_dim , in_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')  #
        # self.linear_1 = nn.Linear(in_dim, in_dim // 4)
        # self.linear_2 = nn.Linear(in_dim // 4, in_dim)
        # self.linear_3 = nn.Linear(in_dim, in_dim // 4)
        # self.linear_4 = nn.Linear(in_dim // 4, in_dim)


    def forward(self, input1, input2):
      
        x = input1
        y = input2

        x_dwt = self.dwt(x)
        y_dwt = self.dwt(y)
        q1 = self.query_conv1(x_dwt)
        q2 = self.query_conv2(y_dwt)
        q_in1 = torch.cat([q1,q2],1)
        q_in2 = torch.cat([q2,q1],1)
        m_batchsize, C, width, height = x_dwt.size()
        proj_query = q_in1.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x_dwt).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel**-.5) * energy

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv(x_dwt).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out1 = out.view(m_batchsize, C, width, height)
        # out1 = torch.mul(feats, out1)
        out1 = self.gamma * out1 + x_dwt
        out1 = self.idwt(out1)
        out1 = self.gamma2 * out1 + input1
        
        m_batchsize, C, width, height = y_dwt.size()
        proj_query = q_in2.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv2(y_dwt).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel**-.5) * energy

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv2(y_dwt).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out2 = out.view(m_batchsize, C, width, height)
        # out2 = torch.mul(feats, out2)
        out2 = self.gamma * out2 + y_dwt
        out2 = self.idwt(out2)
        out2 = self.gamma2 * out2 + input2
        # diff = self.diff_conv(torch.abs(out1-out2))
        # out = self.concat2_conv(torch.cat([out1,out2],1))
        # out = F.interpolate(out, [width*self.ds,height*self.ds])
        # out = out + abs_result
        # out = self.gamma * out + concat_result

        return out1, out2
    

class CrossAttention_DWT_s(nn.Module):
    """ Basic self-attention module
    """

    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(CrossAttention_DWT_s, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in //8
        self.activation = activation
        self.ds = ds  #
        self.pool = nn.AvgPool2d(self.ds)
        #print('ds: ',ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim*4, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim*4, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim*4, out_channels=in_dim*4, kernel_size=1)
        self.query_conv1 = nn.Conv2d(in_channels=in_dim*4, out_channels=in_dim // 2 , kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=in_dim*4, out_channels=in_dim // 2, kernel_size=1)
        self.query_conv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.query_conv4 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # self.query_conv5 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv6 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv7 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv8 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv5 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv6 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv7 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.query_conv8 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim*4, out_channels=in_dim, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim*4, out_channels=in_dim*4, kernel_size=1)

        self.key_conv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        # self.concat_conv = Conv3Relu(in_dim * 2, in_dim)
        # self.concat2_conv = Conv3Relu(in_dim * 2, in_dim)
        # self.diff_conv = Conv3Relu(in_dim , in_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')  #
        # self.linear_1 = nn.Linear(in_dim, in_dim // 4)
        # self.linear_2 = nn.Linear(in_dim // 4, in_dim)
        # self.linear_3 = nn.Linear(in_dim, in_dim // 4)
        # self.linear_4 = nn.Linear(in_dim // 4, in_dim)


    def forward(self, input1, input2):
      
        x = input1
        y = input2

        x_dwt = self.dwt(x)
        y_dwt = self.dwt(y)
        q1 = self.query_conv1(x_dwt)
        q2 = self.query_conv2(y_dwt)
        q_in1 = torch.cat([q1,q2],1)
        q_in2 = torch.cat([q2,q1],1)
        m_batchsize, C, width, height = x_dwt.size()
        proj_query = q_in1.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x_dwt).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel**-.5) * energy
        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)
        proj_value = self.value_conv(x_dwt).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out1 = out.view(m_batchsize, C, width, height)
        # out1 = torch.mul(feats, out1)
        out1 = self.gamma * out1 + x_dwt
        out1 = self.idwt(out1)
        # out1 = self.gamma2 * out1 + input1
        
         
        m_batchsize, C, width, height = y_dwt.size()
        proj_query = q_in2.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv2(y_dwt).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel**-.5) * energy
        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)
        proj_value = self.value_conv2(y_dwt).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out2 = out.view(m_batchsize, C, width, height)
        out2 = self.gamma * out2 + y_dwt
        out2 = self.idwt(out2)

        q1 = self.query_conv3(x)
        q2 = self.query_conv4(y)
        q_in1 = torch.cat([q1,q2],1)
        q_in2 = torch.cat([q2,q1],1)
        m_batchsize, C, width, height = x.size()
        proj_query = q_in1.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv3(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel**-.5) * energy
        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)
        proj_value = self.value_conv3(x).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out3 = out.view(m_batchsize, C, width, height)
        # out1 = torch.mul(feats, out1)
        out1 = self.gamma2 * (out1 + out3) + input1

        m_batchsize, C, width, height = y.size()
        proj_query = q_in2.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv4(y).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel**-.5) * energy
        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)
        proj_value = self.value_conv4(y).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out4 = out.view(m_batchsize, C, width, height)
        # out1 = torch.mul(feats, out1)
        out2 = self.gamma2 * (out2 + out4) + input2


        return out1, out2
    

class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            4, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x, y, x_s, y_s):
        x_compress = self.compress(x)
        y_compress = self.compress(y)
        concat_compress = torch.cat([x_compress,y_compress],1)
        x_out = self.spatial(concat_compress)
        scale = torch.sigmoid_(x_out)
        return x_s * scale, y_s * scale


class TripletAttention(nn.Module):
    def __init__(
        self,
        in_dim,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial

        self.key_channel = in_dim //8
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.query_conv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.query_conv4 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        # self.subabs = Conv3Relu(in_dim, in_dim)
        self.Sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.dwtc1 = DWTC(in_dim*4)
        self.stage1dwt_Conv1 = Conv3Relu(in_dim * 8, in_dim * 4)
        # self.stage1fusion_Conv1 = Conv3Relu(in_dim * 2, in_dim)
        # self.stage1fusion_Conv2 = Conv3Relu(in_dim * 2, in_dim)
        # self.concat_conv = Conv3Relu(in_dim * 2, in_dim)
        # self.concat2_conv = Conv3Relu(in_dim * 2, in_dim)
        # self.diff_conv = Conv3Relu(in_dim , in_dim)
        self.fc = nn.Linear(1, 1, bias=True)  # 添加一个线性变换层
        self.fc.weight.data.fill_(0.2)  # 初始化权重为0.2
        self.fc.bias.data.fill_(0.4)
        self.softmax = nn.Softmax(dim=-1)  #
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x, y):
        # x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        # y_perm1 = y.permute(0, 2, 1, 3).contiguous()
        # x_out1,y_out1 = self.ChannelGateH(x_perm1,y_perm1)
        # x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        # y_out11 = y_out1.permute(0, 2, 1, 3).contiguous()

        # x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        # y_perm2 = y.permute(0, 3, 2, 1).contiguous()
        # x_out2,y_out2 = self.ChannelGateW(x_perm2,y_perm2)
        # x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        # y_out21 = y_out2.permute(0, 3, 2, 1).contiguous()
        fa1 = x
        fb1 = y
        # change1a = self.stage1fusion_Conv2(torch.cat([fa1, fb1], 1))
        fa1dwt = self.dwt(fa1)
        fa1dwt = self.dwtc1(fa1dwt)
        fb1dwt= self.dwt(fb1)
        fb1dwt = self.dwtc1(fb1dwt)
        f1dwt = self.stage1dwt_Conv1(torch.cat([fa1dwt, fb1dwt], 1))
        f1idwt = self.idwt(f1dwt)
        # change1a = self.stage1fusion_Conv1(torch.cat([change1a, f1idwt], 1))
        # mask = self.subabs(torch.abs(x-y))

        mask = self.Tanh(f1idwt)
        mask = self.fc(mask.view(-1, 1)).view_as(mask)
        
        # print(mask)
        # mask_numpy = mask.cpu().detach().numpy()
        # with open('mask.txt', 'ab') as f:
        #     np.savetxt(f, mask_numpy.flatten())
        if not self.no_spatial:
            # x_out,y_out = self.SpatialGate(x,y)
            input1 = x
            input2 = y
            # input3 = input1 + (input2 - input1) / 3
            # input4 = input3 + (input2 - input1) / 3
            q1 = self.query_conv1(input1)
            q2 = self.query_conv2(input2)
            # q3 = self.query_conv3(input3)
            # q4 = self.query_conv4(input4)
            q_in1 = torch.cat([q1,q2],1)
            q_in2 = torch.cat([q2,q1],1)
            m_batchsize, C, width, height = x.size()
            proj_query = q_in1.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
            proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
            energy = torch.bmm(proj_query, proj_key)  # transpose check
            energy = (self.key_channel**-.5) * energy

            attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

            proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out1 = out.view(m_batchsize, C, width, height)
            # x_out = self.gamma * out1 + input1
            x_out = (1-mask) * out1 + mask * input1
            
            m_batchsize, C, width, height = y.size()
            proj_query = q_in2.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
            proj_key = self.key_conv2(y).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
            energy = torch.bmm(proj_query, proj_key)  # transpose check
            energy = (self.key_channel**-.5) * energy

            attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

            proj_value = self.value_conv2(y).view(m_batchsize, -1, width * height)  # B X C X N

            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out2 = out.view(m_batchsize, C, width, height)
            # y_out = self.gamma * out2 + input2
            y_out = (1-mask) * out2 + mask * input2

            x_perm1 = x.permute(0, 2, 1, 3).contiguous()
            y_perm1 = y.permute(0, 2, 1, 3).contiguous()
            x_outs = x_out.permute(0, 2, 1, 3).contiguous()
            y_outs = y_out.permute(0, 2, 1, 3).contiguous()
            x_out1,y_out1 = self.ChannelGateH(x_perm1,y_perm1,x_outs,y_outs)
            x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
            y_out11 = y_out1.permute(0, 2, 1, 3).contiguous()

            x_perm2 = x.permute(0, 3, 2, 1).contiguous()
            y_perm2 = y.permute(0, 3, 2, 1).contiguous()
            x_outl = x_out.permute(0, 3, 2, 1).contiguous()
            y_outl = y_out.permute(0, 3, 2, 1).contiguous()
            x_out2,y_out2 = self.ChannelGateW(x_perm2,y_perm2,x_outl,y_outl)
            x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
            y_out21 = y_out2.permute(0, 3, 2, 1).contiguous()
            x_out = (1 / 2) * (x_out11 + x_out21)
            y_out = (1 / 2) * (y_out11 + y_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out,y_out,f1idwt
