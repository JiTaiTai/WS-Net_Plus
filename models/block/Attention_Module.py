import torch
import torch.nn as nn
from models.block.High_Frequency_Module import HighFrequencyModule

class HighFrequencyEnhancementStage(nn.Module):
    def __init__(self, input_channel, input_size, ratio=0.5):
        super(HighFrequencyEnhancementStage, self).__init__()
        self.input_channel = input_channel
        self.input_size = input_size
        self.ratio_channel = int(ratio * input_channel)
        self.Global_pooling = nn.AvgPool2d(self.input_size)
        self.FC_1 = nn.Linear(self.input_channel, int(self.input_channel * ratio))
        self.ReLU = nn.PReLU(int(self.input_channel * ratio))
        self.FC_2 = nn.Linear(int(self.input_channel * ratio), self.input_channel)
        self.Sigmoid = nn.Sigmoid()
        self.HighFre = HighFrequencyModule(input_channel=self.input_channel,smooth=True)
        self.Channelfusion = nn.Conv2d(2 * self.input_channel, self.input_channel, kernel_size=1, stride=1)

    # ChannelAttention +HighFrequency
    def forward(self, x):
        residual = x  # residual & x's shape [batch size, channel, input size, input size]
        x_hf = self.HighFre(residual)
        x = self.Global_pooling(x)  # x's shape [batch size, channel, 1, 1]
        x = x.view(-1, self.input_channel)  # x's shape [batch size, channel]
        x = self.FC_1(x)  # x's shape [batch size, ratio channel]
        x = self.ReLU(x)
        x = self.FC_2(x)  # x's shape [batch size, channel]
        x = self.Sigmoid(x)
        x = torch.unsqueeze(x, dim=2)  # x's shape [batch size, channel, 1]
        residual_0 = residual.view(-1, self.input_channel, self.input_size ** 2)
        residual_0 = torch.mul(residual_0, x)
        residual_0 = residual_0.contiguous().view(-1, self.input_channel, self.input_size, self.input_size)
        x_output = residual + residual_0
        x_output = torch.cat((x_output, x_hf), dim=1)
        x_output = self.Channelfusion(x_output)
        return x_output

class TH(nn.Module):
    def __init__(self, input_channel):
        super(TH, self).__init__()
        self.input_channel = input_channel
        self.query_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)

        self.query_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.query_conv2_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.key_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.value_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1) 
        self.Sigmoid = nn.Sigmoid()

    def att(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        # proj_key_att = self.softmax(proj_key)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = self.softmax(energy) 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        # print(proj_value.shape)
        out = torch.bmm(proj_value, energy.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        return x
    def att2(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv_(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv_(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        # proj_key_att = self.softmax(proj_key)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = self.softmax(energy) 
        proj_value = self.value_conv_(x).view(m_batchsize, -1, width * height)  # B X C X N
        # print(proj_value.shape)
        out = torch.bmm(proj_value, energy.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        return x
    # ChannelAttention +HighFrequency

    # def forward(self, x):
    #     m_batchsize, C, width, height = x.size()
    #     first = x
    #     # print(x.shape)
    #     x = torch.split(x, 16, dim = 3)

    #     # print(x[0].shape)
    #     z = [x[0],x[1],x[2],x[3]]
        
    #     for i in range(0,4):
    #         y = torch.split(x[i], 16, dim = 2)
    #         c = [y[0],y[1],y[2],y[3]]
    #         for j in range(0,4):
    #             c[j] = self.att(y[j])
    #         #     # print(y[j].shape) 此y[j]torch.Size([16, 50, 8, 8]) 直接做attention
    #         z[i] = torch.cat((c[0],c[1],c[2],c[3]),dim = 2, out = None)
    #         # print(z[i].shape)
    #     res1 = torch.cat((z[0],z[1],z[2],z[3]),dim = 3, out = None)
    #     # print(res.shape)

    #     x = first
    #     x = torch.split(x, 16, dim = 3)

    #     # print(x[0].shape)
    #     c = [x[0],x[1],x[2],x[3]]
    #     for i in range(0,4):
    #         y = torch.split(x[i], 16, dim = 2)
    #         z = [y[0],y[1],y[2],y[3]]
    #         for j in range(0,4):
    #             # print(y[j].shape)
    #             z[j] = y[j].reshape(m_batchsize,12800,1,1)
    #             # print(z[j].shape)
    #         c[i] = torch.cat((z[0],z[1],z[2],z[3]),dim = 2, out = None)
    #         # print(c[i].shape)
    #     attready = torch.cat((c[0],c[1],c[2],c[3]),dim = 3, out = None)
    #     # print(attready.shape)
    #     attready = self.att2(attready)
    #     att = torch.split(attready, 1, dim = 3)
    #     for i in range(0,4):
    #         ori = torch.split(att[i], 1, dim = 2)
    #         # print(ori)
    #         for j in range(0,4):
    #             z[j]=ori[j].reshape(m_batchsize,50,16,16)
    #             # print(z[j].shape)
    #             # print(z[j])
    #         c[i] = torch.cat((z[0],z[1],z[2],z[3]),dim = 2, out = None)
    #     res2 = torch.cat((c[0],c[1],c[2],c[3]),dim = 3, out = None)
    #     res = res1 + res2
    #     return res

    def forward(self, x):
        residual = x
        # print(x.shape)
        m_batchsize, C, width, height = x.size()
        first = x
        # print(x.shape)
        # print(x.shape)
        x = torch.split(x, 8, dim = 3)

        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        # print(x[3].shape)
        # print(x[4].shape)
        # print(x[5].shape)
        # print(x[6].shape)
        # print(x[7].shape)
        z = [x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]]
        
        for i in range(0,8):
            y = torch.split(x[i], 8, dim = 2)
            c = [y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7]]
            for j in range(0,8):
                c[j] = self.att(y[j])
            #     # print(y[j].shape) 此y[j]torch.Size([16, 50, 8, 8]) 直接做attention
            z[i] = torch.cat((c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]),dim = 2, out = None)
            # print(z[i].shape)
        res1 = torch.cat((z[0],z[1],z[2],z[3],z[4],z[5],z[6],z[7]),dim = 3, out = None)
        # print(res.shape)

        x = first
        x = torch.split(x, 8, dim = 3)

        # print(x[0].shape)
        c = [x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]]
        for i in range(0,8):
            y = torch.split(x[i], 8, dim = 2)
            z = [y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7]]
            for j in range(0,8):
                # print(y[j].shape)
                z[j] = y[j].reshape(m_batchsize,C*64,1,1)
                # print(z[j].shape)
            c[i] = torch.cat((z[0],z[1],z[2],z[3],z[4],z[5],z[6],z[7]),dim = 2, out = None)
            # print(c[i].shape)
        attready = torch.cat((c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]),dim = 3, out = None)
        # print(attready.shape)
        attready = self.att2(attready)
        att = torch.split(attready, 1, dim = 3)
        for i in range(0,8):
            ori = torch.split(att[i], 1, dim = 2)
            # print(ori)
            for j in range(0,8):
                z[j]=ori[j].reshape(m_batchsize,C,8,8)
                # print(z[j].shape)
                # print(z[j])
            c[i] = torch.cat((z[0],z[1],z[2],z[3],z[4],z[5],z[6],z[7]),dim = 2, out = None)
        res2 = torch.cat((c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]),dim = 3, out = None)
        res = res1 + res2
        x = self.Sigmoid(res)
        # print(x.shape)
        # print(residual.shape)
        mask = torch.mul(residual, x)
        
        output = residual + mask
        return output
        


class TH2(nn.Module):
    def __init__(self, input_channel):
        super(TH2, self).__init__()
        self.input_channel = input_channel
        self.query_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)

        self.query_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.query_conv2_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.key_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.value_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1) 
        self.Sigmoid = nn.Sigmoid()

    def att(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        # proj_key_att = self.softmax(proj_key)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = self.softmax(energy) 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        # print(proj_value.shape)
        out = torch.bmm(proj_value, energy.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        return x
    def att2(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv_(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv_(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        # proj_key_att = self.softmax(proj_key)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = self.softmax(energy) 
        proj_value = self.value_conv_(x).view(m_batchsize, -1, width * height)  # B X C X N
        # print(proj_value.shape)
        out = torch.bmm(proj_value, energy.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        return x
    # ChannelAttention +HighFrequency

    def forward(self, x):
        residual = x
        m_batchsize, C, width, height = x.size()
        first = x
        # print(x.shape)
        x = torch.split(x, 8, dim = 3)

        # print(x[0].shape)
        z = [x[0],x[1],x[2],x[3]]
        
        for i in range(0,4):
            y = torch.split(x[i], 8, dim = 2)
            c = [y[0],y[1],y[2],y[3]]
            for j in range(0,4):
                c[j] = self.att(y[j])
            #     # print(y[j].shape) 此y[j]torch.Size([16, 50, 8, 8]) 直接做attention
            z[i] = torch.cat((c[0],c[1],c[2],c[3]),dim = 2, out = None)
            # print(z[i].shape)
        res1 = torch.cat((z[0],z[1],z[2],z[3]),dim = 3, out = None)
        # print(res.shape)

        x = first
        x = torch.split(x, 8, dim = 3)

        # print(x[0].shape)
        c = [x[0],x[1],x[2],x[3]]
        for i in range(0,4):
            y = torch.split(x[i], 8, dim = 2)
            z = [y[0],y[1],y[2],y[3]]
            for j in range(0,4):
                # print(y[j].shape)
                z[j] = y[j].reshape(m_batchsize,6400,1,1)
                # print(z[j].shape)
            c[i] = torch.cat((z[0],z[1],z[2],z[3]),dim = 2, out = None)
            # print(c[i].shape)
        attready = torch.cat((c[0],c[1],c[2],c[3]),dim = 3, out = None)
        # print(attready.shape)
        attready = self.att2(attready)
        att = torch.split(attready, 1, dim = 3)
        for i in range(0,4):
            ori = torch.split(att[i], 1, dim = 2)
            # print(ori)
            for j in range(0,4):
                z[j]=ori[j].reshape(m_batchsize,100,8,8)
                # print(z[j].shape)
                # print(z[j])
            c[i] = torch.cat((z[0],z[1],z[2],z[3]),dim = 2, out = None)
        res2 = torch.cat((c[0],c[1],c[2],c[3]),dim = 3, out = None)
        res = res1 + res2
        x = self.Sigmoid(res)
        mask = torch.mul(residual, x)
        output = residual + mask
        return output

class TH3(nn.Module):
    def __init__(self, input_channel):
        super(TH3, self).__init__()
        self.input_channel = input_channel
        self.query_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)

        self.query_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.query_conv2_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.key_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.value_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1) 
        self.Sigmoid = nn.Sigmoid()

    def att(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        # proj_key_att = self.softmax(proj_key)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = self.softmax(energy) 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        # print(proj_value.shape)
        out = torch.bmm(proj_value, energy.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        return x
    def att2(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv_(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv_(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        # proj_key_att = self.softmax(proj_key)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = self.softmax(energy) 
        proj_value = self.value_conv_(x).view(m_batchsize, -1, width * height)  # B X C X N
        # print(proj_value.shape)
        out = torch.bmm(proj_value, energy.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        return x
    # ChannelAttention +HighFrequency

    # def forward(self, x):
    #     m_batchsize, C, width, height = x.size()
    #     first = x
    #     # print(x.shape)
    #     x = torch.split(x, 16, dim = 3)

    #     # print(x[0].shape)
    #     z = [x[0],x[1],x[2],x[3]]
        
    #     for i in range(0,4):
    #         y = torch.split(x[i], 16, dim = 2)
    #         c = [y[0],y[1],y[2],y[3]]
    #         for j in range(0,4):
    #             c[j] = self.att(y[j])
    #         #     # print(y[j].shape) 此y[j]torch.Size([16, 50, 8, 8]) 直接做attention
    #         z[i] = torch.cat((c[0],c[1],c[2],c[3]),dim = 2, out = None)
    #         # print(z[i].shape)
    #     res1 = torch.cat((z[0],z[1],z[2],z[3]),dim = 3, out = None)
    #     # print(res.shape)

    #     x = first
    #     x = torch.split(x, 16, dim = 3)

    #     # print(x[0].shape)
    #     c = [x[0],x[1],x[2],x[3]]
    #     for i in range(0,4):
    #         y = torch.split(x[i], 16, dim = 2)
    #         z = [y[0],y[1],y[2],y[3]]
    #         for j in range(0,4):
    #             # print(y[j].shape)
    #             z[j] = y[j].reshape(m_batchsize,12800,1,1)
    #             # print(z[j].shape)
    #         c[i] = torch.cat((z[0],z[1],z[2],z[3]),dim = 2, out = None)
    #         # print(c[i].shape)
    #     attready = torch.cat((c[0],c[1],c[2],c[3]),dim = 3, out = None)
    #     # print(attready.shape)
    #     attready = self.att2(attready)
    #     att = torch.split(attready, 1, dim = 3)
    #     for i in range(0,4):
    #         ori = torch.split(att[i], 1, dim = 2)
    #         # print(ori)
    #         for j in range(0,4):
    #             z[j]=ori[j].reshape(m_batchsize,50,16,16)
    #             # print(z[j].shape)
    #             # print(z[j])
    #         c[i] = torch.cat((z[0],z[1],z[2],z[3]),dim = 2, out = None)
    #     res2 = torch.cat((c[0],c[1],c[2],c[3]),dim = 3, out = None)
    #     res = res1 + res2
    #     return res

    def forward(self, x, mask1):
        residual = x
        x = torch.mul(x, mask1)
        m_batchsize, C, width, height = x.size()
        first = x
        # print(x.shape)
        x = torch.split(x, 8, dim = 3)

        # print(x[0].shape)
        z = [x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]]
        
        for i in range(0,8):
            y = torch.split(x[i], 8, dim = 2)
            c = [y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7]]
            for j in range(0,8):
                c[j] = self.att(y[j])
            #     # print(y[j].shape) 此y[j]torch.Size([16, 50, 8, 8]) 直接做attention
            z[i] = torch.cat((c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]),dim = 2, out = None)
            # print(z[i].shape)
        res1 = torch.cat((z[0],z[1],z[2],z[3],z[4],z[5],z[6],z[7]),dim = 3, out = None)
        # print(res.shape)

        x = first
        x = torch.split(x, 8, dim = 3)

        # print(x[0].shape)
        c = [x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]]
        for i in range(0,8):
            y = torch.split(x[i], 8, dim = 2)
            z = [y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7]]
            for j in range(0,8):
                # print(y[j].shape)
                z[j] = y[j].reshape(m_batchsize,C*64,1,1)
                # print(z[j].shape)
            c[i] = torch.cat((z[0],z[1],z[2],z[3],z[4],z[5],z[6],z[7]),dim = 2, out = None)
            # print(c[i].shape)
        attready = torch.cat((c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]),dim = 3, out = None)
        # print(attready.shape)
        attready = self.att2(attready)
        att = torch.split(attready, 1, dim = 3)
        for i in range(0,8):
            ori = torch.split(att[i], 1, dim = 2)
            # print(ori)
            for j in range(0,8):
                z[j]=ori[j].reshape(m_batchsize,C,8,8)
                # print(z[j].shape)
                # print(z[j])
            c[i] = torch.cat((z[0],z[1],z[2],z[3],z[4],z[5],z[6],z[7]),dim = 2, out = None)
        res2 = torch.cat((c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]),dim = 3, out = None)
        res = res1 + res2
        x = self.Sigmoid(res)
        mask = torch.mul(residual, x)
        output = residual + mask
        return output
        


class TH4(nn.Module):
    def __init__(self, input_channel):
        super(TH4, self).__init__()
        self.input_channel = input_channel
        self.query_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)

        self.query_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.query_conv2_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.key_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.value_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1) 
        self.Sigmoid = nn.Sigmoid()

    def att(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        # proj_key_att = self.softmax(proj_key)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = self.softmax(energy) 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        # print(proj_value.shape)
        out = torch.bmm(proj_value, energy.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        return x
    def att2(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv_(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv_(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        # proj_key_att = self.softmax(proj_key)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = self.softmax(energy) 
        proj_value = self.value_conv_(x).view(m_batchsize, -1, width * height)  # B X C X N
        # print(proj_value.shape)
        out = torch.bmm(proj_value, energy.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        return x
    # ChannelAttention +HighFrequency

    def forward(self, x, mask1):
        residual = x
        x = torch.mul(x, mask1)
        m_batchsize, C, width, height = x.size()
        first = x
        # print(x.shape)
        x = torch.split(x, 8, dim = 3)

        # print(x[0].shape)
        z = [x[0],x[1],x[2],x[3]]
        
        for i in range(0,4):
            y = torch.split(x[i], 8, dim = 2)
            c = [y[0],y[1],y[2],y[3]]
            for j in range(0,4):
                c[j] = self.att(y[j])
            #     # print(y[j].shape) 此y[j]torch.Size([16, 50, 8, 8]) 直接做attention
            z[i] = torch.cat((c[0],c[1],c[2],c[3]),dim = 2, out = None)
            # print(z[i].shape)
        res1 = torch.cat((z[0],z[1],z[2],z[3]),dim = 3, out = None)
        # print(res.shape)

        x = first
        x = torch.split(x, 8, dim = 3)

        # print(x[0].shape)
        c = [x[0],x[1],x[2],x[3]]
        for i in range(0,4):
            y = torch.split(x[i], 8, dim = 2)
            z = [y[0],y[1],y[2],y[3]]
            for j in range(0,4):
                # print(y[j].shape)
                z[j] = y[j].reshape(m_batchsize,6400,1,1)
                # print(z[j].shape)
            c[i] = torch.cat((z[0],z[1],z[2],z[3]),dim = 2, out = None)
            # print(c[i].shape)
        attready = torch.cat((c[0],c[1],c[2],c[3]),dim = 3, out = None)
        # print(attready.shape)
        attready = self.att2(attready)
        att = torch.split(attready, 1, dim = 3)
        for i in range(0,4):
            ori = torch.split(att[i], 1, dim = 2)
            # print(ori)
            for j in range(0,4):
                z[j]=ori[j].reshape(m_batchsize,100,8,8)
                # print(z[j].shape)
                # print(z[j])
            c[i] = torch.cat((z[0],z[1],z[2],z[3]),dim = 2, out = None)
        res2 = torch.cat((c[0],c[1],c[2],c[3]),dim = 3, out = None)
        res = res1 + res2
        x = self.Sigmoid(res)
        mask = torch.mul(residual, x)
        output = residual + mask
        return output
        


class SpatialAttentionStage(nn.Module):
    def __init__(self, input_channel):
        super(SpatialAttentionStage, self).__init__()
        self.bn_momentum = 0.1
        self.input_channel = input_channel
        # down 1
        self.conv1_1 = nn.Conv2d(self.input_channel, self.input_channel // 2,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn1_1 = nn.BatchNorm2d(self.input_channel // 2,
                                    momentum=self.bn_momentum)
        self.ReLU1_1 = nn.PReLU(self.input_channel // 2)
        self.conv1_2 = nn.Conv2d(self.input_channel // 2, self.input_channel // 2,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn1_2 = nn.BatchNorm2d(self.input_channel // 2,
                                    momentum=self.bn_momentum)
        self.ReLU1_2 = nn.PReLU(self.input_channel // 2)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2,
                                        stride=2)
        # down 2
        self.conv2_1 = nn.Conv2d(self.input_channel // 2, self.input_channel // 4,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn2_1 = nn.BatchNorm2d(self.input_channel // 4,
                                    momentum=self.bn_momentum)
        self.ReLU2_1 = nn.PReLU(self.input_channel // 4)
        self.conv2_2 = nn.Conv2d(self.input_channel // 4, self.input_channel // 4,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn2_2 = nn.BatchNorm2d(self.input_channel // 4,
                                    momentum=self.bn_momentum)
        self.ReLU2_2 = nn.PReLU(self.input_channel // 4)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=2,
                                        stride=2)
        # bottom
        self.conv_b_1 = nn.Conv2d(self.input_channel // 4, self.input_channel // 8,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.bn_b_1 = nn.BatchNorm2d(self.input_channel // 8,
                                     momentum=self.bn_momentum)
        self.ReLU_b_1 = nn.PReLU(self.input_channel // 8)
        self.conv_b_2 = nn.Conv2d(self.input_channel // 8, self.input_channel // 8,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.bn_b_2 = nn.BatchNorm2d(self.input_channel // 8,
                                     momentum=self.bn_momentum)
        self.ReLU_b_2 = nn.PReLU(self.input_channel // 8)

        self.query_conv = nn.Conv2d(in_channels=input_channel//4, out_channels=input_channel // 16, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=input_channel//4, out_channels=input_channel//4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_channel//4, out_channels=input_channel // 16, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_channel//4, out_channels=input_channel//4, kernel_size=1)
        self.query_conv_ = nn.Conv2d(in_channels=input_channel//4, out_channels=input_channel // 16, kernel_size=1)
        self.query_conv2_ = nn.Conv2d(in_channels=input_channel//4, out_channels=input_channel//4, kernel_size=1)
        self.key_conv_ = nn.Conv2d(in_channels=input_channel//4, out_channels=input_channel // 16, kernel_size=1)
        self.value_conv_ = nn.Conv2d(in_channels=input_channel//4, out_channels=input_channel//4, kernel_size=1)

        # up 1
        self.convtrans_1 = nn.ConvTranspose2d(self.input_channel // 8, self.input_channel // 16,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              output_padding=1)
        self.convtrans_1_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3_1 = nn.Conv2d(self.input_channel // 8 + self.input_channel // 4, self.input_channel // 16,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.conv3_1_2 = nn.Conv2d(self.input_channel // 8, self.input_channel // 16,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)                         
        self.bn3_1 = nn.BatchNorm2d(self.input_channel // 16,
                                    momentum=self.bn_momentum)
        self.ReLU3_1 = nn.PReLU(self.input_channel // 16)
        self.conv3_2 = nn.Conv2d(self.input_channel // 16, self.input_channel // 16,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn3_2 = nn.BatchNorm2d(self.input_channel // 16,
                                    momentum=self.bn_momentum)
        self.ReLU3_2 = nn.PReLU(self.input_channel // 16)
        # up 2
        self.convtrans_2 = nn.ConvTranspose2d(self.input_channel // 16, self.input_channel // 32,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              output_padding=1)
        self.convtrans_2_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4_1 = nn.Conv2d(self.input_channel // 16 + self.input_channel // 2, self.input_channel // 32,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.conv4_1_2 = nn.Conv2d(self.input_channel // 16, self.input_channel // 32,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn4_1 = nn.BatchNorm2d(self.input_channel // 32,
                                    momentum=self.bn_momentum)
        self.ReLU4_1 = nn.PReLU(self.input_channel // 32)
        self.conv4_2 = nn.Conv2d(self.input_channel // 32, self.input_channel // 32,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn4_2 = nn.BatchNorm2d(self.input_channel // 32,
                                    momentum=self.bn_momentum)
        self.ReLU4_2 = nn.PReLU(self.input_channel // 32)
        # out
        self.conv5_1 = nn.Conv2d(self.input_channel // 32, self.input_channel // 32,
                                 kernel_size=1,
                                 stride=1)
        self.bn5_1 = nn.BatchNorm2d(self.input_channel // 32,
                                    momentum=self.bn_momentum)
        self.ReLU5_1 = nn.PReLU(self.input_channel // 32)
        self.conv5_2 = nn.Conv2d(self.input_channel // 32, 1,
                                 kernel_size=1,
                                 stride=1)
        self.bn5_2 = nn.BatchNorm2d(1,
                                    momentum=self.bn_momentum)
        # self.ReLU5_2 = nn.PReLU(1)
        self.Sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1) 

    def forward(self, x):
        residual = x
        # down 1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.ReLU1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.ReLU1_2(x)
        # skip connection
        skip_1 = x
        x = self.maxpooling1(x)
        # down 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.ReLU2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.ReLU2_2(x)
        # skip connection
        skip_2 = x
        x = self.maxpooling2(x)
        # bottom
        # x = self.conv_b_1(x)
        # x = self.bn_b_1(x)
        # x = self.ReLU_b_1(x)
        # x = self.conv_b_2(x)
        # x = self.bn_b_2(x)
        # x = self.ReLU_b_2(x)
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        # proj_key_att = self.softmax(proj_key)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = self.softmax(energy) 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        # print(proj_value.shape)
        out1_1 = torch.bmm(proj_value, energy.permute(0, 2, 1))

        mul_query = self.query_conv2(x).view(m_batchsize, -1, width * height)
        out2 = mul_query * proj_value
        out = out1_1 + out2
        out = out.view(m_batchsize, C, width, height)
        x = out
        x = self.conv_b_1(x)
        x = self.bn_b_1(x)
        x = self.ReLU_b_1(x)
        # up 1
        x = self.convtrans_1_2(x)
        # cat skip connection
        x = torch.cat((x, skip_2), dim=1)
        x = self.conv3_1(x)
        #x = self.conv3_1_2(x)
        x = self.bn3_1(x)
        x = self.ReLU3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.ReLU3_2(x)
        # up 2
        # x = self.convtrans_2(x)
        x = self.convtrans_2_2(x)
        # cat skip connection
        x = torch.cat((x, skip_1), dim=1)
        x = self.conv4_1(x)
        #x = self.conv4_1_2(x)
        x = self.bn4_1(x)
        x = self.ReLU4_1(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.ReLU4_2(x)
        # out
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.ReLU5_1(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.Sigmoid(x)
        mask = torch.mul(residual, x)
        output = residual + mask
        return output

class SA(nn.Module):
    def __init__(self, input_channel,activation=nn.ReLU):
        super(SA, self).__init__()
        self.chanel_in = input_channel
        self.key_channel = self.chanel_in // 16
        self.activation = activation
        self.ds = 8  #
        self.pool = nn.AvgPool2d(self.ds)
        self.bn_momentum = 0.1
        self.conv1_1 = nn.Conv2d(input_channel//16, 1,
                                 kernel_size=1,
                                 stride=1,
                                )
        self.conv1_2 = nn.Conv2d(input_channel, 1,
                                 kernel_size=1,
                                 stride=1,
                                )
        self.convc = nn.Conv2d(input_channel, input_channel,
                                 kernel_size=1,
                                 stride=1,
                                )
        self.bn1_1 = nn.BatchNorm2d(1,
                                    momentum=self.bn_momentum)
        # self.ReLU1_1 = nn.PReLU(self.input_channel // 2)
        #print('ds: ',ds)
        self.query_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel // 16, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel // 16, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.query_conv_ = nn.Conv2d(in_channels=input_channel, out_channels=input_channel // 16, kernel_size=1)
        self.query_conv2_ = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.key_conv_ = nn.Conv2d(in_channels=input_channel, out_channels=input_channel // 16, kernel_size=1)
        self.value_conv_ = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.att_conv = nn.Conv2d(in_channels = input_channel//16 + input_channel//16, out_channels=input_channel // 16,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.con_conv = nn.Conv2d(in_channels = input_channel+input_channel, out_channels=input_channel,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.att_bn = nn.BatchNorm2d(input_channel // 16,momentum=self.bn_momentum)
        self.att_ReLu = nn.PReLU(input_channel//16)
        self.softmax = nn.Softmax(dim=-1) 
        self.Sigmoid = nn.Sigmoid()


    def forward(self, x):
        residual = x
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        # print(proj_query.shape)
        # print(proj_key.shape)
        proj_key_att = self.softmax(proj_key)
        # print(proj_key_att.shape)
        energy = torch.bmm(proj_query, proj_key_att)  # transpose check
        # print(energy.shape)
        #attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        # print(proj_value.shape)
        out1_1 = torch.bmm(proj_value, energy.permute(0, 2, 1))
        # out1_1 = out1_1.view(m_batchsize, C, width, height)

        # proj_query_2 = self.query_conv_(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        # proj_key_2 = self.key_conv_(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        # proj_key_att_2 = self.softmax(proj_key_2)
        # energy_2 = torch.bmm(proj_query_2, proj_key_att_2)  # transpose check

        # proj_value_2 = self.value_conv_(x).view(m_batchsize, -1, width * height)  # B X C X N
        # # print(proj_value.shape)
        # out1_2 = torch.bmm(proj_value_2, energy_2.permute(0, 2, 1))
        # out1_2 = out1_2.view(m_batchsize, C, width, height)
        # out1 = self.con_conv(torch.cat([out1_1,out1_2],1))

        mul_query = self.query_conv2(x).view(m_batchsize, -1, width * height)
        out2 = mul_query * proj_value
        # out2 = out2.view(m_batchsize, C, width, height)
        # out = self.con_conv(torch.cat([out1_1,out2],1))
        out = out1_1 + out2
        out = out.view(m_batchsize, C, width, height)

        # out = F.interpolate(out, [width*self.ds,height*self.ds])
        # out = out + input
        # out = self.gamma * out + x
        
        # out = self.convc(out)
        # proj_query = self.query_conv(x)
        # proj_key = self.key_conv(x)
        # proj_value = self.value_conv(x)
        # attention = self.att_conv(torch.cat([proj_key, proj_query], 1))
        # attention = self.att_bn(attention)
        # attention = self.att_ReLu(attention)
        # out = attention * proj_value
        # out = self.conv1_1(out)
        # out = self.conv1_2(out)
        # out = self.bn1_1(out)
        out = self.Sigmoid(out)
        # mask = torch.mul(residual, out)
        mask = residual * out
        output = residual + mask
        return output

class HFAB(nn.Module):
    def __init__(self, input_channel, input_size, ratio=0.5):
        super(HFAB, self).__init__()
        self.SA = SpatialAttentionStage(input_channel=input_channel)
        self.SA2 = SA(input_channel=input_channel)
        self.HF = HighFrequencyEnhancementStage(input_channel=input_channel,
                                                               input_size=input_size,
                                                               ratio=ratio)
    def forward(self, x):
        x = self.SA(x)
        # x = self.SA2(x)
        #x = self.HF(x)

        return x

