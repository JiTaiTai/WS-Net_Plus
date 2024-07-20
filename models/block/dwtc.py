import torch
import torch.nn as nn
import torch.nn.functional as F

class DWTC(nn.Module):
    def __init__(self, in_channels):
        super(DWTC, self).__init__()

        self.in_channels = in_channels

        self.linear_1 = nn.Linear(self.in_channels, 4)
        self.linear_2 = nn.Linear(self.in_channels // 4, self.in_channels)
        self.extract = nn.Sequential(nn.Conv2d(in_channels, in_channels, (3, 3), padding=(1, 1),
                                               stride=(1, 1), bias=False),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU(inplace=True))
        self.extract2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, (3, 3), padding=(1, 1),
                                               stride=(1, 1), bias=False),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU(inplace=True))
        # self.Softmax = nn.Softmax()

    def forward(self, x):
        
        # x = self.extract(torch.cat([ll, lh, hl, hh], dim=1))
        list1 = torch.split(x,self.in_channels//4,dim=1)
        list1 = list(list1)
        ll = list1[0]
        lh = list1[1]
        hl = list1[2]
        hh = list1[3]
        # out = x
        n_b, n_c, h, w = x.size()
        # res = x 
        x = self.extract(x)
        feats = F.adaptive_avg_pool2d(x, (1, 1)).view((n_b, n_c))
        feats = F.relu(self.linear_1(feats))
        feats = torch.tanh(feats)
        # feats = F.softmax(feats,dim=1)
        # print(feats)
        feats = feats.view((n_b, 4, 1, 1))
        y = torch.split(feats,1,dim=1)
        y = list(y)
        # print(y[0],y[1],y[2],y[3])
        # print(y[0].shape)
        # print(ll.shape)
        # y[0] = y[0].expand_as(ll).clone()
        rll = torch.mul(y[0], ll)
        ll = rll + ll
        # y[1] = y[1].expand_as(lh).clone()
        rlh = torch.mul(y[1], lh)
        lh = rlh + lh
        # y[2] = y[2].expand_as(hl).clone()
        rhl = torch.mul(y[2], hl)
        hl = rhl + hl
        # y[3] = y[3].expand_as(hh).clone()
        rhh = torch.mul(y[3], hh)
        hh = rhh + hh
        x = self.extract2(torch.cat([ll, lh, hl, hh], dim=1))
        out = x


        # feats = feats.view((n_b, n_c, 1, 1))
        # feats = feats.expand_as(input_).clone()
        # outfeats = torch.mul(feats, input_)

        return out
