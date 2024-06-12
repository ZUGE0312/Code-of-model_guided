import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from module import *
from function import *
import numpy as np
from SSB_net import SSB_net
def get_WB_filter(size):
    """make a 2D weight bilinear kernel suitable for WB_Conv"""
    ligne = []
    colonne = []
    for i in range(size):
        if (i + 1) <= np.floor(math.sqrt(16)):
            ligne.append(i + 1)
            colonne.append(i + 1)
        else:
            ligne.append(ligne[i - 1] - 1.0)
            colonne.append(colonne[i - 1] - 1.0)
    BilinearFilter = np.zeros(size * size)
    for i in range(size):
        for j in range(size):
            BilinearFilter[(j + i * size)] = (ligne[i] * colonne[j] / 16)
    filter0 = np.reshape(BilinearFilter, (size, size))
    return torch.from_numpy(filter0).float()

class MoG_DUN(nn.Module):
    def __init__(self, args):
        super(MoG_DUN, self).__init__()
        self.Ch = args.num_channel
        self.s = args.T
        self.device = args.device

        self.act = nn.LeakyReLU()

        self.bn = False

        self.G0 = 64
        self.KS = 3
        self.NBS = 5
        num_blocks = 8
        #self.masic_att = LKA(self.Ch)

        ## The modules for learning the measurement matrix A and A^T
        self.AT = nn.Sequential(BasicBlock(self.Ch, self.G0, self.KS, bn=self.bn, act=self.act),
                                ResBlock(default_conv, self.G0, self.KS, bn=self.bn, act=self.act), ResBlock(default_conv, self.G0, self.KS, bn=self.bn, act=self.act),
                                BasicBlock(self.G0, self.Ch, self.KS, bn=self.bn, act=self.act))
        self.A = nn.Sequential(BasicBlock(self.Ch, self.G0, self.KS, bn=self.bn, act=self.act),
                                ResBlock(default_conv, self.G0, self.KS, bn=self.bn, act=self.act), ResBlock(default_conv, self.G0, self.KS, bn=self.bn, act=self.act),
                                BasicBlock(self.G0, self.Ch, self.KS, bn=self.bn, act=self.act))


        self.Fe_e = nn.ModuleList([nn.Sequential(*[default_conv(self.Ch, self.G0, self.KS), 
                                                   default_conv(self.G0, self.G0, self.KS)])
                                   for _ in range(self.s)])

        self.Fe_f = nn.ModuleList([nn.Sequential(*[nn.Conv2d((2 * i + 3) * self.G0, self.G0, 1)]) for i in range(self.s - 1)])

        self.denoiser = nn.Sequential(BasicBlock(self.G0, self.G0, self.KS, bn=self.bn, act=self.act),
                                      SSB_net(dim=self.G0, num_blocks=num_blocks, dim_head=self.G0 , heads=1,channel=self.G0),
                                      BasicBlock(self.G0, self.G0, self.KS, bn=self.bn, act=self.act))

        self.W_generator = nn.Sequential(nn.Conv2d(self.G0, self.Ch, 3,1,1),
                                         )

        self.U_generator = nn.Sequential(nn.Conv2d(self.G0, self.Ch, 3,1,1),
                                         )
        
        # self.W_generator = nn.Sequential(BasicBlock(self.G0, self.G0, self.KS, bn=self.bn, act=self.act),
        #                                  BasicBlock(self.G0, self.Ch, self.KS, bn=self.bn, act=self.act),
        #                                  default_conv(self.Ch, self.Ch, kernel_size=1))
        #
        # self.U_generator = nn.Sequential(BasicBlock(self.G0, self.G0, self.KS, bn=self.bn, act=self.act),
        #                                  BasicBlock(self.G0, self.Ch, self.KS, bn=self.bn, act=self.act),
        #                                  default_conv(self.Ch, self.Ch, kernel_size=1))

        
        self.NLBlock = nn.ModuleList([blockNL(self.Ch, self.NBS) for _ in range(self.s)])

        self.delta = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(self.s)])
        self.delta1 = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(self.s)])
        self.eta = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(self.s)])
        self.gama = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(self.s)])
        self.eta1 = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(self.s)])
        self.gama1 = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(self.s)])
        self.WB_Conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3, bias=False,
                                 groups=16)
        cout, cin, h, w = self.WB_Conv.weight.data.size()
        self.WB_Conv.weight.data = get_WB_filter(7).view(1, 1, h, w).repeat(cout, cin, 1, 1)
    def forward(self, y, mask):
        # print(y.shape, mask.shape)  # b, 1, h, w  b, c, h, w
        #X_gt = y * mask
        #maskng = get_maskng(mask.clone())
        #Xt = interpp(y, mask)
        #Xt = self.masic_att(Xt)
        y1 = y.repeat(1, mask.size(1), 1, 1)
        Xt = self.WB_Conv(y1 * mask)
        # return Xt

        fea_list = []

        for i in range(0, self.s):
            # AXt = x2y(self.A(Xt), mask)  # y = Ax

            NL = self.NLBlock[i](Xt)
            if i == 0:
                Et = NL - Xt

            Res1 = self.AT(y2x(x2y(self.A(Xt + Et), mask) - y, mask))
            Et = Et - 2 * self.delta1[i] * (self.eta[i] * Res1 + self.gama[i] * (Xt + Et - NL))

            # Res1 = self.AT(y2x(x2y(self.A(Xt), mask) - y, mask))   # A^T * (Ax − y)
            #Res2 = self.AT(y2x(x2y(self.A(Xt + Et), mask) - y, mask))


            fea = self.Fe_e[i](Xt)
            fea_list.append(fea)
            if i != 0:
                fea = self.Fe_f[i-1](torch.cat(fea_list, 1))
            decode0 = self.denoiser(fea)
                
            fea_list.append(decode0)

            W = torch.exp(self.W_generator(decode0))
            U = self.U_generator(decode0)
            #print(W.shape,U.shape)
            #Xt = Xt - 2 * self.delta[i] * (self.AT(y2x(x2y(self.A(Xt), mask) - y, mask)) + self.eta1[i] * y2x(x2y(self.A(Xt + Et), mask) - y, mask) + self.gama1[i] * (Xt + Et - NL) + (Xt - U).mul(W))
            Xt = Xt - 2 * self.delta[i] * (self.AT(y2x(x2y(self.A(Xt), mask) - y, mask)+self.eta1[i] * y2x(x2y(self.A(Xt + Et), mask) - y, mask)) + self.gama1[i] * (Xt + Et - NL) + (Xt - U).mul(W))
            #Xt = Xt - 2 * self.delta[i] * (self.eta[i] * Res2 + self.gama[i] * (Xt + Et - NL) + (Xt - U).mul(W))

            #Xt = X_gt + Xt.mul(maskng)


        return Xt


        return Xt


