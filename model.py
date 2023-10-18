import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from module import *
from function import *


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

        self.masic_att = LKA(self.Ch)

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
                                      ResBlock(default_conv, self.G0, self.KS, bn=self.bn, act=self.act), ResBlock(default_conv, self.G0, self.KS, bn=self.bn, act=self.act),
                                      ResBlock(default_conv, self.G0, self.KS, bn=self.bn, act=self.act), ResBlock(default_conv, self.G0, self.KS, bn=self.bn, act=self.act),
                                      ResBlock(default_conv, self.G0, self.KS, bn=self.bn, act=self.act), ResBlock(default_conv, self.G0, self.KS, bn=self.bn, act=self.act),
                                      ResBlock(default_conv, self.G0, self.KS, bn=self.bn, act=self.act), ResBlock(default_conv, self.G0, self.KS, bn=self.bn, act=self.act),
                                      ResBlock(default_conv, self.G0, self.KS, bn=self.bn, act=self.act), ResBlock(default_conv, self.G0, self.KS, bn=self.bn, act=self.act),
                                      BasicBlock(self.G0, self.G0, self.KS, bn=self.bn, act=self.act))

        
        self.W_generator = nn.Sequential(BasicBlock(self.G0, self.G0, self.KS, bn=self.bn, act=self.act),
                                         BasicBlock(self.G0, self.Ch, self.KS, bn=self.bn, act=self.act),
                                         default_conv(self.Ch, self.Ch, kernel_size=1))

        self.U_generator = nn.Sequential(BasicBlock(self.G0, self.G0, self.KS, bn=self.bn, act=self.act),
                                         BasicBlock(self.G0, self.Ch, self.KS, bn=self.bn, act=self.act),
                                         default_conv(self.Ch, self.Ch, kernel_size=1))

        
        self.NLBlock = nn.ModuleList([blockNL(self.Ch, self.NBS) for _ in range(self.s)])

        self.delta = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(self.s)])
        self.delta1 = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(self.s)])
        self.eta = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(self.s)])
        self.gama = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(self.s)])


    def forward(self, y, mask):
        # print(y.shape, mask.shape)  # b, 1, h, w  b, c, h, w
        X_gt = y * mask
        maskng = get_maskng(mask.clone())
        Xt = interpp(y, mask)
        Xt = self.masic_att(Xt)
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
            Res2 = self.AT(y2x(x2y(self.A(Xt + Et), mask) - y, mask))


            fea = self.Fe_e[i](Xt)
            fea_list.append(fea)
            if i != 0:
                fea = self.Fe_f[i-1](torch.cat(fea_list, 1))
            decode0 = self.denoiser(fea)
                
            fea_list.append(decode0)

            W = self.W_generator(decode0)
            U = self.U_generator(decode0)

            # Xt = Xt - 2 * self.delta[i] * (Res1 + self.eta[i] * Res2 + self.gama[i] * (Xt + Et - NL) + (Xt - U).mul(W))
            Xt = Xt - 2 * self.delta[i] * (self.eta[i] * Res2 + self.gama[i] * (Xt + Et - NL) + (Xt - U).mul(W))

            Xt = X_gt + Xt.mul(maskng)


        return Xt


