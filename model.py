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

class SSMP_HIP(nn.Module):
    def __init__(self, args):
        super(SSMP_HIP, self).__init__()
        self.Ch = args.num_channel
        self.s = args.T
        self.device = args.device

        self.act = nn.LeakyReLU()

        self.bn = False

        self.G0 = 64
        self.KS = 3
        self.NBS = 5
        num_blocks = 8

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
        y1 = y.repeat(1, mask.size(1), 1, 1)
        Xt = self.WB_Conv(y1 * mask)

        fea_list = []

        for i in range(0, self.s):
            NL = self.NLBlock[i](Xt)
            if i == 0:
                Et = NL - Xt

            Res1 = self.AT(y2x(x2y(self.A(Xt + Et), mask) - y, mask))
            Et = Et - 2 * self.delta1[i] * (self.eta[i] * Res1 + self.gama[i] * (Xt + Et - NL))

            fea = self.Fe_e[i](Xt)
            fea_list.append(fea)
            if i != 0:
                fea = self.Fe_f[i-1](torch.cat(fea_list, 1))
            decode0 = self.denoiser(fea)
                
            fea_list.append(decode0)

            W = torch.exp(self.W_generator(decode0))
            U = self.U_generator(decode0)
            Xt = Xt - 2 * self.delta[i] * (self.AT(y2x(x2y(self.A(Xt), mask) - y, mask)+self.eta1[i] * y2x(x2y(self.A(Xt + Et), mask) - y, mask)) + self.gama1[i] * (Xt + Et - NL) + (Xt - U).mul(W))
        return Xt

if __name__ == '__main__':
    #from thop import profile
    import os
    import time
    from collections import OrderedDict
    import scipy.io as sio
    import thop
    import argparse

    base_dir = os.path.split(os.path.realpath(__file__))[0]
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--num_worker', type=int, default=1)
    parser.add_argument("--num_trainset", type=int, default=104)  # 900 / 104
    parser.add_argument("--num_channel", type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument("--patch_size", type=tuple, default=(128, 128))
    parser.add_argument("--test_size", type=tuple, default=(512, 512))  # mask_size
    parser.add_argument('--max_epoch', type=int, default=8000)
    parser.add_argument('--point_epoch', '-p', type=int, default=1)
    parser.add_argument('--mode', '-m', type=str, default='train', choices=['train', 'valid', 'test'])
    parser.add_argument('--test_real', type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

    # Model parameters
    parser.add_argument('--model_name', type=str, default='mog-dun')  # , choices=['mog-dun', 'MIMO-UNetPlus'])
    parser.add_argument("--T", type=int, default=6)

    parser.add_argument('--dataset', type=str, default='CAVE')

    args = parser.parse_args()

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    height = 256
    width = 256
    input_size = (1,1,256,256)
    mask_size = (1, 16, 256, 256)
    mask_path = "G:/Dataset/SCI_data/mask_simu.mat"
    mask = sio.loadmat(mask_path)['mask']  # [256, 256]
    mask = torch.from_numpy(mask)

    model = SSMP_HIP(args)#.cuda()
    flops, params = thop.profile(model, inputs=(torch.randn(input_size),torch.randn(mask_size)))
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    print(f"FLOPs: {flops / 1e9} G")  #
    print(f"Params: {params / 1e6} M")  #


