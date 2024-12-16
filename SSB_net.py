from torch import nn
import torch
from einops import rearrange
import torch.nn.functional as F
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SPABlock(nn.Module):
    def __init__(self,channels):
        super(SPABlock,self).__init__()
        #self.CA = CABlock(channels//2)
        self.conv3x3_1 = nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,groups=channels)
        self.conv1x1_1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.conv3x3_2 = nn.Conv2d(channels//2, channels, kernel_size=3, stride=1,padding=1)

        #self.act = nn.GELU()
        self.layer_norm = LayerNorm2d(channels)#nn.LayerNorm([3, 4], eps=1e-6)
        self.LN = True
    def act(self,x):
        x1,x2 = x.chunk(2,dim=1)
        return x1*x2
    def forward(self,input):

        # if self.LN:
        #     out = self.layer_norm(input)
        # else:
        #     out = input
        out = self.conv1x1_1(input)
        out = self.conv3x3_1(out)
        out = self.act(out)
        #out = self.CA(out)
        out = self.conv3x3_2(out)

        return out#+input

class FFNBlock(nn.Module):
    def __init__(self,channels):
        super(FFNBlock,self).__init__()
        self.conv3x3 = nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1)
        self.conv1x1_1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.conv1x1_2 = nn.Conv2d(channels//2, channels, kernel_size=1, stride=1)
        #self.act = nn.GELU()
        self.layer_norm = LayerNorm2d(channels)#nn.LayerNorm([3, 4], eps=1e-6)
        self.LN = True
    def act(self,x):
        x1,x2 = x.chunk(2,dim=1)
        return x1*x2
    def forward(self,input):
        if self.LN:
            out = self.layer_norm(input)
        else:
            out = input
        out = self.conv1x1_1(out)
        out = self.act(out)
        out = self.conv1x1_2(out)
        return out+input

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads
    ):
        super(MS_MSA,self).__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """

        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)  # b x (hw) x (dim_head*heads)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v  # v.shape = b x heads x (hw) x dim_head
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2).contiguous()    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out = self.proj(x).view(b, h, w, c)
        
        #out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        #out = out_c + out_p

        return out


class SSBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            channel
    ):
        super(SSBlock,self).__init__()
        self.spa = SPABlock(channel)
        self.msab = MS_MSA(dim=channel, dim_head=dim_head, heads=heads)
        self.ffn = FFNBlock(channel)
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        #x1, x2 = x.chunk(2, dim=1)

        x1 = x.permute(0, 2, 3, 1).contiguous()
        x1 = self.msab(x1)
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x2 = self.spa(x)
        #print(x1.shape,x2.shape)
        out = x1+x2#torch.cat([x1,x2],dim=1)
        out = self.ffn(out)
        return x+out

class SSB_net(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
            channel
    ):
        super(SSB_net,self).__init__()

        #self.convin = nn.Conv2d(dim, dim, 3, 1, 1)
        # 4 heads 8 headsdim 64 dim
        self.base_64_encode = SSBlock(dim=dim,dim_head=dim,heads=heads,channel = channel)
        self.downsample_1 = nn.Conv2d(dim, 96, 4, 2, 1)
        self.base_128_encode = SSBlock(dim=96,dim_head=96,heads=heads,channel = 96)
        self.downsample_2 = nn.Conv2d(96, 128, 4, 2, 1)
        self.bottle = SSBlock(dim=128,dim_head=128,heads=heads,channel = 128)
        self.upsample_1 = nn.ConvTranspose2d(128, 96, 4, 2, 1)
        self.base_128_decode = SSBlock(dim=96,dim_head=96,heads=heads,channel = 96)
        self.upsample_2 = nn.ConvTranspose2d(96, dim, 4, 2, 1)
        self.base_64_decode = SSBlock(dim=dim,dim_head=64,heads=heads,channel = channel)

        #self.convout = nn.Conv2d(dim, dim, 3, 1, 1)

        self.fusion_1 = nn.Conv2d(192, 96, 3, 1, 1)
        self.fusion_2 = nn.Conv2d(128, dim, 3, 1, 1)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        #fea_in = self.convin(x)
        fea_1 = self.base_64_encode(x)
        fea = self.downsample_1(fea_1)
        fea_2 = self.base_128_encode(fea)
        fea = self.downsample_2(fea_2)
        fea = self.bottle(fea)
        fea = self.upsample_1(fea)
        fea = self.fusion_1(torch.cat([fea,fea_2],dim=1))
        fea = self.base_128_decode(fea)
        fea = self.upsample_2(fea)
        fea = self.fusion_2(torch.cat([fea,fea_1],dim=1))
        fea = self.base_64_decode(fea)
        #out = self.convout(fea)
        out = fea+x
        return out

if __name__ == '__main__':
    import os
    import time

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    height = 128
    width = 128
    model = SSB_net(64, 8, 4,8,64).cuda()
    #model = model.cuda()
    print(model)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    x = torch.randn((4, 64, height, width)).to(device)
    # m = torch.randn((1, 1, height, width)).to(device)
    for i in range(1):
        with torch.no_grad():
            start = time.time()
            x_rec = model(x)
            end = time.time()
            print('time:', end - start)
    print(x_rec.shape)