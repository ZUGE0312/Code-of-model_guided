# coding:utf-8


import torch
import torch.nn as nn


def x2y(x, mask):
    return torch.sum(x.mul(mask), dim=1, keepdim=True)


def y2x(y, mask):
    return y.mul(mask)


def get_maskng(mask):
    p0 = torch.where(mask == 0)
    p1 = torch.where(mask == 1)
    mask[p0] = 1
    mask[p1] = 0
    return mask


def replace(gt, pred, mask):
    p0 = torch.where(mask == 0)
    p1 = torch.where(mask == 1)
    mask[p0] = 1
    mask[p1] = 0
    return gt + pred.mul(mask)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


def interp2d(x, y, xp, yp, zp):
    """
    Bilinearly interpolate over regular 2D grid.
    `xp` and `yp` are 1D arrays defining grid coordinates of sizes :math:`N_x`
    and :math:`N_y` respectively, and `zp` is the 2D array, shape
    :math:`(N_x, N_y)`, containing the gridded data points which are being
    interpolated from. Note that the coordinate grid should be regular, i.e.
    uniform grid spacing. `x` and `y` are either scalars or 1D arrays giving
    the coordinates of the points at which to interpolate. If these are outside
    the boundaries of the coordinate grid, the resulting interpolated values
    are evaluated at the boundary.
    Parameters
    ----------
    x : 1D array or scalar
        x-coordinates of interpolating point(s).
    y : 1D array or scalar
        y-coordinates of interpolating point(s).
    xp : 1D array, shape M
        x-coordinates of data points zp. Note that this should be a *regular*
        grid, i.e. uniform spacing.
    yp : 1D array, shape N
        y-coordinates of data points zp. Note that this should be a *regular*
        grid, i.e. uniform spacing.
    zp : 2D array, shape (M, N)
        Data points on grid from which to interpolate.
    Returns
    -------
    z : 1D array or scalar
        Interpolated values at given point(s).
    """
    # if scalar, turn into array
    # scalar = False
    # if not isinstance(x, (list, np.ndarray)):
    #     scalar = True
    #     x = np.array([x])
    #     y = np.array([y])

    # grid spacings and sizes
    hx = xp[1] - xp[0]
    hy = yp[1] - yp[0]
    Nx = xp.size(0)
    Ny = yp.size(0)

    # snap beyond-boundary points to boundary
    x.clone()[x < xp[0]] = xp[0]
    y.clone()[y < yp[0]] = yp[0]
    x.clone()[x > xp[-1]] = xp[-1]
    y.clone()[y > yp[-1]] = yp[-1]

    # find indices of surrounding points
    i1 = torch.floor((x - xp[0]) / hx).long().cuda()
    i1[i1 == Nx - 1] = Nx - 2
    # i1 = torch.where(i1 == Nx - 1, i1, Nx - 2)
    j1 = torch.floor((y - yp[0]) / hy).long().cuda()
    j1[j1 == Ny - 1] = Ny - 2
    # j1 = torch.where(j1 == Ny - 1, j1, Ny - 2)
    i2 = i1 + 1
    j2 = j1 + 1

    # get coords and func at surrounding points
    x1 = xp[i1]
    x2 = xp[i2]
    y1 = yp[j1]
    y2 = yp[j2]
    z11 = zp[i1, j1]
    z21 = zp[i2, j1]
    z12 = zp[i1, j2]
    z22 = zp[i2, j2]

    # interpolate
    t11 = z11 * (x2 - x) * (y2 - y)
    t21 = z21 * (x - x1) * (y2 - y)
    t12 = z12 * (x2 - x) * (y - y1)
    t22 = z22 * (x - x1) * (y - y1)
    z = (t11 + t21 + t12 + t22) / (hx * hy)
    # if scalar:
    #     z = z[0]
    return z


def interpp(y, mask):
    # print(y.shape, mask.shape)  # b, 1, h, w  b, c, h, w
    y = y.repeat(1, mask.size(1), 1, 1)
    mask = mask.repeat(y.size(0) // mask.size(0), 1, 1, 1)
    index = mask.bool()
    x = y
    x_1 = []
    for i in range(mask.size(1)):
        x_1.append(x[:,i,:,:].masked_select(index[:,i,:,:]).reshape(mask.size(0), mask.size(2) // 4, mask.size(3) // 4))
    
    w, h = y.size(2), y.size(3)
    gt_i = torch.arange(0, w, 4).cuda()
    gt_j = torch.arange(0, h, 4).cuda()
    # gt_ii, gt_jj = torch.meshgrid(gt_i, gt_j)
    inter_i = torch.arange(0, w, 1).cuda()
    inter_j = torch.arange(0, h, 1).cuda()
    inter_ii, inter_jj = torch.meshgrid(inter_i, inter_j)
    x_2 = torch.zeros(size=(y.size(0), len(x_1), y.size(2), y.size(3)))
    for c in range(len(x_1)):
        for b in range(y.size(0)):
            x_2[b, c, :, :] = (interp2d(inter_ii, inter_jj, gt_i, gt_j, x_1[c][b, :, :]))

    x_3 = y
    for c in range(y.size(1)):
        for i in range(4):
            for j in range(4):
                if mask[0,c,i,j] == 1:
                    x_3[:,c,i:,j:] = x_2[:,c,:mask.size(2)-i,:mask.size(3)-j]
    # x_3：对齐
    return x_3

