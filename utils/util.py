import torch
import numpy as np


def lr_decay_mine(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def orth_project(cam, pts):
    """

    :param cam: b*[s,tx,ty]
    :param pts: b*k*3
    :return:
    """
    s = cam[:, 0:1].unsqueeze(1).repeat(1, pts.shape[1], 2)
    T = cam[:, 1:].unsqueeze(1).repeat(1, pts.shape[1], 1)

    return torch.mul(s, pts[:, :, :2] + T)


def opt_cam(x, x_target):
    """
    :param x: N K 3 or  N K 2
    :param x_target: N K 3 or  N K 2
    :return:
    """
    if x_target.shape[2] == 2:
        vis = torch.ones_like(x_target[:, :, :1])
    else:
        vis = (x_target[:, :, :1] > 0).float()
    vis[:, :2] = 0
    xxt = x_target[:, :, :2]
    xx = x[:, :, :2]
    x_vis = vis * xx
    xt_vis = vis * xxt
    num_vis = torch.sum(vis, dim=1, keepdim=True)
    mu1 = torch.sum(x_vis, dim=1, keepdim=True) / num_vis
    mu2 = torch.sum(xt_vis, dim=1, keepdim=True) / num_vis
    xmu = vis * (xx - mu1)
    xtmu = vis * (xxt - mu2)

    eps = 1e-6 * torch.eye(2).float().cuda()
    Ainv = torch.inverse(torch.matmul(xmu.transpose(1, 2), xmu) + eps.unsqueeze(0))
    B = torch.matmul(xmu.transpose(1, 2), xtmu)
    tmp_s = torch.matmul(Ainv, B)
    scale = ((tmp_s[:, 0, 0] + tmp_s[:, 1, 1]) / 2.0).unsqueeze(1)

    scale = torch.clamp(scale, 0.7, 10)
    trans = mu2.squeeze(1) / scale - mu1.squeeze(1)
    opt_cam = torch.cat([scale, trans], dim=1)
    return opt_cam


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m
