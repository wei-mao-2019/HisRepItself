from utils import amass3d as datasets
# from utils import dpw3d as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from progress.bar import Bar
import time
import h5py
import torch.optim as optim


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    in_features = opt.in_features  # 54
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred = nn.DataParallel(net_pred)
    net_pred.cuda()

    model_path_len = '{}/ckpt_best.pth.tar'.format(opt.ckpt)
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    net_pred.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))
    net_pred = net_pred.module

    print('>>> loading datasets')
    test_dataset = datasets.Datasets(opt, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=False)

    ret_test = run_model(net_pred, is_train=2, data_loader=test_loader, opt=opt, epo=0)
    print('testing error: {:.3f}'.format(ret_test['#1']))
    ret_log = np.array([])
    head = np.array([])
    for k in ret_test.keys():
        ret_log = np.append(ret_log, [ret_test[k]])
        head = np.append(head, ['test_' + k])
    log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_bml1')


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    # l_beta = 0
    # j17to14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = np.array(range(opt.output_n)) + 1
        m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    itera = 1
    in_n = opt.input_n
    out_n = opt.output_n
    joint_used = np.arange(4, 22)
    seq_in = opt.kernel_size
    idx = np.expand_dims(np.arange(seq_in + 1), axis=1) + np.expand_dims(np.arange(out_n), axis=0)
    st = time.time()
    for i, (p3d_h36) in enumerate(data_loader):
        batch_size, seq_n, _, _ = p3d_h36.shape
        n += batch_size
        bt = time.time()
        p3d_h36 = p3d_h36.float().cuda()[:, :, joint_used] * 1000
        p3d_sup = p3d_h36.clone()[:, -out_n - seq_in:]
        p3d_src = p3d_h36.clone().reshape([batch_size, in_n + out_n, len(joint_used) * 3])

        p3d_out_all = net_pred(p3d_src, output_n=25, input_n=in_n, itera=itera)

        p3d_out = p3d_out_all[:, seq_in:].transpose(1, 2).reshape([batch_size, 25 * itera, len(joint_used), 3])[:,
                  :out_n]

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()

    ret = {}
    m_p3d_h36 = m_p3d_h36 / n
    for j in range(out_n):
        ret["#{:d}".format(titles[j])] = m_p3d_h36[j]

    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)
