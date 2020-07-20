from utils import h36motion3d as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    ckpt = './checkpoint/pretrained/h36m_3d_in50_out10_dctn20/'
    batch_size = 1
    opt.ckpt = ckpt
    print('>>> create models')
    net_pred = AttModel.AttModel(in_features=66, kernel_size=10, d_model=256,
                                 num_stage=12, dct_n=20)
    net_pred.cuda()
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    start_epoch = ckpt['epoch']
    err_best = ckpt['err']
    net_pred.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    print('>>> loading datasets')

    acts = ["walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"]
    good_idx = pd.read_csv('./checkpoint/pretrained/seq_selected.csv')
    good_idx = good_idx.values
    sele = {}
    for gi in range(good_idx.shape[0]):
        if good_idx[gi, 0] in sele.keys():
            sele[good_idx[gi, 0]].append(int(good_idx[gi, 1]))
        else:
            sele[good_idx[gi, 0]] = [int(good_idx[gi, 1])]

    err = np.zeros([2, opt.output_n])
    n = 0
    for act in acts:
        if not act in sele.keys():
            continue
        test_dataset = datasets.Datasets(opt, split=2, actions=[act])
        print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                 pin_memory=True)
        # evaluation
        ret, nt = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, good_idx=sele[act])
        err += ret
        n += nt
    err = err / n
    head = np.array(['input_n'])
    for k in range(1, opt.output_n + 1):
        head = np.append(head, [f'#{k}'])
    value = np.expand_dims(np.array(['in50', 'in100']), axis=1)
    value = np.concatenate([value, err.astype(np.str)], axis=1)
    log.save_csv_log(opt, head, value, is_create=True, file_name='test_in50_in100')


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None, good_idx=[]):
    net_pred.eval()
    titles = np.array(range(opt.output_n)) + 1
    m_p3d_h36 = np.zeros([2, opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    seq_in = opt.kernel_size
    # joints at same loc
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    itera = 3
    for i, (p3d_h36) in enumerate(data_loader):
        # print(i)
        if not (i in good_idx):
            continue
        batch_size, seq_n, _ = p3d_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        p3d_h36 = p3d_h36.float().cuda()
        p3d_src = p3d_h36.clone()[:, :, dim_used]

        p3d_src_50 = p3d_h36.clone()[:, -50 - out_n:, dim_used]
        p3d_out_all = net_pred(p3d_src_50, input_n=50, output_n=10, itera=itera)

        p3d_out_all = p3d_out_all[:, 10:].transpose(1, 2).reshape([batch_size, 10 * itera, -1])[:, :out_n]
        p3d_out_50 = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out_50[:, :, dim_used] = p3d_out_all
        p3d_out_50[:, :, index_to_ignore] = p3d_out_50[:, :, index_to_equal]
        p3d_out_50 = p3d_out_50.reshape([-1, out_n, 32, 3])

        p3d_src_100 = p3d_h36.clone()[:, :, dim_used]
        p3d_out_all = net_pred(p3d_src_100, input_n=100, output_n=10, itera=itera)

        p3d_out_all = p3d_out_all[:, 10:].transpose(1, 2).reshape([batch_size, 10 * itera, -1])[:, :out_n]
        p3d_out_100 = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out_100[:, :, dim_used] = p3d_out_all
        p3d_out_100[:, :, index_to_ignore] = p3d_out_100[:, :, index_to_equal]
        p3d_out_100 = p3d_out_100.reshape([-1, out_n, 32, 3])

        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 32, 3])

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out_50, dim=3), dim=2), dim=0)
        m_p3d_h36[0] += mpjpe_p3d_h36.cpu().data.numpy()

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out_100, dim=3), dim=2), dim=0)
        m_p3d_h36[1] += mpjpe_p3d_h36.cpu().data.numpy()
    return m_p3d_h36, n


if __name__ == '__main__':
    option = Options().parse()
    main(option)
