from utils import h36motion as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import h5py
import torch.optim as optim


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    in_features = 48
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.cuda()
    model_path_len = '{}/ckpt_best.pth.tar'.format(opt.ckpt)
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    start_epoch = ckpt['epoch'] + 1
    err_best = ckpt['err']
    lr_now = ckpt['lr']
    net_pred.load_state_dict(ckpt['state_dict'])
    # net.load_state_dict(ckpt)
    # optimizer.load_state_dict(ckpt['optimizer'])
    # lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2)
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    head = np.array(['act'])
    for k in range(1, opt.output_n + 1):
        head = np.append(head, [f'#{k}'])

    acts = ["walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"]
    errs = np.zeros([len(acts) + 1, opt.output_n])
    for i, act in enumerate(acts):
        test_dataset = datasets.Datasets(opt, split=2, actions=[act])
        print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
        test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                 pin_memory=True)

        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        print('testing error: {:.3f}'.format(ret_test['#1']))
        ret_log = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
        errs[i] = ret_log
    errs[-1] = np.mean(errs[:-1], axis=0)
    acts = np.expand_dims(np.array(acts + ["average"]), axis=1)
    value = np.concatenate([acts, errs.astype(np.str)], axis=1)
    log.save_csv_log(opt, head, value, is_create=True, file_name='test_pre_action_256_seq')


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    net_pred.eval()

    titles = np.array(range(opt.output_n)) + 1
    m_ang_seq = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    dim_used = np.array([6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42,
                         43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85,
                         86])
    seq_in = opt.kernel_size

    itera = 3
    idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
            out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))
    for i, (ang_h36) in enumerate(data_loader):
        # print(i)
        batch_size, seq_n, _ = ang_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        ang_h36 = ang_h36.float().cuda()
        ang_sup = ang_h36.clone()[:, :, dim_used][:, -out_n - seq_in:]
        ang_src = ang_h36.clone()[:, :, dim_used]
        # ang_src = ang_src.permute(1, 0, 2)  # seq * n * dim
        # ang_src = ang_src[:in_n]
        ang_out_all = net_pred(ang_src, output_n=10, dct_n=opt.dct_n,
                               itera=itera, input_n=in_n)
        ang_out_all = ang_out_all[:, seq_in:].transpose(1, 2).reshape([batch_size, 10 * itera, -1])[:, :out_n]
        ang_out = ang_h36.clone()[:, in_n:in_n + out_n]
        ang_out[:, :, dim_used] = ang_out_all

        ang_out_euler = ang_out.reshape([-1, 99]).reshape([-1, 3])
        ang_gt_euler = ang_h36[:, in_n:in_n + out_n].reshape([-1, 99]).reshape([-1, 3])

        import utils.data_utils as data_utils
        ang_out_euler = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(ang_out_euler))
        ang_out_euler = ang_out_euler.view(-1, out_n, 99)
        ang_gt_euler = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(ang_gt_euler))
        ang_gt_euler = ang_gt_euler.view(-1, out_n, 99)

        eulererr_ang_seq = torch.sum(torch.norm(ang_out_euler - ang_gt_euler, dim=2), dim=0)
        m_ang_seq += eulererr_ang_seq.cpu().data.numpy()

    ret = {}
    m_ang_h36 = m_ang_seq / n
    for j in range(out_n):
        ret["#{:d}".format(titles[j])] = m_ang_h36[j]
    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)
