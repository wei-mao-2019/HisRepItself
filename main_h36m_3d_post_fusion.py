from utils import h36motion3d as datasets
from model import AttModel, GCN
from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
# import h5py
import torch.optim as optim


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    in_features = opt.in_features  # 66
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    #
    net_pred_pose = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                      num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred_pose.cuda()
    model_path_len = './checkpoint/pretrained/h36m_3d_in50_out10_dctn20/ckpt_best.pth.tar'
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    net_pred_pose.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))
    net_pred_pose.eval()

    #
    joint_idxs = np.arange(in_features).reshape([-1, 3])
    joint_parts = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15, 16], [17, 18, 19, 20, 21]]
    parts_idx = []
    for ii, jidx in enumerate(joint_parts):
        parts_idx.append(joint_idxs[jidx].reshape([-1]).tolist())

    net_pred_part = AttModel.AttModelPerParts(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                              num_stage=opt.num_stage, dct_n=opt.dct_n, parts_idx=parts_idx)
    net_pred_part.cuda()
    model_path_len = './checkpoint/pretrained/main_h36m_3d_part_in50_out10_ks10_dctn20/ckpt_best.pth.tar'
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    net_pred_part.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))
    net_pred_part.eval()

    #
    parts_idx = np.arange(in_features).reshape([-1, 3]).tolist()
    net_pred_joint = AttModel.AttModelPerParts(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                               num_stage=opt.num_stage, dct_n=opt.dct_n, parts_idx=parts_idx)
    net_pred_joint.cuda()
    model_path_len = './checkpoint/pretrained/main_h36m_3d_joint_in50_out10_ks10_dctn20/ckpt_best.pth.tar'
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    net_pred_joint.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))
    net_pred_joint.eval()

    net_pred = GCN.GCN(input_feature=(kernel_size + opt.output_n) * 3, hidden_feature=d_model, p_dropout=0.3,
                       num_stage=opt.num_stage, node_n=in_features)
    net_pred.cuda()

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.is_load or opt.is_eval:
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
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    if not opt.is_eval:
        dataset = datasets.Datasets(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        valid_dataset = datasets.Datasets(opt, split=1)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)

    test_dataset = datasets.Datasets(opt, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)

    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, net_pred_pose, net_pred_part, net_pred_joint, is_train=3,
                             data_loader=test_loader, opt=opt)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test')
        # print('testing error: {:.3f}'.format(ret_test['m_p3d_h36']))
    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, net_pred_pose, net_pred_part, net_pred_joint, optimizer, is_train=0,
                                  data_loader=data_loader, epo=epo, opt=opt)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            ret_valid = run_model(net_pred, net_pred_pose, net_pred_part, net_pred_joint, is_train=1,
                                  data_loader=valid_loader, opt=opt, epo=epo)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
            ret_test = run_model(net_pred, net_pred_pose, net_pred_part, net_pred_joint, is_train=3,
                                 data_loader=test_loader, opt=opt, epo=epo)
            print('testing error: {:.3f}'.format(ret_test['#1']))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid['m_p3d_h36'] < err_best:
                err_best = ret_valid['m_p3d_h36']
                is_best = True
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)


def run_model(net_pred, net_pred_pose, net_pred_part, net_pred_joint, optimizer=None, is_train=0, data_loader=None,
              epo=1, opt=None):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = np.array(range(opt.output_n)) + 1
        m_p3d_h36 = np.zeros([opt.output_n])
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

    itera = 1
    idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
            out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))
    st = time.time()
    for i, (p3d_h36) in enumerate(data_loader):
        # print(i)
        batch_size, seq_n, _ = p3d_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        p3d_h36 = p3d_h36.float().cuda()
        p3d_sup = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_src = p3d_h36.clone()[:, :, dim_used]
        with torch.no_grad():
            p3d_out_pose = net_pred_pose(p3d_src, input_n=in_n, output_n=out_n, itera=itera)[:, :, 0]
            p3d_out_part = net_pred_part(p3d_src, input_n=in_n, output_n=out_n, itera=itera)[:, :, 0]
            p3d_out_joint = net_pred_joint(p3d_src, input_n=in_n, output_n=out_n, itera=itera)[:, :, 0]
        p3d_in = torch.cat([p3d_out_pose, p3d_out_part, p3d_out_joint], dim=1).transpose(1, 2)
        out_est = net_pred(p3d_in, is_out_resi=False)
        out_est = torch.nn.functional.softmax(out_est[:, 0:1, :3], dim=2)

        p3d_out_pose = p3d_out_pose[:, 10:].reshape([batch_size, -1]).unsqueeze(1)
        p3d_out_part = p3d_out_part[:, 10:].reshape([batch_size, -1]).unsqueeze(1)
        p3d_out_joint = p3d_out_joint[:, 10:].reshape([batch_size, -1]).unsqueeze(1)
        p3d_out_cand = torch.cat([p3d_out_pose, p3d_out_part, p3d_out_joint], dim=1)
        p3d_out_all = torch.matmul(out_est, p3d_out_cand).squeeze(1).reshape([batch_size, out_n, -1])

        p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out[:, :, dim_used] = p3d_out_all
        p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]
        p3d_out = p3d_out.reshape([-1, out_n, 32, 3])

        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 32, 3])

        p3d_out_all = p3d_out_all.reshape([batch_size, out_n, len(dim_used) // 3, 3])

        # 2d joint loss:
        grad_norm = 0
        if is_train == 0:
            loss_p3d = torch.mean(torch.norm(p3d_out_all - p3d_sup[:, seq_in:], dim=3))
            loss_all = loss_p3d
            optimizer.zero_grad()
            loss_all.backward()
            grad_norm = nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values
            l_p3d += loss_p3d.cpu().data.numpy() * batch_size

        if is_train <= 1:  # if is validation or train simply output the overall mean error
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3))
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        else:
            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{:.3f}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))
    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)
