from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
from model import GCN
import utils.util as util
import numpy as np


class AttModel(Module):

    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=2, dct_n=10):
        super(AttModel, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)
        assert kernel_size == 10

        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.gcn = GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """

        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:
        :return:
        """
        dct_n = self.dct_n
        src = src[:, :input_n]  # [bs,in_n,dim]
        src_tmp = src.clone()
        bs = src.shape[0]
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()

        vn = input_n - self.kernel_size - output_n + 1
        vl = self.kernel_size + output_n
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # [32,40,66*11]

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []

        key_tmp = self.convK(src_key_tmp / 1000.0)
        for i in range(itera):
            query_tmp = self.convQ(src_query_tmp / 1000.0)
            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])

            input_gcn = src_tmp[:, idx]
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
            dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)
            dct_out_tmp = self.gcn(dct_in_tmp)
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                   dct_out_tmp[:, :, :dct_n].transpose(1, 2))
            outputs.append(out_gcn.unsqueeze(2))
            if itera > 1:
                # update key-value query
                out_tmp = out_gcn.clone()[:, 0 - output_n:]
                src_tmp = torch.cat([src_tmp, out_tmp], dim=1)

                vn = 1 - 2 * self.kernel_size - output_n
                vl = self.kernel_size + output_n
                idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                          np.expand_dims(np.arange(vn, -self.kernel_size - output_n + 1), axis=1)

                src_key_tmp = src_tmp[:, idx_dct[0, :-1]].transpose(1, 2)
                key_new = self.convK(src_key_tmp / 1000.0)
                key_tmp = torch.cat([key_tmp, key_new], dim=2)

                src_dct_tmp = src_tmp[:, idx_dct].clone().reshape(
                    [bs * self.kernel_size, vl, -1])
                src_dct_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_dct_tmp).reshape(
                    [bs, self.kernel_size, dct_n, -1]).transpose(2, 3).reshape(
                    [bs, self.kernel_size, -1])
                src_value_tmp = torch.cat([src_value_tmp, src_dct_tmp], dim=1)

                src_query_tmp = src_tmp[:, -self.kernel_size:].transpose(1, 2)

        outputs = torch.cat(outputs, dim=2)
        return outputs


class AttModelPerParts(Module):

    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=2, dct_n=10,
                 parts_idx=[[1, 2, 3], [4, 5, 6]]):
        super(AttModelPerParts, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.parts_idx = parts_idx
        self.in_features = in_features
        self.dct_n = dct_n
        ks = int((kernel_size + 1) / 2)
        assert kernel_size == 10
        convQ = []
        convK = []
        for i in range(len(parts_idx)):
            pi = parts_idx[i]
            convQ.append(nn.Sequential(nn.Conv1d(in_channels=len(pi), out_channels=d_model, kernel_size=6,
                                                 bias=False),
                                       nn.ReLU(),
                                       nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                                 bias=False),
                                       nn.ReLU()))
            convK.append(nn.Sequential(nn.Conv1d(in_channels=len(pi), out_channels=d_model, kernel_size=6,
                                                 bias=False),
                                       nn.ReLU(),
                                       nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                                 bias=False),
                                       nn.ReLU()))
        self.convQ = nn.ModuleList(convQ)
        self.convK = nn.ModuleList(convK)

        self.gcn = GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """

        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:
        :return:
        """
        dct_n = self.dct_n
        src = src[:, :input_n]  # [bs,in_n,dim]
        src_tmp = src.clone()
        bs = src.shape[0]
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()

        vn = input_n - self.kernel_size - output_n + 1
        vl = self.kernel_size + output_n
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3)  # .reshape([bs, vn, -1])  # [32,40,66*11]

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []
        key_tmp = []
        for ii, pidx in enumerate(self.parts_idx):
            key_tmp.append(self.convK[ii](src_key_tmp[:, pidx] / 1000.0).unsqueeze(1))
        key_tmp = torch.cat(key_tmp, dim=1)

        for i in range(itera):
            query_tmp = []
            for ii, pidx in enumerate(self.parts_idx):
                query_tmp.append(self.convQ[ii](src_query_tmp[:, pidx] / 1000.0).unsqueeze(1))
            query_tmp = torch.cat(query_tmp, dim=1)
            score_tmp = torch.matmul(query_tmp.transpose(2, 3), key_tmp) + 1e-15
            att_tmp = score_tmp / (torch.sum(score_tmp, dim=3, keepdim=True))
            dct_att_tmp = torch.zeros([bs, self.in_features, self.dct_n]).float().cuda()
            for ii, pidx in enumerate(self.parts_idx):
                dct_att_tt = torch.matmul(att_tmp[:, ii], src_value_tmp[:, :, pidx]
                                          .reshape([bs, -1, len(pidx) * self.dct_n])).squeeze(1)
                dct_att_tt = dct_att_tt.reshape([bs, len(pidx), self.dct_n])
                dct_att_tmp[:, pidx, :] = dct_att_tt

            input_gcn = src_tmp[:, idx]
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
            dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)
            dct_out_tmp = self.gcn(dct_in_tmp)
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                   dct_out_tmp[:, :, :dct_n].transpose(1, 2))
            outputs.append(out_gcn.unsqueeze(2))
            # update key-value query
            out_tmp = out_gcn.clone()[:, 0 - output_n:]
            src_tmp = torch.cat([src_tmp, out_tmp], dim=1)

            vn = 1 - 2 * self.kernel_size - output_n
            vl = self.kernel_size + output_n
            idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                      np.expand_dims(np.arange(vn, -self.kernel_size - output_n + 1), axis=1)

            src_key_tmp = src_tmp[:, idx_dct[0, :-1]].transpose(1, 2)

            key_new = []
            for ii, pidx in enumerate(self.parts_idx):
                key_new.append(self.convK[ii](src_key_tmp[:, pidx] / 1000.0).unsqueeze(1))
            key_new = torch.cat(key_new, dim=1)
            key_tmp = torch.cat([key_tmp, key_new], dim=3)

            src_dct_tmp = src_tmp[:, idx_dct].clone().reshape(
                [bs * self.kernel_size, vl, -1])
            src_dct_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_dct_tmp).reshape(
                [bs, self.kernel_size, dct_n, -1]).transpose(2, 3)  # .reshape([bs, self.kernel_size, -1])
            src_value_tmp = torch.cat([src_value_tmp, src_dct_tmp], dim=1)

            src_query_tmp = src_tmp[:, -self.kernel_size:].transpose(1, 2)

        outputs = torch.cat(outputs, dim=2)
        return outputs