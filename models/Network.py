import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.Dynamic_Conv import dynamic_conv_blcok
from models.models import register
from models.LFEM import local_enhanced_blcok, default_conv, Upsampler
from utils import make_coord

import numpy as np


class TwoDEncDec(torch.nn.Module):
    def __init__(self, chans, M):
        super(TwoDEncDec, self).__init__()
        self.M = M
        self.conv = torch.nn.Conv2d(1, chans, 3, 1, 1)

        self.conv_m = torch.nn.Conv2d(chans, 1, 3, 1, 1)

        # For Height dimennsion
        self.conv_H = torch.nn.Sequential(
            # torch.nn.AdaptiveAvgPool2d((None, 1)),
            torch.nn.Conv2d(chans, self.M, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            torch.nn.Sigmoid()
        )

        # For Width dimennsion

        self.conv_W = torch.nn.Sequential(
            # torch.nn.AdaptiveAvgPool2d((1, 64)),         #64 for training, 256 for testing
            torch.nn.Conv2d(chans, self.M, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            torch.nn.Sigmoid()
        )
        self.sigmoid = torch.nn.Sigmoid()

    # end

    def forward(self, x):
        N_, C_, H_, W_ = x.shape


        if H_ != 64:
            patch_size = 64
            new_matrix_h = torch.zeros([N_, C_, H_, W_]).cuda()
            new_matrix_w = torch.zeros([N_, C_, H_, W_]).cuda()

            for i in range(0, 256, patch_size):  # 256 or 240
                x_patch = x[:, :, i:i + patch_size, i:i + patch_size]
                s0_3_h = self.conv_H(x_patch)

                s0_3_w = self.conv_W(x_patch)
                new_matrix_h[:, i:i + patch_size, i:i + patch_size, :] = s0_3_h
                new_matrix_w[:, i:i + patch_size, i:i + patch_size, :] = s0_3_w

                # new_matrix[:, i:i + patch_size, i:i + patch_size,:] = torch.einsum('bij,bkj->bikj', s0_3_h, s0_3_w)
            output_x = new_matrix_h.reshape(N_, -1, C_)
            output_y = new_matrix_w.reshape(N_, -1, C_)

        else:
            output_x = self.conv_H(x)

            output_y = self.conv_W(x)



        #########################################################################################

        return output_x, output_y


class SAM(nn.Module):
    def __init__(self, nf, use_residual=True, learnable=True):
        super(SAM, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

        if self.learnable:
            self.conv_shared = nn.Sequential(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True),
                                             nn.ReLU(inplace=True))
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref):
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            style = self.conv_shared(torch.cat([lr, ref], dim=1))
            gamma = self.conv_gamma(style)
            beta = self.conv_beta(style)

        b, c, h, w = lr.size()
        lr = lr.view(b, c, h * w)
        lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        out = ref_normed * gamma + beta

        return out




class EncDec(torch.nn.Module):
    def __init__(self, chans, M):
        super(EncDec, self).__init__()
        self.M = M
        half_chans = chans
        # chans = chans * 2
        self.conv = torch.nn.Conv2d(1, chans // 2, 3, 1, 1)

        self.conv_m = torch.nn.Conv2d(chans // 2, 1, 3, 1, 1)

        self.conv_down = torch.nn.Conv2d(chans, chans // 2, 3, 1, 1)

        self.conv_gamma = torch.nn.Conv2d(chans, half_chans, 3, 1, 1)
        self.conv_beta = torch.nn.Conv2d(chans, half_chans, 3, 1, 1)

        # For Channel dimennsion
        self.conv_C = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(chans, self.M * chans, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            torch.nn.Sigmoid()
        )

        # For Height dimennsion
        self.conv_H = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((None, 1)),
            torch.nn.Conv2d(chans, self.M, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            torch.nn.Sigmoid()
        )

        # For Width dimennsion
        self.conv_W = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, None)),
            torch.nn.Conv2d(chans, self.M, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            torch.nn.Sigmoid()
        )
        self.norm_layer = nn.InstanceNorm2d(chans, affine=False)

        self.sigmoid = torch.nn.Sigmoid()

    # end

    def forward(self, x, y):
        N_, C_, H_, W_ = x.shape
        # print(x.shape,y.shape)
        # fuse = torch.cat([x,y],dim=1)
        fuse = x

        s0_3_c = self.conv_C(fuse)
        s0_3_c = s0_3_c.view(N_, self.M, -1, 1, 1)

        s0_3_h = self.conv_H(fuse)
        s0_3_h = s0_3_h.view(N_, self.M, 1, -1, 1)

        s0_3_w = self.conv_W(fuse)
        s0_3_w = s0_3_w.view(N_, self.M, 1, 1, -1)

        cube0 = (s0_3_c * s0_3_h * s0_3_w).mean(1) * fuse
        #########################################################################################


        gamma = self.conv_gamma(cube0)
        beta = self.conv_beta(cube0)

        tar = x.view(N_, C_, -1)
        tar_mean = torch.mean(tar, dim=-1, keepdim=True).unsqueeze(3)
        tar_std = torch.std(tar, dim=-1, keepdim=True).unsqueeze(3)

        gamma = gamma + tar_std
        beta = beta + tar_mean

        out_y = self.norm_layer(y) * gamma + beta

        return out_y


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

    def bis(self, input, dim, index):
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, V, K, Q):
        ### search
        Q_unfold = F.unfold(Q, kernel_size=(3, 3), padding=1)
        K_unfold = F.unfold(K, kernel_size=(3, 3), padding=1)
        K_unfold = K_unfold.permute(0, 2, 1)

        K_unfold = F.normalize(K_unfold, dim=2)  # [N, Hr*Wr, C*k*k]
        Q_unfold = F.normalize(Q_unfold, dim=1)  # [N, C*k*k, H*W]

        R_lv3 = torch.bmm(K_unfold, Q_unfold)  # [N, Hr*Wr, H*W]
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1)  # [N, H*W]

        ### transfer
        V_unfold = F.unfold(V, kernel_size=(3, 3), padding=1)

        T_lv3_unfold = self.bis(V_unfold, 2, R_lv3_star_arg)

        T_lv3 = F.fold(T_lv3_unfold, output_size=Q.size()[-2:], kernel_size=(3, 3), padding=1) / (3. * 3.)

        S = R_lv3_star.view(R_lv3_star.size(0), 1, Q.size(2), Q.size(3))

        return S, T_lv3


@register('DINet')
class DINet(nn.Module):
    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=256):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.ref_conv = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.ref_aux = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        # self.Dycon = OSAdapt(channels=self.encoder.out_dim)
        self.Dycon = dynamic_conv_blcok()

        self.phase = nn.Linear(2, hidden_dim // 2, bias=False)
        self.local_enhanced_block = local_enhanced_blcok()
        self.down_proj_t = nn.Conv2d(hidden_dim, self.encoder.out_dim // 2, 3, padding=1)
        self.down_proj_r = nn.Conv2d(hidden_dim, self.encoder.out_dim // 2, 3, padding=1)
        self.n_resblocks = 1
        self.conv1 = nn.Conv2d(self.encoder.out_dim, self.encoder.out_dim // 2, kernel_size=1)
        self.b_tail = nn.Conv2d(self.encoder.out_dim // 2, hidden_dim, kernel_size=1)
        self.kron_proj = EncDec(chans=self.encoder.out_dim // 2, M=16)
        self.adap = SAM(hidden_dim, use_residual=True, learnable=True)
        m_transformers = [Transformer() for _ in range(self.n_resblocks)]
        self.transformers = nn.Sequential(*m_transformers).cuda()
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})
        self.use_dynamic_conv = False


    def gen_feat(self, inp, ref, ref_hr, scale, flag=False):
        self.inp = inp

        self.feat_coord = make_coord(inp.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(inp.shape[0], 2, *inp.shape[-2:])
        self.ref_feat_hr, self.ref_loss = self.local_enhanced_block(ref)

        self.feat = self.encoder(inp)
        self.feat, self.ref_feat_hr = self.Dycon(self.feat, self.ref_feat_hr, scale, flag)

        self.coeff = self.coef(self.feat)
        self.freqq = self.freq(self.feat)
        self.ref_feat_hr_res = self.ref_conv(self.ref_feat_hr)
        self.ref_feat_hr_aux = self.ref_aux(self.ref_feat_hr)

        return self.feat

    def query_rgb(self, coord, cell=None, flag=False):
        feat = self.feat
        coef = self.coeff
        freq = self.freqq
        ref_feat_hr_res = self.ref_feat_hr_res
        ref_feat_hr_aux = self.ref_feat_hr_aux

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        feat_coord = self.feat_coord.cuda()

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_freq = F.grid_sample(
                    freq, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                ref_hr_res = F.grid_sample(
                    ref_feat_hr_res, coord_.flip(-1).unsqueeze(1),
                    align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)  # R
                ref_hr_aux = F.grid_sample(
                    ref_feat_hr_aux, coord_.flip(-1).unsqueeze(1),
                    align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)  # R

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                # prepare cell
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]

                # fourier projection
                bs, q = coord.shape[:2]

                # basis generation
                phase = self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2)
                q_freq = q_freq + phase
                q_freq = torch.cat((torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1)

                # ref information
                q_freq_ref = torch.stack(torch.split(ref_hr_res, 2, dim=-1), dim=-1)
                q_freq_ref = torch.mul(q_freq_ref, rel_coord.unsqueeze(-1))
                q_freq_ref = torch.sum(q_freq_ref, dim=-2)
                q_freq_ref += phase
                q_freq_ref = torch.cat((torch.cos(np.pi * q_freq_ref), torch.sin(np.pi * q_freq_ref)), dim=-1)
                # 
                # # integration
                inp = torch.mul(q_coef, q_freq)
                ref = torch.mul(ref_hr_aux, q_freq_ref)

                # # fusion block
                inp_dwt = (inp.permute(0, 2, 1)).unsqueeze(-1)
                ref_dwt = (ref.permute(0, 2, 1)).unsqueeze(-1)
                bs, chans, q, _ = inp_dwt.shape
                size = int(torch.tensor(q).sqrt())

                inp_dwt = self.down_proj_t(inp_dwt.reshape(bs, chans, size, size))
                ref_dwt_old = self.down_proj_r(ref_dwt.reshape(bs, chans, size, size))
                ref_dwt = self.kron_proj(inp_dwt, ref_dwt_old)

                if size != 64:
                    patch_size = 64

                    patches = []
                    new_matrix = torch.zeros([bs, 32, size, size]).cuda()  # 32 chans
                    # for i in range(0, 240, patch_size):     # 256 or 240
                    for i in range(0, 256, patch_size):  # 256 or 240
                        inp_dwt_patch = inp_dwt[:, :, i:i + patch_size, i:i + patch_size]
                        ref_dwt_patch = ref_dwt[:, :, i:i + patch_size, i:i + patch_size]
                        for i in range(self.n_resblocks):
                            S, T = self.transformers[i](ref_dwt_patch, ref_dwt_patch, inp_dwt_patch)
                            T = torch.cat([inp_dwt_patch, T], 1)
                            T = self.conv1(T)
                            inp_dwt_patch = inp_dwt_patch + T * S
                            new_matrix[:, :, i:i + patch_size, i:i + patch_size] = inp_dwt_patch
                            patches.append(inp_dwt_patch)
                    inp_dwt = new_matrix
                else:
                    for i in range(self.n_resblocks):
                        S, T = self.transformers[i](ref_dwt, ref_dwt, inp_dwt)
                        T = torch.cat([inp_dwt, T], 1)
                        T = self.conv1(T)
                        inp_dwt = inp_dwt + T * S

                inp_final = self.b_tail(inp_dwt).reshape(bs, q, -1) + inp
                pred = self.imnet((inp_final).contiguous().view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0];
        areas[0] = areas[3];
        areas[3] = t
        t = areas[1];
        areas[1] = areas[2];
        areas[2] = t

        ret = 0

        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear', \
                             padding_mode='border', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        return ret

    def forward(self, inp, inp_hr_coord, inp_cell, ref, ref_hr, scale=None, flag=True):
        self.gen_feat(inp, ref, ref_hr, scale, flag)
        return self.query_rgb(inp_hr_coord, inp_cell, flag), self.ref_loss

