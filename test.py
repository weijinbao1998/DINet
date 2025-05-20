import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchinfo import summary
import time
import datasets
import models
import utils
from thop import profile





def batched_predict(model, inp,ref, ref_hr, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp,ref, ref_hr)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

def natural_batched_predict(model, inp, coord, scale, bsize):
    if coord is None:
        with torch.no_grad():
            pred = model(inp)
    else:
        with torch.no_grad():
            model.gen_feat(inp)
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                pred = model.query_rgb(coord[:, ql:qr, :], scale[:, ql:qr, :])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)
    return pred


def natural_eval_psnr(loader, model, data_norm=None, eval_type=None,
              eval_bsize=None, verbose=False):
    if model is not None:
        model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()

    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    scale = None
    if eval_type is None:
        metric_fn = utils.calc_psnr
    else:
        dataset = eval_type.split('-')[0]
        # scale = int(eval_type.split('-')[1])
        scale = float(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset=dataset, scale=scale)

    val_res = utils.Averager()
    pbar = tqdm(loader, leave=False, desc='val')

    output_idx = 1
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, batch['coord'], batch['scale'])
        else:
            if scale is not None and scale > 4 and cell_decode:
                pred = batched_predict(model, inp, batch['coord'], batch['scale'] * scale / 4, eval_bsize)
            else:
                pred = batched_predict(model, inp, batch['coord'], batch['scale'], eval_bsize)

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        b, c, ih, iw = batch['inp'].shape
        s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
        shape = [b, round(ih * s), round(iw * s), c]
        pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
        batch['gt'] = batch['gt'].view(*shape).permute(0, 3, 1, 2).contiguous()
        # print(pred.shape,batch['gt'].shape)
        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if eval_type is not None:
            data_case = eval_type.split('-')
            save_path = f'./output/{model_name}/{data_case[0].upper()}/{data_case[1]}x'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for i in range(pred.shape[0]):
                transforms.ToPILImage()(pred[i]).save(f'{save_path}/test_{output_idx:>03}.png')
                output_idx += 1

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()

from PIL import Image

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))

def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, window_size=0, scale_max=None, fast=False,
              verbose=False, save_path=None,flag=None,test=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'ref': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['ref']
    ref_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    ref_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    t = data_norm['ref']
    ref_hr_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    ref_hr_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
        metric_fn_ssim = utils.calc_ssim
        scale = 8
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()
    val_res_ssim = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    count = 0
    for batch in pbar:

        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        ref = (batch['ref'] - ref_sub) / ref_div
        ref_hr = (batch['ref_hr'] - ref_hr_sub) / ref_hr_div  # SwinIR Evaluation - reflection padding
        if window_size != 0:
            _, _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]
            
            ref = torch.cat([ref, torch.flip(ref, [2])], 2)[:, :, :h_old + h_pad, :]
            ref = torch.cat([ref, torch.flip(ref, [3])], 3)[:, :, :, :w_old + w_pad]

            coord = utils.make_coord((scale * (h_old + h_pad), scale * (w_old + w_pad))).unsqueeze(0).cuda()
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / inp.shape[-2] / scale
            cell[:, :, 1] *= 2 / inp.shape[-1] / scale
        else:
            h_pad = 0
            w_pad = 0

            coord = batch['inp_hr_coord']
            cell = batch['inp_cell']
            if flag:
                scale = (scale_max, scale_max)
            else:
                scale = batch['scale']

        if eval_bsize is None:
            with torch.no_grad():
                pred, _ = model(inp, coord, cell, ref, ref_hr,scale,flag)

                if test:
                    start_time = time.time()
                    flops, params = profile(model.cuda(), (torch.randn(1, 1, 64, 64).cuda(),torch.randn(1, 4096, 2).cuda(),torch.randn(1,4096, 2).cuda(),torch.randn(1, 1, 64, 64).cuda(),torch.randn(1, 1, 64, 64).cuda()))
                    end_time = time.time()
                    print(' params: %.2f M,flops: %.2f G, inference times: %s ms' % (params / 1000000.0,flops / 1000000000.0,  round((end_time-start_time)*1000)))
        else:
            if fast:
                pred = model(inp, coord, cell * max(scale / scale_max, 1))
            else:
                pred = batched_predict(model, inp, coord, cell * max(scale / scale_max, 1),
                                       eval_bsize)  # cell clip for extrapolation

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)



        save = True

        if save == True:
            pred_temp = pred.view(256, 256).cpu()
            # pred_temp = pred.view(240, 240).cpu()

            figname = str(count) + '.png'
            figpath = os.path.join(save_path, figname)
            transforms.ToPILImage()(pred_temp).save(figpath)
            count = count + 1

        if eval_type is not None and fast == False:  # reshape for shaving-eval
            # gt reshape
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

            # prediction reshape
            ih += h_pad
            iw += w_pad
            s = math.sqrt(coord.shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            pred = pred[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]

        # res = metric_fn(pred, batch['gt'])
        # res_ssim = metric_fn_ssim(pred, batch['gt'])
        # inp = inp * gt_div + gt_sub
        # inp.clamp_(0, 1)
        # res = metric_fn(resize_fn(inp[0],(256,256)).view(1,-1,1).cuda(), batch['gt'].cuda())
        # res_ssim = metric_fn_ssim(resize_fn(inp[0],(256,256)).view(1,-1,1).cuda(), batch['gt'].cuda())
        res = metric_fn(pred, batch['gt'])
        res_ssim = metric_fn_ssim(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])
        val_res_ssim.add(res_ssim.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item(), val_res_ssim.item()

import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', default='config/test_set5_2.yaml')
    # parser.add_argument('--model', default='save/DIV2K/epoch-last-btc.pth')
    # parser.add_argument('--model', default='save/DIV2K/epoch-last-btc6srno.pth')
    # parser.add_argument('--config', default='config/test_mcsr-brain_IVD.yaml')
    # parser.add_argument('--model', default='save/IVD/epoch-last-rdn-orgbtc.pth')
    parser.add_argument('--config', default='config/test_mcsr-brain_IXI.yaml')
    parser.add_argument('--model', default='/home/wjb/McASSR/save/IXI/epoch-last-rdn-DINet.pth')
    # parser.add_argument('--model', default='/home/wjb/McASSR/save/IXI/epoch-last-ITNSR.pth')
    # parser.add_argument('--model', default='/home/wjb/McASSR/save/IVD/epoch-last-lit.pth')



    # parser.add_argument('--config', default='config/test_mcsr-brain_Brats.yaml')
    # parser.add_argument('--model', default='save/Brats/epoch-last-rdn-DINet.pth')
    # parser.add_argument('--model', default='save/Brats/epoch-last.pth')
    parser.add_argument('--window', default='0')
    parser.add_argument('--scale_max', default='2')
    parser.add_argument('--fast', default=False)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--test', default=False)
    # parser.add_argument('--save_path', default='/ssd/pyw/Arbitray-scale-MRI/McASSR/save/visual_results/IVD_fig1')
    # parser.add_argument('--save_path', default='/ssd/pyw/Arbitray-scale-MRI/McASSR/save/visual_results/IXI_fig1')
    parser.add_argument('--save_path', default='/home/wjb/McASSR/results/real_world')


    args = parser.parse_args()

    model_name = os.path.basename(args.model)  # 获取模型文件名
    model_name_without_ext = os.path.splitext(model_name)[0]  # 去掉扩展名
    args.save_path = os.path.join(args.save_path, model_name_without_ext)

    # 如果文件夹不存在，则创建它
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print(f"Updated save_path: {args.save_path}")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=0, pin_memory=True)

    model_spec = torch.load(args.model,weights_only=False)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    # res =  natural_eval_psnr(loader, model,
    #                       data_norm=config.get('data_norm'),
    #                       eval_type=config.get('eval_type'),
    #                       eval_bsize=config.get('eval_bsize'),
    #                       verbose=True)
    res, ssim = eval_psnr(loader, model,
                          data_norm=config.get('data_norm'),
                          eval_type=config.get('eval_type'),
                          eval_bsize=config.get('eval_bsize'),
                          window_size=int(args.window),
                          scale_max=int(args.scale_max),
                          fast=args.fast,
                          verbose=True,
                          save_path=args.save_path,
                          flag=True,
                          test=args.test)
    print(args.model)
    # print('psnr: {:.2f}'.format(res))

    print('psnr: {:.2f}'.format(res), 'ssim: {:.4f}'.format(ssim))
    # log_file_path = os.path.join(args.save_path, 'results.log')
    # logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')
    # log_message = 'psnr: {:.2f}, ssim: {:.4f}'.format(res, ssim)
    # logging.info(log_message)