import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input1', default='D:\MC-VarNet\dataset\IXI/test\T2_2x/00006907.png')
    parser.add_argument('--input2', default='D:\MC-VarNet\dataset\IXI/test\T2_160/00006907.png')
    parser.add_argument('--input3', default='D:\MC-VarNet\dataset\IXI/test\PD_160/00006907.png')

    parser.add_argument('--model', default='F:\McASSR\save\_train_mcsr-brain\epoch-last.pth')
    parser.add_argument('--scale', default='2.0', type=float)
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    scale_max = 4  # Maximum scale factor during training

    img1 = transforms.ToTensor()(Image.open(args.input1).convert('L'))
    img2 = transforms.ToTensor()(Image.open(args.input2).convert('L'))
    img3 = transforms.ToTensor()(Image.open(args.input3).convert('L'))

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    print(img2.shape[-2] * (args.scale))
    h = (round(float(img2.shape[-2] * (args.scale))))
    w = (round(float(img2.shape[-1] * (args.scale))))
    scale = h / img2.shape[-2]
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w

    cell_factor = max(scale / scale_max, 1)
    pred = batched_predict(model, ((img1 - 0.5) / 0.5).cuda().unsqueeze(0),((img2 - 0.5) / 0.5).cuda().unsqueeze(0),((img3 - 0.5) / 0.5).cuda().unsqueeze(0),
                           coord.unsqueeze(0), cell_factor * cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(args.output)