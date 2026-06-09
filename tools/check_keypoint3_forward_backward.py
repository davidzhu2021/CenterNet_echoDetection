from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import os
import sys

import torch
from torch.utils.data import DataLoader


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIB_DIR = os.path.join(ROOT, 'src', 'lib')
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from opts import opts
from datasets.dataset_factory import get_dataset
from models.model import create_model
from trains.ctdet import CtdetLoss


def build_options(batch_size, hm_mode, arch):
    opt = opts().parse([
        'ctdet',
        '--dataset', 'sonar',
        '--arch', arch,
        '--gpus', '-1',
        '--num_workers', '0',
        '--batch_size', str(batch_size),
        '--input_res', '256',
        '--hm_mode', hm_mode,
    ])
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    opt.device = torch.device('cpu')
    return opt, Dataset


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if hasattr(value, 'to') else value
        for key, value in batch.items()
    }


def main():
    parser = argparse.ArgumentParser(description='Check hm_mode forward/loss/backward.')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--arch', default='swin_tiny_fpn_cnnstem',
                        choices=[
                            'res_50',
                            'swin_tiny',
                            'swin_tiny_fpn',
                            'swin_tiny_fpn_cnnstem',
                            'swin_tiny_bifpn',
                            'swin_tiny_bifpn_cnnstem',
                        ])
    parser.add_argument('--hm_mode', default='endpoint2',
                        choices=['keypoint3', 'endpoint2'])
    args = parser.parse_args()

    opt, Dataset = build_options(args.batch_size, args.hm_mode, args.arch)
    dataset = Dataset(opt, 'train')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = move_batch_to_device(next(iter(loader)), opt.device)

    model = create_model(opt.arch, opt.heads, opt.head_conv).to(opt.device)
    model.train()
    criterion = CtdetLoss(opt)

    outputs = model(batch['input'].float())
    loss, loss_stats = criterion(outputs, batch)

    if not torch.isfinite(loss):
        raise RuntimeError('loss is not finite: {}'.format(loss.item()))

    model.zero_grad()
    loss.backward()

    for key in ('loss', 'hm_loss', 'wh_loss', 'off_loss'):
        value = loss_stats[key].detach().cpu().item()
        if math.isnan(value) or math.isinf(value):
            raise RuntimeError('{} is not finite: {}'.format(key, value))
        print('{}: {:.6f}'.format(key, value))


if __name__ == '__main__':
    main()
