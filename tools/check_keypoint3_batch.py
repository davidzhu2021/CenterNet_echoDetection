from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from torch.utils.data import DataLoader


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIB_DIR = os.path.join(ROOT, 'src', 'lib')
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from opts import opts
from datasets.dataset_factory import get_dataset


def shape_of(value):
    return tuple(value.shape)


def build_options(hm_mode, batch_size):
    opt = opts().parse([
        'ctdet',
        '--dataset', 'sonar',
        '--arch', 'res_50',
        '--gpus', '-1',
        '--num_workers', '0',
        '--batch_size', str(batch_size),
        '--input_res', '256',
        '--hm_mode', hm_mode,
    ])
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    return opt, Dataset


def check_mode(hm_mode, batch_size):
    opt, Dataset = build_options(hm_mode, batch_size)
    dataset = Dataset(opt, 'train')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    print('[hm_mode={}]'.format(hm_mode))
    print('  heads:', opt.heads)
    for key in ('input', 'hm', 'wh', 'reg', 'reg_mask', 'wh_mask', 'ind'):
        print('  {} shape: {}'.format(key, shape_of(batch[key])))


def main():
    parser = argparse.ArgumentParser(description='Check sonar keypoint heatmap batch shapes.')
    parser.add_argument('--hm_mode', default='mixed', choices=['mixed', 'keypoint3', 'both'])
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    modes = ['mixed', 'keypoint3'] if args.hm_mode == 'both' else [args.hm_mode]
    for mode in modes:
        check_mode(mode, args.batch_size)


if __name__ == '__main__':
    main()
