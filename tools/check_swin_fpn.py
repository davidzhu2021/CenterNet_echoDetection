import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIB_DIR = os.path.join(ROOT, 'src', 'lib')
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from models.model import create_model


def fmt_shape(tensor):
    return tuple(tensor.shape)


def main():
    heads = {'hm': 1, 'wh': 2, 'reg': 2}
    x = torch.randn(2, 3, 256, 256)

    for arch in ('swin_tiny_fpn', 'swin_tiny_fpn_cnnstem'):
        print(f'[{arch}]')
        model = create_model(arch, heads=heads, head_conv=64)
        model.eval()

        with torch.no_grad():
            outputs, debug = model(x, return_debug=True)

        print('  [Swin stages]')
        for name, tensor in debug['stages'].items():
            print(f'    {name}: {fmt_shape(tensor)}')

        print('  [FPN outputs]')
        for name, tensor in debug['fpn'].items():
            print(f'    {name}: {fmt_shape(tensor)}')

        print('  [Heads]')
        ret = outputs[-1]
        for head in ('hm', 'wh', 'reg'):
            print(f'    {head}: {fmt_shape(ret[head])}')


if __name__ == '__main__':
    main()
