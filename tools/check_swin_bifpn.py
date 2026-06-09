import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIB_DIR = os.path.join(ROOT, 'src', 'lib')
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from models.model import create_model


HM_CHANNELS = {
    'mixed': 1,
    'keypoint3': 3,
    'endpoint2': 2,
}


def fmt_shape(tensor):
    return tuple(tensor.shape)


def main():
    x = torch.randn(2, 3, 256, 256)

    for arch in ('swin_tiny_bifpn', 'swin_tiny_bifpn_cnnstem'):
        for hm_mode, hm_channels in HM_CHANNELS.items():
            heads = {'hm': hm_channels, 'wh': 2, 'reg': 2}
            print(f'[{arch}] hm_mode={hm_mode}')
            model = create_model(arch, heads=heads, head_conv=64)
            model.eval()

            with torch.no_grad():
                outputs, debug = model(x, return_debug=True)

            print('  [Backbone stages]')
            for name, tensor in debug['stages'].items():
                print(f'    {name}: {fmt_shape(tensor)}')

            print('  [BiFPN outputs]')
            for name, tensor in debug['bifpn'].items():
                print(f'    {name}: {fmt_shape(tensor)}')

            print('  [Heads]')
            ret = outputs[-1]
            for head in ('hm', 'wh', 'reg'):
                print(f'    {head}: {fmt_shape(ret[head])}')


if __name__ == '__main__':
    main()
