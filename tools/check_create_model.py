import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIB_DIR = os.path.join(ROOT, 'src', 'lib')
sys.path.insert(0, LIB_DIR)

from models.model import create_model  # noqa: E402


def check_model(arch):
    heads = {'hm': 1, 'wh': 2, 'reg': 2}
    model = create_model(arch, heads=heads, head_conv=64)
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        output = model(x)[-1]

    print('[{}]'.format(arch))
    for head in ('hm', 'wh', 'reg'):
        print('  {}: {}'.format(head, tuple(output[head].shape)))


def main():
    for arch in ('res_50', 'swin_tiny'):
        check_model(arch)


if __name__ == '__main__':
    main()
