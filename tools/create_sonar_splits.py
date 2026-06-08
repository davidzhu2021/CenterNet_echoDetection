from __future__ import print_function

import argparse
import csv
import os
import random


def read_sample_ids(labels_path):
    with open(labels_path, 'r', newline='') as f:
        ids = [int(row['sample_id']) for row in csv.DictReader(f)]
    if len(ids) != len(set(ids)):
        raise ValueError('labels.csv contains duplicate sample_id values')
    return ids


def write_split(path, sample_ids):
    with open(path, 'w') as f:
        for sample_id in sample_ids:
            f.write('{}\n'.format(sample_id))


def main():
    parser = argparse.ArgumentParser(description='Create fixed sonar train/val/test splits.')
    parser.add_argument('--data_dir', default='data/sonar', help='sonar data directory')
    parser.add_argument('--seed', default=317, type=int, help='fixed random seed')
    parser.add_argument('--train_ratio', default=0.70, type=float, help='train split ratio')
    parser.add_argument('--val_ratio', default=0.15, type=float, help='validation split ratio')
    args = parser.parse_args()

    labels_path = os.path.join(args.data_dir, 'labels.csv')
    split_dir = os.path.join(args.data_dir, 'splits')
    os.makedirs(split_dir, exist_ok=True)

    sample_ids = read_sample_ids(labels_path)
    rng = random.Random(args.seed)
    rng.shuffle(sample_ids)

    n = len(sample_ids)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    splits = {
        'train': sample_ids[:n_train],
        'val': sample_ids[n_train:n_train + n_val],
        'test': sample_ids[n_train + n_val:],
    }

    for split, ids in splits.items():
        write_split(os.path.join(split_dir, split + '.txt'), ids)

    print('Created sonar splits in {}'.format(split_dir))
    print('seed:', args.seed)
    print('train:', len(splits['train']))
    print('val:', len(splits['val']))
    print('test:', len(splits['test']))


if __name__ == '__main__':
    main()
