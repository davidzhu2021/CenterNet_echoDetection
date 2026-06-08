from __future__ import print_function

import argparse
import csv
import os
import random


def read_split_ids(path):
    with open(path, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]


def read_label_ids(path):
    with open(path, 'r', newline='') as f:
        return [int(row['sample_id']) for row in csv.DictReader(f)]


def main():
    parser = argparse.ArgumentParser(description='Check fixed sonar train/val/test splits.')
    parser.add_argument('--data_dir', default='data/sonar', help='sonar data directory')
    parser.add_argument('--seed', default=317, type=int, help='expected split seed')
    args = parser.parse_args()

    labels_path = os.path.join(args.data_dir, 'labels.csv')
    split_dir = os.path.join(args.data_dir, 'splits')
    label_ids = read_label_ids(labels_path)
    all_ids = set(label_ids)

    splits = {}
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(split_dir, split + '.txt')
        assert os.path.exists(split_path), 'missing split file: {}'.format(split_path)
        splits[split] = read_split_ids(split_path)
        assert len(splits[split]) == len(set(splits[split])), '{} split has duplicates'.format(split)
        assert set(splits[split]).issubset(all_ids), '{} split has ids not in labels.csv'.format(split)

    train_ids = set(splits['train'])
    val_ids = set(splits['val'])
    test_ids = set(splits['test'])
    assert train_ids.isdisjoint(val_ids), 'train and val overlap'
    assert train_ids.isdisjoint(test_ids), 'train and test overlap'
    assert val_ids.isdisjoint(test_ids), 'val and test overlap'
    assert train_ids | val_ids | test_ids == all_ids, 'splits do not cover labels.csv'

    n = len(label_ids)
    expected_train = int(n * 0.70)
    expected_val = int(n * 0.15)
    expected_test = n - expected_train - expected_val
    actual = {
        'train': len(splits['train']),
        'val': len(splits['val']),
        'test': len(splits['test']),
    }
    expected = {
        'train': expected_train,
        'val': expected_val,
        'test': expected_test,
    }
    assert actual == expected, 'split counts mismatch: expected {}, got {}'.format(expected, actual)

    expected_ids = list(label_ids)
    rng = random.Random(args.seed)
    rng.shuffle(expected_ids)
    expected_splits = {
        'train': expected_ids[:expected_train],
        'val': expected_ids[expected_train:expected_train + expected_val],
        'test': expected_ids[expected_train + expected_val:],
    }
    for split in ['train', 'val', 'test']:
        assert splits[split] == expected_splits[split], (
            '{} split does not match seed {}'.format(split, args.seed))

    print('Sonar splits OK')
    print('seed:', args.seed)
    print('counts:', actual)


if __name__ == '__main__':
    main()
