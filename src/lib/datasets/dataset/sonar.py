# src/lib/datasets/dataset/sonar.py
import os

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class SonarDataset(Dataset):
    num_classes = 1
    default_resolution = [256, 256]
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def _read_split_ids(self, split_path):
        with open(split_path, 'r') as f:
            return [int(line.strip()) for line in f if line.strip()]

    def _load_split_df(self, df, split):
        split_path = os.path.join(self.data_dir, 'splits', split + '.txt')
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                'Missing sonar split file: {}. Run '
                '`python tools/create_sonar_splits.py --data_dir {}` first.'
                .format(split_path, self.data_dir))

        split_ids = self._read_split_ids(split_path)
        df_by_id = df.set_index('sample_id', drop=False)
        missing_ids = [sample_id for sample_id in split_ids
                       if sample_id not in df_by_id.index]
        if missing_ids:
            raise ValueError(
                '{} split contains sample_id values not in labels.csv: {}'
                .format(split, missing_ids[:10]))

        return df_by_id.loc[split_ids].reset_index(drop=True)

    def __init__(self, opt, split):
        self.data_dir = os.path.join(opt.data_dir, 'sonar')
        self.mat_dir = os.path.join(self.data_dir, 'matrices')
        self.opt = opt
        self.split = split

        df = pd.read_csv(os.path.join(self.data_dir, 'labels.csv'))
        if split not in ['train', 'val', 'test']:
            raise ValueError('Unsupported sonar split: {}'.format(split))
        self.df = self._load_split_df(df, split)

        self.num_samples = len(self.df)
        print(f'[SonarDataset] {split}: {self.num_samples} samples')

        self.sig_name = {1: 'LFM', 2: 'HFM'}

    def __len__(self):
        return self.num_samples

    def _render_gaussian(self, heatmap, cx, cy, sigma=3):
        H, W = heatmap.shape
        x = np.arange(0, W)
        y = np.arange(0, H)
        xx, yy = np.meshgrid(x, y)
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        np.maximum(heatmap, g, out=heatmap)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        class_id = int(row['class_id'])
        sig_code = int(row['sig_type'])

        fname = (
            f"sample_{int(row['sample_id']):05d}_"
            f"{self.sig_name[sig_code]}_cat{class_id}.mat"
        )
        tf = sio.loadmat(
            os.path.join(self.mat_dir, fname))['tf_matrix'].astype(np.float32)

        inp = np.stack([tf, tf, tf], axis=0)

        hm_size = self.opt.output_res
        scale = hm_size / 256.0
        max_objs = self.opt.max_objs
        hm_mode = getattr(self.opt, 'hm_mode', 'mixed')
        default_hm_channels = {
            'mixed': self.num_classes,
            'keypoint3': 3,
            'endpoint2': 2
        }[hm_mode]
        hm_channels = getattr(
            self.opt, 'hm_channels',
            default_hm_channels)

        hm = np.zeros((hm_channels, hm_size, hm_size), dtype=np.float32)
        wh = np.zeros((max_objs, 2), dtype=np.float32)
        reg = np.zeros((max_objs, 2), dtype=np.float32)
        ind = np.zeros(max_objs, dtype=np.int64)
        reg_mask = np.zeros(max_objs, dtype=np.uint8)
        wh_mask = np.zeros(max_objs, dtype=np.uint8)
        kp_offset = np.zeros((max_objs, 4), dtype=np.float32)

        if class_id == 2 and row['cx'] > 0:
            s = scale

            cx_hm = np.clip(row['cx'] * s, 0, hm_size - 1)
            cy_hm = np.clip(row['cy'] * s, 0, hm_size - 1)
            x1_hm = np.clip(row['x_head'] * s, 0, hm_size - 1)
            y1_hm = np.clip(row['y_head'] * s, 0, hm_size - 1)
            x2_hm = np.clip(row['x_tail'] * s, 0, hm_size - 1)
            y2_hm = np.clip(row['y_tail'] * s, 0, hm_size - 1)

            cx_int = int(cx_hm)
            cy_int = int(cy_hm)
            x1_int = int(x1_hm)
            y1_int = int(y1_hm)
            x2_int = int(x2_hm)
            y2_int = int(y2_hm)

            dist = np.sqrt((x2_int - x1_int) ** 2 + (y2_int - y1_int) ** 2)
            sigma = 2 if dist < 18 else 3

            if hm_mode == 'keypoint3':
                hm_ids = (0, 1, 2)
            elif hm_mode == 'endpoint2':
                hm_ids = (0, 1, 1)
            else:
                hm_ids = (0, 0, 0)

            self._render_gaussian(hm[hm_ids[0]], cx_int, cy_int, sigma=sigma)
            self._render_gaussian(hm[hm_ids[1]], x1_int, y1_int, sigma=sigma)
            self._render_gaussian(hm[hm_ids[2]], x2_int, y2_int, sigma=sigma)

            pts = [
                (cx_hm, cy_hm, cx_int, cy_int),
                (x1_hm, y1_hm, x1_int, y1_int),
                (x2_hm, y2_hm, x2_int, y2_int),
            ]
            for k, (px, py, px_int, py_int) in enumerate(pts):
                reg[k] = [px - px_int, py - py_int]
                ind[k] = py_int * hm_size + px_int
                reg_mask[k] = 1

            bw = abs(row['x_tail'] - row['x_head']) * s
            bh = abs(row['y_tail'] - row['y_head']) * s
            wh[0] = [bw, bh]
            wh_mask[0] = 1
            kp_offset[0] = [
                (row['x_head'] - row['cx']) * s,
                (row['y_head'] - row['cy']) * s,
                (row['x_tail'] - row['cx']) * s,
                (row['y_tail'] - row['cy']) * s,
            ]

        ret = {
            'input': torch.from_numpy(inp),
            'hm': torch.from_numpy(hm),
            'reg_mask': torch.from_numpy(reg_mask),
            'wh_mask': torch.from_numpy(wh_mask),
            'ind': torch.from_numpy(ind),
            'wh': torch.from_numpy(wh),
            'reg': torch.from_numpy(reg),
            'kp_offset': torch.from_numpy(kp_offset),
        }
        return ret
