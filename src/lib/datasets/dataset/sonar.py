# src/lib/datasets/dataset/sonar.py
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import pandas as pd
import os
import cv2


class SonarDataset(Dataset):
    num_classes = 1          # 只有target一类（reverb作为背景处理）
    default_resolution = [256, 256]
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def __init__(self, opt, split):
        self.data_dir = os.path.join(opt.data_dir, 'sonar')
        self.mat_dir  = os.path.join(self.data_dir, 'matrices')
        self.opt      = opt
        self.split    = split

        # 读取CSV
        df = pd.read_csv(os.path.join(self.data_dir, 'labels.csv'))

        # 按8:2划分train/val
        np.random.seed(42)
        idx     = np.random.permutation(len(df))
        n_train = int(len(df) * 0.8)
        if split == 'train':
            self.df = df.iloc[idx[:n_train]].reset_index(drop=True)
        else:
            self.df = df.iloc[idx[n_train:]].reset_index(drop=True)

        self.num_samples = len(self.df)
        print(f'[SonarDataset] {split}: {self.num_samples} samples')

        # sig_type映射
        self.sig_name = {1: 'LFM', 2: 'HFM'}

    def __len__(self):
        return self.num_samples

    def _render_gaussian(self, heatmap, cx, cy, sigma=3):
        H, W = heatmap.shape
        x = np.arange(0, W)
        y = np.arange(0, H)
        xx, yy = np.meshgrid(x, y)
        g = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        np.maximum(heatmap, g, out=heatmap)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        class_id = int(row['class_id'])
        sig_code = int(row['sig_type'])

        # --- 读取时频矩阵 ---
        fname = (f"sample_{int(row['sample_id']):05d}_"
                 f"{self.sig_name[sig_code]}_cat{class_id}.mat")
        tf = sio.loadmat(
            os.path.join(self.mat_dir, fname))['tf_matrix'].astype(np.float32)


        # 复制三通道 [3, 256, 256]
        inp = np.stack([tf, tf, tf], axis=0)

        # --- 构造官方ctdet格式的GT ---
        hm_size  = self.opt.output_res   # 默认64
        scale    = hm_size / 256.0       # 0.25
        max_objs = self.opt.max_objs     # 默认128

        hm       = np.zeros((self.num_classes, hm_size, hm_size), dtype=np.float32)
        wh       = np.zeros((max_objs, 2),  dtype=np.float32)
        reg      = np.zeros((max_objs, 2),  dtype=np.float32)
        ind      = np.zeros(max_objs,       dtype=np.int64)
        reg_mask = np.zeros(max_objs,       dtype=np.uint8)

        # 额外：头尾偏移（你的自定义输出）
        kp_offset = np.zeros((max_objs, 4), dtype=np.float32)

        num_objs = 0
        if class_id == 2 and row['cx'] > 0:
            cx_hm = np.clip(row['cx'] * scale, 0, hm_size - 1)
            cy_hm = np.clip(row['cy'] * scale, 0, hm_size - 1)
            cx_int = int(cx_hm)
            cy_int = int(cy_hm)

            self._render_gaussian(hm[0], cx_int, cy_int, sigma=3)

            # wh：用头尾点的像素跨度作为"宽高"
            bw = abs(row['x_tail'] - row['x_head']) * scale
            bh = abs(row['y_tail'] - row['y_head']) * scale
            wh[0]  = [bw, bh]

            # reg：中心点在heatmap上的小数偏移（subpixel）
            reg[0] = [cx_hm - cx_int, cy_hm - cy_int]

            # ind：中心点在heatmap展平后的索引
            ind[0] = cy_int * hm_size + cx_int

            reg_mask[0] = 1
            num_objs    = 1

            # 头尾偏移
            kp_offset[0] = [
                (row['x_head'] - row['cx']) * scale,
                (row['y_head'] - row['cy']) * scale,
                (row['x_tail'] - row['cx']) * scale,
                (row['y_tail'] - row['cy']) * scale,
            ]

        ret = {
            'input':      torch.from_numpy(inp),
            'hm':         torch.from_numpy(hm),
            'reg_mask':   torch.from_numpy(reg_mask),
            'ind':        torch.from_numpy(ind),
            'wh':         torch.from_numpy(wh),
            'reg':        torch.from_numpy(reg),
            'kp_offset':  torch.from_numpy(kp_offset),
        }
        return ret