# dataset.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决Windows下OpenMP冲突
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows上用微软雅黑
plt.rcParams['axes.unicode_minus'] = False
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import pandas as pd
import os

def render_gaussian_heatmap(size, cx, cy, sigma=3):
    """在(cx,cy)处生成高斯热力图"""
    x = np.arange(0, size)
    y = np.arange(0, size)
    xx, yy = np.meshgrid(x, y)
    heatmap = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    return heatmap.astype(np.float32)

class SonarDataset(Dataset):
    def __init__(self, csv_path, mat_dir, heatmap_size=64):
        self.df           = pd.read_csv(csv_path)
        self.mat_dir      = mat_dir
        self.heatmap_size = heatmap_size       # 输入256，输出64（stride=4）
        self.input_size   = 256
        self.scale        = heatmap_size / self.input_size  # 0.25

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        class_id = int(row['class_id'])  # 1=reverb, 2=target

        #--- 读取时频矩阵 ---
        fname  = f"sample_{int(row['sample_id']):04d}_cat{class_id}.mat"
        tf     = sio.loadmat(os.path.join(self.mat_dir, fname))['tf_matrix']
        tf     = tf.astype(np.float32)           # [256, 256]

        # 复制三通道，兼容ImageNet预训练权重
        # 后续换成STFT+SPWVD+CWT三通道时改这里即可
        img = np.stack([tf, tf, tf], axis=0)     # [3, 256, 256]

        #--- 构造GT ---
        hm_size = self.heatmap_size
        # 中心点热力图 [1, 64, 64]
        heatmap = np.zeros((1, hm_size, hm_size), dtype=np.float32)
        # 头尾偏移图 [4, 64, 64]：dx_head, dy_head, dx_tail, dy_tail
        kp_map  = np.zeros((4, hm_size, hm_size), dtype=np.float32)

        if class_id == 2 and row['cx'] > 0:
            s = self.scale

            # 缩放到heatmap尺寸
            cx_hm = int(np.clip(row['cx']     * s, 0, hm_size - 1))
            cy_hm = int(np.clip(row['cy']     * s, 0, hm_size - 1))

            # 渲染高斯热力图
            heatmap[0] = render_gaussian_heatmap(hm_size, cx_hm, cy_hm, sigma=3)

            # 头尾偏移（在heatmap尺度下）
            kp_map[0, cy_hm, cx_hm] = (row['x_head'] - row['cx']) * s  # dx_head
            kp_map[1, cy_hm, cx_hm] = (row['y_head'] - row['cy']) * s  # dy_head
            kp_map[2, cy_hm, cx_hm] = (row['x_tail'] - row['cx']) * s  # dx_tail
            kp_map[3, cy_hm, cx_hm] = (row['y_tail'] - row['cy']) * s  # dy_tail

        return {
            'img':      torch.from_numpy(img),              # [3, 256, 256]
            'heatmap':  torch.from_numpy(heatmap),          # [1,  64,  64]
            'kp_map':   torch.from_numpy(kp_map),           # [4,  64,  64]
            'class_id': torch.tensor(class_id - 1),         # 0=reverb, 1=target
        }

    # 在dataset.py末尾加上这段，直接运行验证
if __name__ == '__main__':
        import matplotlib.pyplot as plt

        ds = SonarDataset(
            csv_path=r'D:\gradu\simulation_5\tf_dataset_matrix\labels.csv',
            mat_dir=r'D:\gradu\simulation_5\tf_dataset_matrix\matrices'
        )
        print(f'数据集大小: {len(ds)}')

        # 找一个target样本
        for i in range(len(ds)):
            sample = ds[i]
            if sample['class_id'] == 1:  # target
                break

        print('img shape:    ', sample['img'].shape)  # [3, 256, 256]
        print('heatmap shape:', sample['heatmap'].shape)  # [1,  64,  64]
        print('kp_map shape: ', sample['kp_map'].shape)  # [4,  64,  64]
        print('heatmap max:  ', sample['heatmap'].max())  # 应该接近1.0

        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(sample['img'][0], cmap='jet', origin='upper')
        axes[0].set_title('时频矩阵')
        axes[1].imshow(sample['heatmap'][0], cmap='hot', origin='upper')
        axes[1].set_title('GT Heatmap')
        plt.tight_layout()
        plt.show()