import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
sys.path.insert(0, r'E:\CenterNet\src\lib')

import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from models.model import create_model, load_model

# --- 加载最优模型 ---
model = create_model('res_50',
                     heads={'hm': 1, 'wh': 2, 'reg': 2},
                     head_conv=64)
model = load_model(model,
                   r'E:\CenterNet\exp\ctdet\model_best.pth')
model.eval()

# --- 输入/输出路径 ---
input_dir = r'E:\sonar_data_noaa\test_dataset_targets_only\matrices'
output_dir = r'E:\sonar_data_noaa\test_dataset_targets_only\inference_results'
os.makedirs(output_dir, exist_ok=True)

# --- 遍历所有 .mat 文件 ---
mat_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mat')])
print(f'共找到 {len(mat_files)} 个 .mat 文件，开始推理...\n')

for idx, fname in enumerate(mat_files):
    mat_path = os.path.join(input_dir, fname)
    base_name = os.path.splitext(fname)[0]  # 去掉 .mat 后缀

    # 读取时频矩阵
    tf = sio.loadmat(mat_path)['tf_matrix'].astype(np.float32)
    tf_norm = (tf - 0.5) / 0.5
    inp = torch.from_numpy(
        np.stack([tf_norm, tf_norm, tf_norm], axis=0)
    ).unsqueeze(0)  # [1, 3, H, W]

    # 推理
    with torch.no_grad():
        output = model(inp)[-1]
        hm = output['hm'].sigmoid().squeeze().numpy()  # [64, 64]

    # 找最高置信度点
    cy, cx = np.unravel_index(hm.argmax(), hm.shape)
    max_score = hm.max()
    mean_score = hm.mean()

    # 绘图：上图时频矩阵，下图热图
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))

    axes[0].imshow(tf, aspect='auto', origin='upper', cmap='jet')
    axes[0].set_title(f'{base_name}', fontsize=9)
    axes[0].set_ylabel('TF Matrix', fontsize=9)
    axes[0].axis('off')

    axes[1].imshow(hm, aspect='auto', origin='upper', cmap='hot')
    axes[1].plot(cx, cy, 'b+', markersize=12, linewidth=2)
    axes[1].set_title(f'max score: {max_score:.4f}  mean: {mean_score:.6f}\npeak@({cx*4}, {cy*4})', fontsize=9)
    axes[1].set_ylabel('Heatmap', fontsize=9)
    axes[1].axis('off')

    plt.tight_layout()

    # 保存，文件名与 .mat 同名
    save_path = os.path.join(output_dir, f'{base_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # 释放内存，避免大量文件时 OOM

    print(f'[{idx+1:>4}/{len(mat_files)}] {fname}  max={max_score:.4f}  mean={mean_score:.6f}  -> 已保存')

print(f'\n全部完成！结果保存在: {output_dir}')