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
                   r'E:\CenterNet\exp\ctdet\sonar_exp1\model_best.pth')
model.eval()

# --- 输入/输出路径 ---
input_dir  = r'D:\gradu\simulation_5\tf_dataset\matrices'   # 改成你实际的测试集路径
output_dir = r'E:\CenterNet\inference_results'
os.makedirs(output_dir, exist_ok=True)

# --- 遍历所有 .mat 文件 ---
mat_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mat')])
print(f'共找到 {len(mat_files)} 个 .mat 文件，开始推理...\n')

for idx, fname in enumerate(mat_files):
    mat_path  = os.path.join(input_dir, fname)
    base_name = os.path.splitext(fname)[0]

    # 读取时频矩阵
    tf = sio.loadmat(mat_path)['tf_matrix'].astype(np.float32)
    tf_norm = (tf - 0.5) / 0.5
    inp = torch.from_numpy(
        np.stack([tf_norm, tf_norm, tf_norm], axis=0)
    ).unsqueeze(0)

    # 推理
    with torch.no_grad():
        output    = model(inp)[-1]
        hm        = output['hm'].sigmoid().squeeze().numpy()  # [64, 64]

    # --- top-3解码（中心点+头部端点+尾部端点）---
    flat      = hm.flatten()
    H, W      = hm.shape

    # MaxPool NMS：只保留局部极大值
    hm_tensor = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)
    hm_max    = torch.nn.functional.max_pool2d(
                    hm_tensor, kernel_size=3, stride=1, padding=1)
    keep      = (hm_tensor == hm_max).squeeze().numpy()
    hm_nms    = hm * keep

    flat_nms  = hm_nms.flatten()
    top3_idx  = np.argsort(flat_nms)[-3:][::-1]   # 置信度从高到低

    pts = []
    for i in top3_idx:
        score = flat_nms[i]
        if score < 0.1:   # 过滤极低置信度的点
            continue
        py = i // W
        px = i  % W
        pts.append({'x': px, 'y': py, 'score': score,
                    'x_orig': px * 4, 'y_orig': py * 4})

    # 按y坐标排序：y小=高频端点，y大=低频端点，中间=中心点
    pts_sorted = sorted(pts, key=lambda p: p['y'])

    max_score  = hm.max()
    mean_score = hm.mean()

    # --- 绘图 ---
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))

    axes[0].imshow(tf, aspect='auto', origin='upper', cmap='jet')
    axes[0].set_title(f'{base_name}', fontsize=9)
    axes[0].set_ylabel('TF Matrix', fontsize=9)
    axes[0].axis('off')

    axes[1].imshow(hm, aspect='auto', origin='upper', cmap='hot')

    # 画三个点，用不同颜色区分
    colors  = ['cyan', 'white', 'yellow']   # 高频端点/中心点/低频端点
    labels  = ['p1(high freq)', 'center', 'p2(low freq)']
    for j, pt in enumerate(pts_sorted[:3]):
        c = colors[j] if j < len(colors) else 'white'
        l = labels[j] if j < len(labels) else f'pt{j}'
        axes[1].plot(pt['x'], pt['y'], '+',
                     color=c, markersize=12, linewidth=2, label=l)

    title_str = f'max score: {max_score:.4f}  mean: {mean_score:.6f}\n'
    for j, pt in enumerate(pts_sorted[:3]):
        title_str += f'{labels[j] if j<len(labels) else f"pt{j}"}: ({pt["x_orig"]},{pt["y_orig"]}) s={pt["score"]:.3f}  '
    axes[1].set_title(title_str, fontsize=8)
    axes[1].set_ylabel('Heatmap', fontsize=9)
    axes[1].legend(fontsize=7, loc='upper right')
    axes[1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{base_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 打印结果
    print(f'[{idx+1:>4}/{len(mat_files)}] {fname}  max={max_score:.4f}', end='  ')
    for j, pt in enumerate(pts_sorted[:3]):
        print(f'pt{j+1}=({pt["x_orig"]},{pt["y_orig"]})', end=' ')
    print()

print(f'\n全部完成！结果保存在: {output_dir}')