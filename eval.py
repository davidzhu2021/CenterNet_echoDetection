import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
sys.path.insert(0, r'E:\CenterNet\src\lib')

import torch
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from models.model import create_model, load_model

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
MODEL_PATH = r'E:\CenterNet\exp\ctdet\model_best.pth'
CSV_PATH   = r'D:\gradu\simulation_5\tf_dataset\labels.csv'
MAT_DIR    = r'D:\gradu\simulation_5\tf_dataset\matrices'
THRESHOLD  = 0.3     # 置信度阈值
SIG_NAME   = {1: 'LFM', 2: 'HFM'}

# ─────────────────────────────────────────────
# 加载模型
# ─────────────────────────────────────────────
model = create_model('res_50',
                     heads={'hm': 1, 'wh': 2, 'reg': 2},
                     head_conv=64)
model = load_model(model, MODEL_PATH)
model.eval()

# ─────────────────────────────────────────────
# 加载验证集（与训练时相同的划分）
# ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
# np.random.seed(42)
# idx    = np.random.permutation(len(df))
# n_val  = int(len(df) * 0.2)
# val_df = df.iloc[idx[-n_val:]].reset_index(drop=True)
# print(f'验证集样本数: {len(val_df)}')
val_df = df.reset_index(drop=True)
print(f'测试集样本数: {len(val_df)}')

# ─────────────────────────────────────────────
# 逐样本推理
# ─────────────────────────────────────────────
results = []

for i, (_, row) in enumerate(val_df.iterrows()):
    class_id = int(row['class_id'])
    sig_code = int(row['sig_type'])

    fname = (f"sample_{int(row['sample_id']):05d}_"
             f"{SIG_NAME[sig_code]}_cat{class_id}.mat")
    mat_path = os.path.join(MAT_DIR, fname)

    if not os.path.exists(mat_path):
        print(f'[跳过] 文件不存在: {fname}')
        continue

    tf      = sio.loadmat(mat_path)['tf_matrix'].astype(np.float32)
    tf_norm = (tf - 0.5) / 0.5
    inp     = torch.from_numpy(
                  np.stack([tf_norm, tf_norm, tf_norm], axis=0)
              ).unsqueeze(0)

    with torch.no_grad():
        output = model(inp)[-1]
        hm     = output['hm'].sigmoid().squeeze().numpy()

    max_score = hm.max()
    cy, cx    = np.unravel_index(hm.argmax(), hm.shape)

    # 分类预测
    pred_class = 2 if max_score >= THRESHOLD else 1

    # 定位误差（只对target样本计算）
    loc_error = -1
    if class_id == 2 and row['cx'] > 0:
        pred_cx   = cx * 4
        pred_cy   = cy * 4
        loc_error = float(np.sqrt((pred_cx - row['cx'])**2 +
                                  (pred_cy - row['cy'])**2))

    results.append({
        'sample_id' : int(row['sample_id']),
        'sig_type'  : sig_code,
        'gt_class'  : class_id,
        'pred_class': pred_class,
        'max_score' : max_score,
        'loc_error' : loc_error,
    })

    if (i + 1) % 100 == 0:
        print(f'推理进度: {i+1}/{len(val_df)}')

res_df = pd.DataFrame(results)

# ─────────────────────────────────────────────
# 计算指标
# ─────────────────────────────────────────────
def calc_metrics(df, name='全部'):
    TP = ((df.gt_class == 2) & (df.pred_class == 2)).sum()
    FP = ((df.gt_class == 1) & (df.pred_class == 2)).sum()
    FN = ((df.gt_class == 2) & (df.pred_class == 1)).sum()
    TN = ((df.gt_class == 1) & (df.pred_class == 1)).sum()

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (TP + TN) / len(df)

    target_res = df[(df.gt_class == 2) & (df.loc_error >= 0)]
    mean_loc   = target_res['loc_error'].mean()

    print(f'\n{"="*45}')
    print(f'【{name}】样本数={len(df)}')
    print(f'  TP={TP}  FP={FP}  FN={FN}  TN={TN}')
    print(f'  Accuracy : {accuracy:.4f}')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall   : {recall:.4f}')
    print(f'  F1 Score : {f1:.4f}')
    print(f'  定位误差  : {mean_loc:.2f} px')

    return {'name': name, 'accuracy': accuracy,
            'precision': precision, 'recall': recall,
            'f1': f1, 'loc_error': mean_loc}

# 整体指标
calc_metrics(res_df, '整体验证集')

# 按信号类型分别统计
for sig_code, sig_name in SIG_NAME.items():
    sub = res_df[res_df.sig_type == sig_code]
    if len(sub) > 0:
        calc_metrics(sub, sig_name)

# ─────────────────────────────────────────────
# 更适合汇报的可视化结果
# ─────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SAVE_DIR = 'eval_figures'
os.makedirs(SAVE_DIR, exist_ok=True)

# 为了避免中文乱码，这里图标题和坐标轴尽量用英文
# 如果你想用中文，可以取消下面两行注释
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False


# =========================================================
# 1. 阈值敏感性：Precision / Recall / F1 / Accuracy vs Threshold
# =========================================================
print(f'\n\n{"阈值敏感性分析":=^45}')
print(f'{"阈值":>8} {"Precision":>10} {"Recall":>10} {"F1":>10} {"Accuracy":>10}')

curve_data = []

for thresh in np.arange(0.1, 1.0, 0.1):
    pred = (res_df['max_score'] >= thresh).astype(int) + 1

    TP = ((res_df.gt_class == 2) & (pred == 2)).sum()
    FP = ((res_df.gt_class == 1) & (pred == 2)).sum()
    FN = ((res_df.gt_class == 2) & (pred == 1)).sum()
    TN = ((res_df.gt_class == 1) & (pred == 1)).sum()

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (TP + TN) / (TP + FP + FN + TN + 1e-8)

    print(f'{thresh:>8.1f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {accuracy:>10.4f}')

    curve_data.append({
        'threshold': thresh,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': accuracy,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN
    })

curve_df = pd.DataFrame(curve_data)
curve_df.to_csv(os.path.join(SAVE_DIR, 'threshold_metrics.csv'), index=False)

best_idx = curve_df['F1'].idxmax()
best_t = curve_df.loc[best_idx, 'threshold']
best_f1 = curve_df.loc[best_idx, 'F1']

plt.figure(figsize=(8, 5))
plt.plot(curve_df['threshold'], curve_df['Precision'], marker='o', label='Precision')
plt.plot(curve_df['threshold'], curve_df['Recall'], marker='o', label='Recall')
plt.plot(curve_df['threshold'], curve_df['F1'], marker='o', label='F1')
plt.plot(curve_df['threshold'], curve_df['Accuracy'], marker='o', label='Accuracy')

plt.axvline(best_t, linestyle='--', alpha=0.7)
plt.text(best_t + 0.015, best_f1, f'Best F1={best_f1:.3f}\nT={best_t:.1f}', fontsize=10)

plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.title('Metrics vs Threshold')
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'metrics_vs_threshold.png'), dpi=150)
plt.close()


# =========================================================
# 2. 混淆矩阵：Overall / LFM / HFM
# =========================================================
def plot_confusion_matrix(df, title, save_name):
    TP = ((df.gt_class == 2) & (df.pred_class == 2)).sum()
    FP = ((df.gt_class == 1) & (df.pred_class == 2)).sum()
    FN = ((df.gt_class == 2) & (df.pred_class == 1)).sum()
    TN = ((df.gt_class == 1) & (df.pred_class == 1)).sum()

    cm = np.array([
        [TN, FP],
        [FN, TP]
    ])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Negative', 'Pred Positive'])
    ax.set_yticklabels(['GT Negative', 'GT Positive'])
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground Truth')
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, save_name), dpi=150)
    plt.close()


plot_confusion_matrix(res_df, 'Confusion Matrix - Overall', 'confusion_matrix_overall.png')

for sig_code, sig_name in SIG_NAME.items():
    sub_df = res_df[res_df.sig_type == sig_code]
    if len(sub_df) > 0:
        plot_confusion_matrix(
            sub_df,
            f'Confusion Matrix - {sig_name}',
            f'confusion_matrix_{sig_name}.png'
        )


# =========================================================
# 3. 正负样本得分分布图
# =========================================================
pos_scores = res_df[res_df.gt_class == 2]['max_score'].values
neg_scores = res_df[res_df.gt_class == 1]['max_score'].values

plt.figure(figsize=(8, 5))
plt.hist(pos_scores, bins=30, alpha=0.6, label='Positive Samples')
plt.hist(neg_scores, bins=30, alpha=0.6, label='Negative Samples')
plt.axvline(THRESHOLD, linestyle='--', label=f'Threshold = {THRESHOLD}')

plt.xlabel('Max Heatmap Score')
plt.ylabel('Number of Samples')
plt.title('Score Distribution')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'score_distribution.png'), dpi=150)
plt.close()


# =========================================================
# 4. LFM / HFM 指标柱状图
# =========================================================
def get_metrics_for_bar(df):
    TP = ((df.gt_class == 2) & (df.pred_class == 2)).sum()
    FP = ((df.gt_class == 1) & (df.pred_class == 2)).sum()
    FN = ((df.gt_class == 2) & (df.pred_class == 1)).sum()
    TN = ((df.gt_class == 1) & (df.pred_class == 1)).sum()

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (TP + TN) / (TP + FP + FN + TN + 1e-8)

    return accuracy, precision, recall, f1


bar_names = []
bar_values = []

for sig_code, sig_name in SIG_NAME.items():
    sub_df = res_df[res_df.sig_type == sig_code]
    if len(sub_df) > 0:
        bar_names.append(sig_name)
        bar_values.append(get_metrics_for_bar(sub_df))

if len(bar_values) >= 2:
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    bar_values = np.array(bar_values)

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))

    for i, name in enumerate(bar_names):
        offset = (i - (len(bar_names) - 1) / 2) * width
        plt.bar(x + offset, bar_values[i], width, label=name)

    plt.xticks(x, metrics)
    plt.ylim(0, 1.05)
    plt.ylabel('Metric Value')
    plt.title('Performance Comparison: LFM vs HFM')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'lfm_hfm_metrics_bar.png'), dpi=150)
    plt.close()


# =========================================================
# 5. 定位误差分布：只统计 TP 样本
# =========================================================
tp_df = res_df[
    (res_df.gt_class == 2) &
    (res_df.pred_class == 2) &
    (res_df.loc_error >= 0)
].copy()

if len(tp_df) > 0:
    # Overall 定位误差直方图
    plt.figure(figsize=(8, 5))
    plt.hist(tp_df['loc_error'].values, bins=30, alpha=0.75)
    plt.xlabel('Localization Error (px)')
    plt.ylabel('Number of TP Samples')
    plt.title('Localization Error Distribution - TP Samples')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'loc_error_distribution_tp.png'), dpi=150)
    plt.close()

    # LFM / HFM 箱线图
    loc_data = []
    loc_labels = []

    for sig_code, sig_name in SIG_NAME.items():
        sub_err = tp_df[tp_df.sig_type == sig_code]['loc_error'].values
        if len(sub_err) > 0:
            loc_data.append(sub_err)
            loc_labels.append(sig_name)

    if len(loc_data) > 0:
        plt.figure(figsize=(6, 5))
        plt.boxplot(loc_data, labels=loc_labels)
        plt.ylabel('Localization Error (px)')
        plt.title('Localization Error Boxplot - TP Samples')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'loc_error_boxplot_tp.png'), dpi=150)
        plt.close()

    # LFM / HFM CDF 曲线
    def plot_cdf(data, label):
        data = np.sort(data)
        y = np.arange(1, len(data) + 1) / len(data)
        plt.plot(data, y, label=label)

    plt.figure(figsize=(7, 5))

    has_curve = False
    for sig_code, sig_name in SIG_NAME.items():
        sub_err = tp_df[tp_df.sig_type == sig_code]['loc_error'].values
        if len(sub_err) > 0:
            plot_cdf(sub_err, sig_name)
            has_curve = True

    if has_curve:
        plt.xlabel('Localization Error (px)')
        plt.ylabel('Cumulative Probability')
        plt.title('CDF of Localization Error - TP Samples')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'loc_error_cdf_tp.png'), dpi=150)

    plt.close()


# =========================================================
# 6. 保存每个样本的预测结果，方便后面查 FN/TP 样本
# =========================================================
res_df.to_csv(os.path.join(SAVE_DIR, 'eval_sample_results.csv'), index=False)

# 单独保存 FN 样本，方便你后面可视化漏检
fn_df = res_df[(res_df.gt_class == 2) & (res_df.pred_class == 1)]
fn_df.to_csv(os.path.join(SAVE_DIR, 'fn_samples.csv'), index=False)

tp_df.to_csv(os.path.join(SAVE_DIR, 'tp_samples.csv'), index=False)

print(f'\n所有图和结果表已保存到文件夹: {SAVE_DIR}')
print('生成文件包括：')
print('  1. metrics_vs_threshold.png')
print('  2. confusion_matrix_overall.png')
print('  3. confusion_matrix_LFM.png')
print('  4. confusion_matrix_HFM.png')
print('  5. score_distribution.png')
print('  6. lfm_hfm_metrics_bar.png')
print('  7. loc_error_distribution_tp.png')
print('  8. loc_error_boxplot_tp.png')
print('  9. loc_error_cdf_tp.png')
print('  10. eval_sample_results.csv')
print('  11. fn_samples.csv')
print('  12. tp_samples.csv')