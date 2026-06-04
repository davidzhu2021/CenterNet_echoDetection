import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import sys
sys.path.insert(0, r'E:\CenterNet\src\lib')

import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

from models.model import create_model


# =========================================================
# 配置
# =========================================================
MODEL_PATH = r'E:\CenterNet\exp\ctdet\model_best.pth'
CSV_PATH   = r'D:\gradu\simulation_5\tf_dataset\labels.csv'
MAT_DIR    = r'D:\gradu\simulation_5\tf_dataset\matrices'

SAVE_DIR = 'eval_full_results'

SIG_NAME = {1: 'LFM', 2: 'HFM'}

# 你的 CenterNet 输出一般是输入的 1/4
DOWN_RATIO = 4

# 当前使用阈值
THRESHOLD = 0.3

# decode 时每张图最多保留多少个候选检测结果
TOPK = 20

# 如果 labels.csv 里有 bbox，就用 IoU 匹配
IOU_THRESH = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate sonar CenterNet checkpoints.')
    parser.add_argument('--arch', default='res_50', help='model architecture, e.g. res_50, swin_tiny')
    parser.add_argument('--model_path', default=MODEL_PATH, help='checkpoint path')
    parser.add_argument('--csv_path', default=CSV_PATH, help='labels.csv path')
    parser.add_argument('--mat_dir', default=MAT_DIR, help='directory containing .mat STFT matrices')
    parser.add_argument('--save_dir', default=SAVE_DIR, help='directory to save evaluation results')
    parser.add_argument('--input_res', default=256, type=int, help='input resolution used by the model')
    parser.add_argument('--score_thresh', default=THRESHOLD, type=float, help='score threshold for evaluation')
    parser.add_argument(
        '--device',
        default='auto',
        help='device for inference: auto, cpu, cuda, cuda:0, etc.'
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def load_eval_checkpoint(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.') and not key.startswith('module_list'):
            cleaned_state_dict[key[7:]] = value
        else:
            cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    return model

# 如果 labels.csv 里没有 bbox，只有 cx/cy，就用中心点距离匹配
CENTER_TOL = 40.0  # 单位：px，可根据任务改成 30、40、50


# =========================================================
# 加载模型
# =========================================================
args = parse_args()
MODEL_PATH = args.model_path
CSV_PATH = args.csv_path
MAT_DIR = args.mat_dir
SAVE_DIR = args.save_dir
DOWN_RATIO = args.input_res // 64
THRESHOLD = args.score_thresh

os.makedirs(SAVE_DIR, exist_ok=True)

print('Evaluation config:')
print(f'  arch: {args.arch}')
print(f'  model_path: {MODEL_PATH}')
print(f'  csv_path: {CSV_PATH}')
print(f'  mat_dir: {MAT_DIR}')
print(f'  save_dir: {SAVE_DIR}')

device = resolve_device(args.device)
print('Using device:', device)

model = create_model(
    args.arch,
    heads={'hm': 1, 'wh': 2, 'reg': 2},
    head_conv=64
)

model = load_eval_checkpoint(model, MODEL_PATH, device)
model = model.to(device)
model.eval()


# =========================================================
# 工具函数：CenterNet decode
# =========================================================
def nms_heatmap(hm, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(hm, kernel, stride=1, padding=pad)
    keep = (hmax == hm).float()
    return hm * keep


def decode_centernet_output(output, K=20, down_ratio=4):
    """
    将 CenterNet 的 hm / wh / reg 解码成检测框。

    返回：
    [
        {
            'score': float,
            'cx': float,
            'cy': float,
            'x1': float,
            'y1': float,
            'x2': float,
            'y2': float,
        },
        ...
    ]
    """
    hm = output['hm'].sigmoid()
    wh = output['wh']
    reg = output.get('reg', None)

    hm = nms_heatmap(hm)

    B, C, H, W = hm.shape
    assert B == 1, '当前 decode 函数按 batch=1 写的'

    scores, inds = torch.topk(hm.view(-1), K)

    clses = inds // (H * W)
    inds_spatial = inds % (H * W)
    ys = (inds_spatial // W).long()
    xs = (inds_spatial % W).long()

    xs_float = xs.float()
    ys_float = ys.float()

    if reg is not None:
        reg_x = reg[0, 0, ys, xs]
        reg_y = reg[0, 1, ys, xs]
        xs_float = xs_float + reg_x
        ys_float = ys_float + reg_y
    else:
        xs_float = xs_float + 0.5
        ys_float = ys_float + 0.5

    pred_w = wh[0, 0, ys, xs].clamp(min=1.0)
    pred_h = wh[0, 1, ys, xs].clamp(min=1.0)

    # 注意：这里的坐标先在输出特征图尺度上，再乘 down_ratio 回到输入图像尺度
    x1 = (xs_float - pred_w / 2) * down_ratio
    y1 = (ys_float - pred_h / 2) * down_ratio
    x2 = (xs_float + pred_w / 2) * down_ratio
    y2 = (ys_float + pred_h / 2) * down_ratio

    cx = xs_float * down_ratio
    cy = ys_float * down_ratio

    dets = []
    for i in range(K):
        dets.append({
            'score': float(scores[i].detach().cpu()),
            'cx': float(cx[i].detach().cpu()),
            'cy': float(cy[i].detach().cpu()),
            'x1': float(x1[i].detach().cpu()),
            'y1': float(y1[i].detach().cpu()),
            'x2': float(x2[i].detach().cpu()),
            'y2': float(y2[i].detach().cpu()),
        })

    return dets


# =========================================================
# 工具函数：GT 读取、IoU、匹配
# =========================================================
def get_gt_bbox(row):
    """
    尽量自动从 labels.csv 中读取 GT bbox。

    支持以下几种列名：
    1. x1, y1, x2, y2
    2. xmin, ymin, xmax, ymax
    3. cx, cy, w, h
    4. cx, cy, width, height

    如果没有 bbox 信息，返回 None。
    """
    cols = row.index

    if all(c in cols for c in ['x1', 'y1', 'x2', 'y2']):
        return [
            float(row['x1']),
            float(row['y1']),
            float(row['x2']),
            float(row['y2'])
        ]

    if all(c in cols for c in ['xmin', 'ymin', 'xmax', 'ymax']):
        return [
            float(row['xmin']),
            float(row['ymin']),
            float(row['xmax']),
            float(row['ymax'])
        ]

    if all(c in cols for c in ['cx', 'cy', 'w', 'h']):
        cx, cy, w, h = float(row['cx']), float(row['cy']), float(row['w']), float(row['h'])
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    if all(c in cols for c in ['cx', 'cy', 'width', 'height']):
        cx, cy = float(row['cx']), float(row['cy'])
        w, h = float(row['width']), float(row['height'])
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    return None


def get_gt_center(row):
    if 'cx' in row.index and 'cy' in row.index:
        if float(row['cx']) > 0 and float(row['cy']) > 0:
            return float(row['cx']), float(row['cy'])
    return None


def bbox_iou(box1, box2):
    """
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter + 1e-8
    return inter / union


def match_det_to_gt(det, gt_bbox=None, gt_center=None):
    """
    如果有 bbox，用 IoU 判断。
    如果没有 bbox，用中心点距离判断。
    """
    if gt_bbox is not None:
        pred_box = [det['x1'], det['y1'], det['x2'], det['y2']]
        iou = bbox_iou(pred_box, gt_bbox)
        loc_error = None

        if gt_center is not None:
            loc_error = np.sqrt((det['cx'] - gt_center[0]) ** 2 + (det['cy'] - gt_center[1]) ** 2)

        return iou >= IOU_THRESH, iou, loc_error

    if gt_center is not None:
        loc_error = np.sqrt((det['cx'] - gt_center[0]) ** 2 + (det['cy'] - gt_center[1]) ** 2)
        return loc_error <= CENTER_TOL, None, loc_error

    return False, None, None


# =========================================================
# 读取数据
# =========================================================
df = pd.read_csv(CSV_PATH)
test_df = df.reset_index(drop=True)
print(f'测试集样本数: {len(test_df)}')


# =========================================================
# 推理：保存每张图的所有候选检测结果
# =========================================================
sample_records = []
det_rows = []

for i, (_, row) in enumerate(test_df.iterrows()):
    class_id = int(row['class_id'])
    sig_code = int(row['sig_type'])

    fname = (
        f"sample_{int(row['sample_id']):05d}_"
        f"{SIG_NAME[sig_code]}_cat{class_id}.mat"
    )
    mat_path = os.path.join(MAT_DIR, fname)

    if not os.path.exists(mat_path):
        print(f'[跳过] 文件不存在: {fname}')
        continue

    tf = sio.loadmat(mat_path)['tf_matrix'].astype(np.float32)

    # 和你原脚本保持一致
    tf_norm = (tf - 0.5) / 0.5

    inp = torch.from_numpy(
        np.stack([tf_norm, tf_norm, tf_norm], axis=0)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(inp)[-1]
        dets = decode_centernet_output(output, K=TOPK, down_ratio=DOWN_RATIO)

    gt_exists = class_id == 2
    gt_center = get_gt_center(row) if gt_exists else None
    gt_bbox = get_gt_bbox(row) if gt_exists else None

    sample_id = int(row['sample_id'])

    sample_records.append({
        'sample_id': sample_id,
        'sig_type': sig_code,
        'sig_name': SIG_NAME[sig_code],
        'gt_class': class_id,
        'gt_exists': gt_exists,
        'gt_cx': gt_center[0] if gt_center is not None else np.nan,
        'gt_cy': gt_center[1] if gt_center is not None else np.nan,
        'gt_bbox_x1': gt_bbox[0] if gt_bbox is not None else np.nan,
        'gt_bbox_y1': gt_bbox[1] if gt_bbox is not None else np.nan,
        'gt_bbox_x2': gt_bbox[2] if gt_bbox is not None else np.nan,
        'gt_bbox_y2': gt_bbox[3] if gt_bbox is not None else np.nan,
        'dets': dets,
    })

    for rank, det in enumerate(dets):
        det_rows.append({
            'sample_id': sample_id,
            'sig_type': sig_code,
            'sig_name': SIG_NAME[sig_code],
            'gt_class': class_id,
            'gt_exists': gt_exists,
            'rank': rank + 1,
            'score': det['score'],
            'pred_cx': det['cx'],
            'pred_cy': det['cy'],
            'pred_x1': det['x1'],
            'pred_y1': det['y1'],
            'pred_x2': det['x2'],
            'pred_y2': det['y2'],
        })

    if (i + 1) % 100 == 0 or (i + 1) == len(test_df):
        print(f'推理进度: {i + 1}/{len(test_df)}')


det_df = pd.DataFrame(det_rows)
det_df.to_csv(os.path.join(SAVE_DIR, 'all_decoded_detections.csv'), index=False)


# =========================================================
# 按阈值评估
# =========================================================
def evaluate_records(records, threshold=0.3, name='Overall'):
    """
    输出两套指标：

    1. Image-level：
       只判断这张图有没有目标。
       这和你原来的 TP/FP/FN/TN 更接近。

    2. Detection-level：
       不仅要求有目标响应，还要求位置匹配。
       有 bbox 时用 IoU，没有 bbox 时用中心点距离。
    """

    # 图像级分类指标
    img_TP = img_FP = img_FN = img_TN = 0

    # 检测级定位匹配指标
    det_TP = det_FP = det_FN = 0

    loc_errors = []
    ious = []

    detail_rows = []

    for rec in records:
        gt_exists = rec['gt_exists']
        gt_bbox = None
        if not np.isnan(rec['gt_bbox_x1']):
            gt_bbox = [
                rec['gt_bbox_x1'],
                rec['gt_bbox_y1'],
                rec['gt_bbox_x2'],
                rec['gt_bbox_y2']
            ]

        gt_center = None
        if not np.isnan(rec['gt_cx']):
            gt_center = [rec['gt_cx'], rec['gt_cy']]

        dets = [d for d in rec['dets'] if d['score'] >= threshold]
        dets = sorted(dets, key=lambda x: x['score'], reverse=True)

        pred_exists = len(dets) > 0
        best_det = dets[0] if pred_exists else None

        # ----------------------------
        # 1. 图像级判断
        # ----------------------------
        if gt_exists and pred_exists:
            img_TP += 1
        elif (not gt_exists) and pred_exists:
            img_FP += 1
        elif gt_exists and (not pred_exists):
            img_FN += 1
        else:
            img_TN += 1

        # ----------------------------
        # 2. 检测级匹配
        # ----------------------------
        matched = False
        best_iou = np.nan
        best_loc_error = np.nan

        if gt_exists:
            if best_det is None:
                det_FN += 1
            else:
                matched, best_iou, best_loc_error = match_det_to_gt(
                    best_det,
                    gt_bbox=gt_bbox,
                    gt_center=gt_center
                )

                if matched:
                    det_TP += 1
                    if best_loc_error is not None:
                        loc_errors.append(best_loc_error)
                    if best_iou is not None:
                        ious.append(best_iou)
                else:
                    # 有预测但位置不对：既是一次错误预测，也是漏掉了真实目标
                    det_FP += 1
                    det_FN += 1
        else:
            if best_det is not None:
                det_FP += 1

        detail_rows.append({
            'sample_id': rec['sample_id'],
            'sig_type': rec['sig_type'],
            'sig_name': rec['sig_name'],
            'gt_class': rec['gt_class'],
            'pred_exists': int(pred_exists),
            'top_score': best_det['score'] if best_det is not None else 0.0,
            'matched': int(matched),
            'iou': best_iou,
            'loc_error': best_loc_error,
            'pred_cx': best_det['cx'] if best_det is not None else np.nan,
            'pred_cy': best_det['cy'] if best_det is not None else np.nan,
            'gt_cx': rec['gt_cx'],
            'gt_cy': rec['gt_cy'],
        })

    # 图像级指标
    img_precision = img_TP / (img_TP + img_FP + 1e-8)
    img_recall = img_TP / (img_TP + img_FN + 1e-8)
    img_f1 = 2 * img_precision * img_recall / (img_precision + img_recall + 1e-8)
    img_acc = (img_TP + img_TN) / (img_TP + img_FP + img_FN + img_TN + 1e-8)

    # 检测级指标
    det_precision = det_TP / (det_TP + det_FP + 1e-8)
    det_recall = det_TP / (det_TP + det_FN + 1e-8)
    det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall + 1e-8)

    mean_loc = np.mean(loc_errors) if len(loc_errors) > 0 else np.nan
    mean_iou = np.mean(ious) if len(ious) > 0 else np.nan

    print(f'\n{"=" * 60}')
    print(f'【{name}】threshold={threshold}')
    print(f'样本数: {len(records)}')

    print('\n[Image-level: 是否有目标]')
    print(f'  TP={img_TP}  FP={img_FP}  FN={img_FN}  TN={img_TN}')
    print(f'  Accuracy : {img_acc:.4f}')
    print(f'  Precision: {img_precision:.4f}')
    print(f'  Recall   : {img_recall:.4f}')
    print(f'  F1 Score : {img_f1:.4f}')

    print('\n[Detection-level: 位置匹配后]')
    print(f'  TP={det_TP}  FP={det_FP}  FN={det_FN}')
    print(f'  Precision: {det_precision:.4f}')
    print(f'  Recall   : {det_recall:.4f}')
    print(f'  F1 Score : {det_f1:.4f}')
    print(f'  Mean Loc Error: {mean_loc:.2f} px')
    if not np.isnan(mean_iou):
        print(f'  Mean IoU      : {mean_iou:.4f}')

    return {
        'name': name,
        'threshold': threshold,

        'img_TP': img_TP,
        'img_FP': img_FP,
        'img_FN': img_FN,
        'img_TN': img_TN,
        'img_accuracy': img_acc,
        'img_precision': img_precision,
        'img_recall': img_recall,
        'img_f1': img_f1,

        'det_TP': det_TP,
        'det_FP': det_FP,
        'det_FN': det_FN,
        'det_precision': det_precision,
        'det_recall': det_recall,
        'det_f1': det_f1,

        'mean_loc_error': mean_loc,
        'mean_iou': mean_iou,

        'details': pd.DataFrame(detail_rows)
    }


# =========================================================
# AP 计算
# =========================================================
def voc_ap(rec, prec):
    """
    VOC-style AP，使用积分形式。
    """
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def compute_ap(records, name='Overall'):
    """
    使用所有候选框计算 AP。
    如果有 GT bbox，则是 AP@IoU。
    如果没有 GT bbox，则是 AP@CenterTol。
    """
    preds = []
    num_gt = 0

    for rec in records:
        if rec['gt_exists']:
            num_gt += 1

        for det in rec['dets']:
            preds.append({
                'sample_id': rec['sample_id'],
                'score': det['score'],
                'det': det,
                'rec': rec
            })

    preds = sorted(preds, key=lambda x: x['score'], reverse=True)

    matched_gt = set()
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for i, pred in enumerate(preds):
        rec = pred['rec']

        if not rec['gt_exists']:
            fp[i] = 1
            continue

        gt_key = rec['sample_id']

        gt_bbox = None
        if not np.isnan(rec['gt_bbox_x1']):
            gt_bbox = [
                rec['gt_bbox_x1'],
                rec['gt_bbox_y1'],
                rec['gt_bbox_x2'],
                rec['gt_bbox_y2']
            ]

        gt_center = None
        if not np.isnan(rec['gt_cx']):
            gt_center = [rec['gt_cx'], rec['gt_cy']]

        matched, _, _ = match_det_to_gt(
            pred['det'],
            gt_bbox=gt_bbox,
            gt_center=gt_center
        )

        if matched and gt_key not in matched_gt:
            tp[i] = 1
            matched_gt.add(gt_key)
        else:
            fp[i] = 1

    if num_gt == 0:
        return np.nan

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    rec = cum_tp / (num_gt + 1e-8)
    prec = cum_tp / (cum_tp + cum_fp + 1e-8)

    ap = voc_ap(rec, prec)

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve - {name}, AP={ap:.4f}')
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'pr_curve_{name}.png'), dpi=150)
    plt.close()

    return ap


# =========================================================
# 正式评估
# =========================================================
all_result = evaluate_records(sample_records, threshold=THRESHOLD, name='Overall')
all_result['details'].to_csv(os.path.join(SAVE_DIR, 'details_overall.csv'), index=False)

for sig_code, sig_name in SIG_NAME.items():
    sub_records = [r for r in sample_records if r['sig_type'] == sig_code]
    result = evaluate_records(sub_records, threshold=THRESHOLD, name=sig_name)
    result['details'].to_csv(os.path.join(SAVE_DIR, f'details_{sig_name}.csv'), index=False)


# =========================================================
# 阈值敏感性分析
# =========================================================
threshold_rows = []

print(f'\n\n{"Threshold Sensitivity":=^70}')
print(f'{"Thr":>6} {"Img_P":>8} {"Img_R":>8} {"Img_F1":>8} {"Det_P":>8} {"Det_R":>8} {"Det_F1":>8}')

for thresh in np.arange(0.1, 1.0, 0.1):
    result = evaluate_records(sample_records, threshold=float(thresh), name=f'Thr={thresh:.1f}')

    threshold_rows.append({
        'threshold': thresh,
        'img_precision': result['img_precision'],
        'img_recall': result['img_recall'],
        'img_f1': result['img_f1'],
        'img_accuracy': result['img_accuracy'],
        'det_precision': result['det_precision'],
        'det_recall': result['det_recall'],
        'det_f1': result['det_f1'],
        'mean_loc_error': result['mean_loc_error'],
        'mean_iou': result['mean_iou'],
    })

    print(
        f'{thresh:>6.1f} '
        f'{result["img_precision"]:>8.4f} '
        f'{result["img_recall"]:>8.4f} '
        f'{result["img_f1"]:>8.4f} '
        f'{result["det_precision"]:>8.4f} '
        f'{result["det_recall"]:>8.4f} '
        f'{result["det_f1"]:>8.4f}'
    )

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(os.path.join(SAVE_DIR, 'threshold_metrics_full.csv'), index=False)


plt.figure(figsize=(8, 5))
plt.plot(threshold_df['threshold'], threshold_df['img_f1'], marker='o', label='Image-level F1')
plt.plot(threshold_df['threshold'], threshold_df['det_f1'], marker='o', label='Detection-level F1')
plt.plot(threshold_df['threshold'], threshold_df['img_recall'], marker='o', label='Image-level Recall')
plt.plot(threshold_df['threshold'], threshold_df['det_recall'], marker='o', label='Detection-level Recall')
plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.title('Threshold Sensitivity')
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'threshold_sensitivity_full.png'), dpi=150)
plt.close()


# =========================================================
# AP 评估
# =========================================================
ap_overall = compute_ap(sample_records, name='Overall')
print(f'\nAP Overall: {ap_overall:.4f}')

for sig_code, sig_name in SIG_NAME.items():
    sub_records = [r for r in sample_records if r['sig_type'] == sig_code]
    ap = compute_ap(sub_records, name=sig_name)
    print(f'AP {sig_name}: {ap:.4f}')


# =========================================================
# 得分分布图
# =========================================================
top_score_rows = []
for rec in sample_records:
    top_score = max([d['score'] for d in rec['dets']]) if len(rec['dets']) > 0 else 0.0
    top_score_rows.append({
        'sample_id': rec['sample_id'],
        'sig_type': rec['sig_type'],
        'sig_name': rec['sig_name'],
        'gt_class': rec['gt_class'],
        'gt_exists': rec['gt_exists'],
        'top_score': top_score
    })

top_score_df = pd.DataFrame(top_score_rows)
top_score_df.to_csv(os.path.join(SAVE_DIR, 'top_score_per_sample.csv'), index=False)

pos_scores = top_score_df[top_score_df.gt_exists == True]['top_score'].values
neg_scores = top_score_df[top_score_df.gt_exists == False]['top_score'].values

plt.figure(figsize=(8, 5))
plt.hist(pos_scores, bins=30, alpha=0.6, label='Positive Samples')
plt.hist(neg_scores, bins=30, alpha=0.6, label='Negative Samples')
plt.axvline(THRESHOLD, linestyle='--', label=f'Threshold={THRESHOLD}')
plt.xlabel('Top Detection Score')
plt.ylabel('Number of Samples')
plt.title('Score Distribution')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'score_distribution_full.png'), dpi=150)
plt.close()


# =========================================================
# 保存 FN / FP / TP 样本
# =========================================================
details = all_result['details']

tp_df = details[(details.gt_class == 2) & (details.matched == 1)]
fn_df = details[(details.gt_class == 2) & (details.matched == 0)]
fp_df = details[(details.gt_class == 1) & (details.pred_exists == 1)]

tp_df.to_csv(os.path.join(SAVE_DIR, 'tp_samples_detection_level.csv'), index=False)
fn_df.to_csv(os.path.join(SAVE_DIR, 'fn_samples_detection_level.csv'), index=False)
fp_df.to_csv(os.path.join(SAVE_DIR, 'fp_samples_detection_level.csv'), index=False)


print(f'\n完整评估完成，结果已保存到: {os.path.abspath(SAVE_DIR)}')
print('主要文件：')
print('  all_decoded_detections.csv')
print('  details_overall.csv')
print('  details_LFM.csv')
print('  details_HFM.csv')
print('  threshold_metrics_full.csv')
print('  threshold_sensitivity_full.png')
print('  pr_curve_Overall.png')
print('  pr_curve_LFM.png')
print('  pr_curve_HFM.png')
print('  score_distribution_full.png')
print('  tp_samples_detection_level.csv')
print('  fn_samples_detection_level.csv')
print('  fp_samples_detection_level.csv')
