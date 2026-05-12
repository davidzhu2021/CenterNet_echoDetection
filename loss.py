# loss.py
import torch
import torch.nn.functional as F


def focal_loss(pred, gt, alpha=2, beta=4):
    """
    CenterNet Focal Loss（CornerNet版）
    pred: [B, 1, H, W]  网络输出，已经过Sigmoid，值域0~1
    gt:   [B, 1, H, W]  高斯热力图GT，峰值为1，周围按高斯衰减

    正样本（gt==1）：权重 = (1-pred)^alpha
    负样本（gt<1） ：权重 = (1-gt)^beta * pred^alpha
    """
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)  # 防止log(0)

    pos_mask = (gt == 1).float()               # 峰值点
    neg_mask = (gt <  1).float()               # 其余点

    pos_loss = torch.log(pred) \
               * torch.pow(1 - pred, alpha) \
               * pos_mask

    neg_loss = torch.log(1 - pred) \
               * torch.pow(pred, alpha) \
               * torch.pow(1 - gt, beta) \
               * neg_mask

    num_pos  = pos_mask.sum().clamp(min=1)     # 避免除以0
    loss     = -(pos_loss.sum() + neg_loss.sum()) / num_pos
    return loss


def kp_offset_loss(pred_kp, gt_kp, gt_heatmap):
    """
    关键点偏移L1 Loss
    只在GT中心点（热力图峰值==1）处计算，其余位置忽略

    pred_kp:    [B, 4, H, W]
    gt_kp:      [B, 4, H, W]
    gt_heatmap: [B, 1, H, W]
    """
    # 用热力图峰值位置作为mask
    mask = (gt_heatmap == 1).float()           # [B, 1, H, W]
    mask = mask.expand_as(pred_kp)             # [B, 4, H, W]

    num_pos = mask.sum().clamp(min=1)
    loss    = F.l1_loss(pred_kp * mask, gt_kp * mask, reduction='sum')
    loss    = loss / (num_pos / 4)             # 除以4是因为4个通道共享同一个点数
    return loss


def centernet_loss(pred_hm, pred_kp, gt_hm, gt_kp,
                   hm_weight=1.0, kp_weight=0.1):
    """
    总损失 = hm_weight * focal_loss + kp_weight * kp_offset_loss

    权重设置参考原论文：heatmap loss权重远大于offset loss，
    因为offset只在极少数峰值点上算，天然就比heatmap loss小很多
    """
    loss_hm = focal_loss(pred_hm, gt_hm)
    loss_kp = kp_offset_loss(pred_kp, gt_kp, gt_hm)
    loss    = hm_weight * loss_hm + kp_weight * loss_kp

    return loss, loss_hm, loss_kp


# -------------------------------------------------------
if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    B, H, W = 2, 64, 64

    # 模拟一个batch：1个有目标，1个纯混响
    pred_hm = torch.rand(B, 1, H, W) * 0.2      # 网络初始输出接近0
    pred_kp = torch.randn(B, 4, H, W) * 0.1

    gt_hm   = torch.zeros(B, 1, H, W)
    gt_kp   = torch.zeros(B, 4, H, W)

    # 第0个样本是target，在(32,32)处有GT峰值
    gt_hm[0, 0, 32, 32] = 1.0
    gt_kp[0, :, 32, 32] = torch.tensor([-8.0, 4.0, 8.0, -4.0])  # 头尾偏移

    loss, lhm, lkp = centernet_loss(pred_hm, pred_kp, gt_hm, gt_kp)

    print(f'Total Loss : {loss.item():.4f}')
    print(f'Heatmap Loss: {lhm.item():.4f}')
    print(f'KP Loss     : {lkp.item():.4f}')
    print('Loss计算正常 ✅' if loss.item() > 0 else '❌ Loss异常')