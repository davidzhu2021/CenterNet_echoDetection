# train.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import SonarDataset
from model import CenterNet
from loss import centernet_loss


# -------------------------------------------------------
# 配置参数
# -------------------------------------------------------
CSV_PATH  = r'D:\gradu\simulation_5\tf_dataset_matrix\labels.csv'
MAT_DIR   = r'D:\gradu\simulation_5\tf_dataset_matrix\matrices'
CKPT_DIR  = r'E:\CenterNet\checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

EPOCHS     = 3
BATCH_SIZE = 4       # 数据量少，batch不要太大
LR         = 1e-4
VAL_RATIO  = 0.2
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'使用设备: {DEVICE}')

# -------------------------------------------------------
# 数据
# -------------------------------------------------------
dataset = SonarDataset(csv_path=CSV_PATH, mat_dir=MAT_DIR)
n_val   = max(1, int(len(dataset) * VAL_RATIO))
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val],
                                  generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

print(f'训练集: {n_train} | 验证集: {n_val}')

# -------------------------------------------------------
# 模型
# -------------------------------------------------------
model = CenterNet(pretrained=False).to(DEVICE)

# 如果已下载好预训练权重，改成：
# model = CenterNet(pretrained=True).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# -------------------------------------------------------
# 训练循环
# -------------------------------------------------------
best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):

    # --- 训练 ---
    model.train()
    train_loss = 0.0
    train_lhm  = 0.0
    train_lkp  = 0.0

    for batch in train_loader:
        img    = batch['img'].to(DEVICE)        # [B, 3, 256, 256]
        gt_hm  = batch['heatmap'].to(DEVICE)    # [B, 1, 64, 64]
        gt_kp  = batch['kp_map'].to(DEVICE)     # [B, 4, 64, 64]

        pred_hm, pred_kp = model(img)
        loss, lhm, lkp   = centernet_loss(pred_hm, pred_kp, gt_hm, gt_kp)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        train_loss += loss.item()
        train_lhm  += lhm.item()
        train_lkp  += lkp.item()

    n_train_batch = len(train_loader)
    train_loss /= n_train_batch
    train_lhm  /= n_train_batch
    train_lkp  /= n_train_batch

    # --- 验证 ---
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            img   = batch['img'].to(DEVICE)
            gt_hm = batch['heatmap'].to(DEVICE)
            gt_kp = batch['kp_map'].to(DEVICE)

            pred_hm, pred_kp       = model(img)
            loss, _, _             = centernet_loss(pred_hm, pred_kp, gt_hm, gt_kp)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    scheduler.step()

    # --- 日志 ---
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch [{epoch:3d}/{EPOCHS}] '
              f'Train: {train_loss:.4f} '
              f'(hm={train_lhm:.4f} kp={train_lkp:.4f}) | '
              f'Val: {val_loss:.4f} | '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')

    # --- 保存最优模型 ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        ckpt_path = os.path.join(CKPT_DIR, 'best.pth')
        torch.save({
            'epoch':      epoch,
            'model':      model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'val_loss':   val_loss,
        }, ckpt_path)

print(f'\n训练完成！最优Val Loss: {best_val_loss:.4f}')
print(f'模型保存在: {os.path.join(CKPT_DIR, "best.pth")}')