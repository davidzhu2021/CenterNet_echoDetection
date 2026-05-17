import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import re
import matplotlib.pyplot as plt

log_path = r'E:\CenterNet\exp\ctdet\sonar_exp1\logs_2026-05-12-14-43\log.txt'

train_epochs, train_loss, train_hm, train_wh = [], [], [], []
val_epochs,   val_loss,   val_hm,   val_wh   = [], [], [], []

with open(log_path, 'r') as f:
    for line in f:
        ep_match = re.search(r'epoch: (\d+)', line)
        if not ep_match:
            continue
        epoch = int(ep_match.group(1))

        # 每行可能有1组（只有train）或2组（train+val）loss数值
        all_losses = re.findall(
            r'loss ([0-9.]+) \| hm_loss ([0-9.]+) \| wh_loss ([0-9.]+)', line)

        if len(all_losses) >= 1:
            l, h, w = all_losses[0]
            train_epochs.append(epoch)
            train_loss.append(float(l))
            train_hm.append(float(h))
            train_wh.append(float(w))

        if len(all_losses) >= 2:
            l, h, w = all_losses[1]
            val_epochs.append(epoch)
            val_loss.append(float(l))
            val_hm.append(float(h))
            val_wh.append(float(w))

print(f'Train epochs: {len(train_epochs)}')
print(f'Val epochs:   {len(val_epochs)}')
print(f'Train Loss  epoch1={train_loss[0]:.4f} -> epoch{train_epochs[-1]}={train_loss[-1]:.4f}')
print(f'Val   Loss  epoch{val_epochs[0]}={val_loss[0]:.4f} -> epoch{val_epochs[-1]}={val_loss[-1]:.4f}')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Total Loss
axes[0].plot(train_epochs, train_loss, 'b-',  linewidth=1.5, label='Train')
axes[0].plot(val_epochs,   val_loss,   'ro--', linewidth=1.5, markersize=5, label='Val')
axes[0].set_title('Total Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Heatmap Loss
axes[1].plot(train_epochs, train_hm, 'b-',  linewidth=1.5, label='Train')
axes[1].plot(val_epochs,   val_hm,   'ro--', linewidth=1.5, markersize=5, label='Val')
axes[1].set_title('Heatmap Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# WH Loss
axes[2].plot(train_epochs, train_wh, 'b-',  linewidth=1.5, label='Train')
axes[2].plot(val_epochs,   val_wh,   'ro--', linewidth=1.5, markersize=5, label='Val')
axes[2].set_title('WH Loss')
axes[2].set_xlabel('Epoch')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('train_curve.png', dpi=150, bbox_inches='tight')
print('曲线已保存到 train_curve.png')
plt.show()