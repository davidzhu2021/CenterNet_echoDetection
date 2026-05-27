import timm
import torch

m = timm.create_model('swinv2_tiny_window8_256',
                      features_only=True,
                      out_indices=(3,))
m.eval()

x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    y = m(x)

print('输出shape:', y[-1].shape)
print('通道数:', y[-1].shape[1] if y[-1].dim() == 4 else y[-1].shape[-1])
print('feature_info:', m.feature_info[-1])