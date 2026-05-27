import torch
import torch.nn as nn
import timm

class SwinCenterNet(nn.Module):
    def __init__(self, heads, head_conv,
                 model_name='swinv2_tiny_window8_256',
                 pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,)
        )

        # swinv2_tiny输出通道数768，空间尺寸8×8
        in_channels = 768

        # --- 上采样：8→16→32→64 ---
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        # --- 检测头 ---
        self.heads = heads
        for head, num_output in heads.items():
            fc = nn.Sequential(
                nn.Conv2d(64, head_conv,
                          kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output,
                          kernel_size=1, stride=1, padding=0, bias=True)
            )
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                nn.init.normal_(fc[-1].weight, std=0.001)
                nn.init.constant_(fc[-1].bias, 0)
            self.__setattr__(head, fc)

    def forward(self, x):
        features = self.backbone(x)
        feat = features[-1]   # [B, 8, 8, 768]

        # [B, H, W, C] → [B, C, H, W]
        feat = feat.permute(0, 3, 1, 2).contiguous()  # [B, 768, 8, 8]

        feat = self.upsample(feat)   # [B, 64, 64, 64]

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feat)
        return [ret]