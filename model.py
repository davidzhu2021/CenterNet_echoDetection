# model.py
import torch
import torch.nn as nn
import torchvision.models as models


class DoubleConv(nn.Module):
    """上采样模块中的卷积块"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class DetHead(nn.Module):
    """检测头：一个卷积分支"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, kernel_size=1)
        )
    def forward(self, x):
        return self.head(x)


class CenterNet(nn.Module):
    """
    CenterNet for Sonar TF Matrix
    输入:  [B, 3, 256, 256]
    输出:
        heatmap  [B, 1, 64, 64]  中心点置信度（Sigmoid后）
        kp_map   [B, 4, 64, 64]  头尾偏移 dx_head/dy_head/dx_tail/dy_tail
    """
    def __init__(self, pretrained=True):
        super().__init__()

        # --- Backbone: ResNet50 ---
        # resnet = models.resnet50(pretrained=pretrained)
        resnet = models.resnet50(weights=None)
        # 去掉最后的avgpool和fc，只保留卷积特征提取部分
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # 输出: [B, 2048, 8, 8]（输入256×256时）

        # --- 上采样：8→16→32→64，三次转置卷积 ---
        self.up1 = DoubleConv(2048, 256)   # 8  → 16
        self.up2 = DoubleConv(256,  128)   # 16 → 32
        self.up3 = DoubleConv(128,  64)    # 32 → 64
        # 输出: [B, 64, 64, 64]

        # --- 检测头 ---
        self.heatmap_head = nn.Sequential(
            DetHead(64, 1),
            nn.Sigmoid()          # 输出0~1置信度
        )
        self.kp_head = DetHead(64, 4)     # 偏移不加激活函数，可正可负

        # --- 权重初始化 ---
        self._init_heads()

    def _init_heads(self):
        """检测头用小值初始化，避免训练初期梯度爆炸"""
        for m in [self.heatmap_head, self.kp_head]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.001)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        # heatmap最后一层bias初始化为-2.19（让初始预测接近0，
        # 避免Focal Loss在训练开始时被大量负样本主导）
        self.heatmap_head[0].head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        feat    = self.backbone(x)    # [B, 2048, 8, 8]
        feat    = self.up1(feat)      # [B, 256,  16, 16]
        feat    = self.up2(feat)      # [B, 128,  32, 32]
        feat    = self.up3(feat)      # [B, 64,   64, 64]

        heatmap = self.heatmap_head(feat)   # [B, 1, 64, 64]
        kp_map  = self.kp_head(feat)        # [B, 4, 64, 64]

        return heatmap, kp_map


# -------------------------------------------------------
if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # model = CenterNet(pretrained=True)
    model = CenterNet(pretrained=False)
    model.eval()

    dummy = torch.randn(2, 3, 256, 256)   # batch=2
    hm, kp = model(dummy)

    print('Backbone参数量: ',
          sum(p.numel() for p in model.backbone.parameters()) / 1e6, 'M')
    print('总参数量:       ',
          sum(p.numel() for p in model.parameters()) / 1e6, 'M')
    print('heatmap shape: ', hm.shape)    # [2, 1, 64, 64]
    print('kp_map shape:  ', kp.shape)    # [2, 4, 64, 64]
    print('heatmap range: ', hm.min().item(), '~', hm.max().item())  # 0~1