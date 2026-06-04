import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SimpleFPN(nn.Module):
    def __init__(self, in_channels, fpn_channels=128):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(ch, fpn_channels, kernel_size=1) for ch in in_channels
        ])
        self.smooth = nn.ModuleList([
            ConvBnRelu(fpn_channels, fpn_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

    @staticmethod
    def _upsample_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='nearest')

    def forward(self, features):
        c2, c3, c4, c5 = features

        p5 = self.laterals[3](c5)
        p4 = self.laterals[2](c4) + self._upsample_to(p5, c4)
        p3 = self.laterals[1](c3) + self._upsample_to(p4, c3)
        p2 = self.laterals[0](c2) + self._upsample_to(p3, c2)

        p2 = self.smooth[0](p2)
        p3 = self.smooth[1](p3)
        p4 = self.smooth[2](p4)
        p5 = self.smooth[3](p5)

        fused = (
            p2
            + self._upsample_to(p3, p2)
            + self._upsample_to(p4, p2)
            + self._upsample_to(p5, p2)
        )

        return fused, {'P2': p2, 'P3': p3, 'P4': p4, 'P5': p5, 'fused': fused}


class SwinV2TinyFPNCenterNet(nn.Module):
    def __init__(self, heads, head_conv,
                 model_name='swinv2_tiny_window8_256',
                 pretrained=True,
                 fpn_channels=128):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        in_channels = self._get_feature_channels()
        self.fpn = SimpleFPN(in_channels, fpn_channels=fpn_channels)

        self.heads = heads
        for head, num_output in heads.items():
            fc = nn.Sequential(
                nn.Conv2d(fpn_channels, head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0, bias=True)
            )
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                nn.init.normal_(fc[-1].weight, std=0.001)
                nn.init.constant_(fc[-1].bias, 0)
            self.__setattr__(head, fc)

    def _get_feature_channels(self):
        if hasattr(self.backbone, 'feature_info'):
            try:
                return self.backbone.feature_info.channels()
            except Exception:
                pass
        return [96, 192, 384, 768]

    @staticmethod
    def _to_nchw(feat):
        if feat.ndim != 4:
            raise ValueError('Expected a 4D feature map.')
        # timm Swin features are NHWC; keep this tolerant for future backbones.
        if feat.shape[1] < feat.shape[-1]:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat.contiguous()

    def forward(self, x, return_debug=False):
        raw_features = self.backbone(x)
        features = [self._to_nchw(feat) for feat in raw_features]

        fused, fpn_debug = self.fpn(features)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(fused)

        outputs = [ret]
        if not return_debug:
            return outputs

        stage_debug = {
            'C{}'.format(i + 2): feat for i, feat in enumerate(features)
        }
        return outputs, {'stages': stage_debug, 'fpn': fpn_debug}
