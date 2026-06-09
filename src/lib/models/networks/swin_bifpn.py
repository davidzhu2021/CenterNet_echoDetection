import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBnAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class WeightedFusion(nn.Module):
    def __init__(self, num_inputs, epsilon=1e-4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.epsilon = epsilon

    def forward(self, inputs):
        weights = F.relu(self.weights)
        norm = weights / (weights.sum() + self.epsilon)
        out = 0
        for weight, tensor in zip(norm, inputs):
            out = out + weight * tensor
        return out


class WeightedBiFPN(nn.Module):
    def __init__(self, in_channels, bifpn_channels=128):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(ch, bifpn_channels, kernel_size=1)
            for ch in in_channels
        ])

        self.td_fuse = nn.ModuleDict({
            'P4_td': WeightedFusion(2),
            'P3_td': WeightedFusion(2),
            'P2_td': WeightedFusion(2),
        })
        self.bu_fuse = nn.ModuleDict({
            'P3_out': WeightedFusion(3),
            'P4_out': WeightedFusion(3),
            'P5_out': WeightedFusion(2),
        })

        self.td_smooth = nn.ModuleDict({
            'P4_td': ConvBnAct(bifpn_channels, bifpn_channels, kernel_size=3, padding=1),
            'P3_td': ConvBnAct(bifpn_channels, bifpn_channels, kernel_size=3, padding=1),
            'P2_td': ConvBnAct(bifpn_channels, bifpn_channels, kernel_size=3, padding=1),
        })
        self.bu_smooth = nn.ModuleDict({
            'P3_out': ConvBnAct(bifpn_channels, bifpn_channels, kernel_size=3, padding=1),
            'P4_out': ConvBnAct(bifpn_channels, bifpn_channels, kernel_size=3, padding=1),
            'P5_out': ConvBnAct(bifpn_channels, bifpn_channels, kernel_size=3, padding=1),
        })

    @staticmethod
    def _upsample_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='nearest')

    @staticmethod
    def _downsample_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='nearest')

    def forward(self, features):
        c2, c3, c4, c5 = features
        p2_in = self.laterals[0](c2)
        p3_in = self.laterals[1](c3)
        p4_in = self.laterals[2](c4)
        p5_in = self.laterals[3](c5)

        p5_td = p5_in
        p4_td = self.td_smooth['P4_td'](
            self.td_fuse['P4_td']([p4_in, self._upsample_to(p5_td, p4_in)])
        )
        p3_td = self.td_smooth['P3_td'](
            self.td_fuse['P3_td']([p3_in, self._upsample_to(p4_td, p3_in)])
        )
        p2_out = self.td_smooth['P2_td'](
            self.td_fuse['P2_td']([p2_in, self._upsample_to(p3_td, p2_in)])
        )

        p3_out = self.bu_smooth['P3_out'](
            self.bu_fuse['P3_out']([p3_in, p3_td, self._downsample_to(p2_out, p3_in)])
        )
        p4_out = self.bu_smooth['P4_out'](
            self.bu_fuse['P4_out']([p4_in, p4_td, self._downsample_to(p3_out, p4_in)])
        )
        p5_out = self.bu_smooth['P5_out'](
            self.bu_fuse['P5_out']([p5_in, self._downsample_to(p4_out, p5_in)])
        )

        fused = (
            p2_out
            + self._upsample_to(p3_out, p2_out)
            + self._upsample_to(p4_out, p2_out)
            + self._upsample_to(p5_out, p2_out)
        )

        return fused, {
            'P2': p2_out,
            'P3': p3_out,
            'P4': p4_out,
            'P5': p5_out,
            'fused': fused,
        }


class SwinV2TinyBiFPNCenterNet(nn.Module):
    def __init__(self, heads, head_conv,
                 model_name='swinv2_tiny_window8_256',
                 pretrained=True,
                 bifpn_channels=128,
                 use_cnn_stem=False,
                 in_chans=3):
        super().__init__()

        self.use_cnn_stem = use_cnn_stem
        self.cnn_stem = nn.Identity()
        if use_cnn_stem:
            self.cnn_stem = nn.Sequential(
                nn.Conv2d(in_chans, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(3),
                nn.SiLU(inplace=True),
            )

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        in_channels = self._get_feature_channels()
        self.bifpn = WeightedBiFPN(in_channels, bifpn_channels=bifpn_channels)

        self.heads = heads
        for head, num_output in heads.items():
            fc = nn.Sequential(
                nn.Conv2d(bifpn_channels, head_conv, kernel_size=3, padding=1, bias=True),
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
        x = self.cnn_stem(x)
        raw_features = self.backbone(x)
        features = [self._to_nchw(feat) for feat in raw_features]

        fused, bifpn_debug = self.bifpn(features)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(fused)

        outputs = [ret]
        if not return_debug:
            return outputs

        stage_debug = {
            'C{}'.format(i + 2): feat for i, feat in enumerate(features)
        }
        return outputs, {'stages': stage_debug, 'bifpn': bifpn_debug}
