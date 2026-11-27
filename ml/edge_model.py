"""
edge_model.py — TEED (Tiny and Efficient Edge Detector) 구조
=============================================================
원본: https://github.com/xavysp/TEED  (MIT License, Copyright (c) 2022 Xavier Soria Poma)
논문: Soria et al., "Tiny and Efficient Model for the Edge Detection Generalization", ICCV 2023 Workshop

파라미터 약 5.8만 개. 원본 구조를 그대로 두고, 웹 배포용 래퍼(EdgeNetForExport)만 추가했다.
"""

import torch
import torch.nn as nn


def smish(x):
    return x * torch.tanh(torch.log(1 + torch.sigmoid(x)))


class Smish(nn.Module):
    def forward(self, x):
        return smish(x)


def weight_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class DoubleFusion(nn.Module):
    """3개 스케일 출력을 합쳐 최종 엣지맵 1장으로"""
    def __init__(self, in_ch):
        super().__init__()
        self.DWconv1 = nn.Conv2d(in_ch, in_ch * 8, 3, padding=1, groups=in_ch)
        self.DWconv2 = nn.Conv2d(24, 24, 3, padding=1, groups=24)
        self.AF = Smish()

    def forward(self, x):
        attn = self.DWconv1(self.AF(x))
        attn2 = self.DWconv2(self.AF(attn))
        return smish((attn2 + attn).sum(1, keepdim=True))


class _DenseLayer(nn.Sequential):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.add_module('conv1', nn.Conv2d(in_f, out_f, 3, padding=2))
        self.add_module('smish1', Smish())
        self.add_module('conv2', nn.Conv2d(out_f, out_f, 3))

    def forward(self, x):
        x1, x2 = x
        return 0.5 * (super().forward(smish(x1)) + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_f, out_f):
        super().__init__()
        for i in range(num_layers):
            self.add_module(f'denselayer{i + 1}', _DenseLayer(in_f, out_f))
            in_f = out_f


class UpConvBlock(nn.Module):
    def __init__(self, in_f, up_scale):
        super().__init__()
        pads = [0, 0, 1, 3, 7]
        layers = []
        for i in range(up_scale):
            out_f = 1 if i == up_scale - 1 else 16
            layers += [nn.Conv2d(in_f, out_f, 1), Smish(),
                       nn.ConvTranspose2d(out_f, out_f, 2 ** up_scale, stride=2, padding=pads[up_scale])]
            in_f = out_f
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, in_f, out_f, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_f, out_f, 1, stride=stride)

    def forward(self, x):
        return self.conv(x)


class DoubleConvBlock(nn.Module):
    def __init__(self, in_f, mid_f, out_f=None, stride=1, use_act=True):
        super().__init__()
        self.use_act = use_act
        self.conv1 = nn.Conv2d(in_f, mid_f, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_f, out_f or mid_f, 3, padding=1)
        self.smish = Smish()

    def forward(self, x):
        x = self.conv2(self.smish(self.conv1(x)))
        return self.smish(x) if self.use_act else x


class TED(nn.Module):
    """입력 H, W는 8의 배수. 출력: [스케일1, 스케일2, 스케일3, 융합] 로짓 (B,1,H,W)"""
    def __init__(self):
        super().__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(1, 32, 48)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = SingleConvBlock(32, 48, 1)
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2)
        self.block_cat = DoubleFusion(3)
        self.apply(weight_init)

    def forward(self, x):
        b1 = self.block_1(x)
        b1_side = self.side_1(b1)
        b2 = self.block_2(b1)
        b2_down = self.maxpool(b2)
        b3, _ = self.dblock_3([b2_down + b1_side, self.pre_dense_3(b2_down)])
        outs = [self.up_block_1(b1), self.up_block_2(b2), self.up_block_3(b3)]
        outs.append(self.block_cat(torch.cat(outs, dim=1)))
        return outs


# 입력 정규화 (RGB, 0~255 기준). 학습과 웹 추론이 같은 값을 쓰도록 여기서만 정의
RGB_MEAN = (123.68, 116.779, 103.939)


# 융합 출력은 마지막에 Smish를 거치는데 Smish의 최솟값이 약 -0.25라서
# sigmoid를 씌워도 0.4378 밑으로는 못 내려간다. 0~1 전체를 쓰도록 다시 늘려준다.
FUSED_FLOOR = 0.4378


def edge_prob(fused_logit):
    return ((torch.sigmoid(fused_logit) - FUSED_FLOOR) / (1 - FUSED_FLOOR)).clamp(0, 1)


class EdgeNetForExport(nn.Module):
    """웹용: RGB 0~255 (1,3,H,W) → 엣지 확률 0~1 (1,1,H,W). 정규화·확률 변환을 모델 안에 포함"""
    def __init__(self, ted):
        super().__init__()
        self.ted = ted
        self.register_buffer('mean', torch.tensor(RGB_MEAN).view(1, 3, 1, 1))

    def forward(self, rgb):
        return edge_prob(self.ted(rgb - self.mean)[-1])
