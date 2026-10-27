import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """一个标准卷积拆成两个廉价卷积：

    深度卷积：每个输入通道一个滤波器  (groups=in_channels)
    逐点卷积：1x1 卷积混合通道并投影到 out_channels
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3,
                                   stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class SimpleYOLO_DSConv(nn.Module):
    """与第 09 课相同的检测器，但每个下采样块都是深度可分离的。
    第一个入口卷积保持标准（3 个输入通道）。"""
    def __init__(self, S=20):
        super().__init__()
        self.S = S
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01, inplace=True),
            DepthwiseSeparableConv(16, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            nn.Conv2d(512, 5, 1),
        )

    def forward(self, x):
        return self.features(x)
