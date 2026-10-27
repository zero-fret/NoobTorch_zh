import torch.nn as nn


class OverfitNet(nn.Module):
    """故意过大的全连接网络。用少量训练样本时它可以死记硬背，
    让过拟合这节课一目了然。"""
    def __init__(self, hidden=256):
        super(OverfitNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
