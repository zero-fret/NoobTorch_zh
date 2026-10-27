import torch.nn as nn


class SimpleNet(nn.Module):
    """用于 MNIST 的三层全连接网络。

    28x28 图像 -> 展平 -> 128 -> 64 -> 10 类分数。
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)   # 784 个输入 -> 128 个神经元
        self.fc2 = nn.Linear(128, 64)        # 128 -> 64
        self.fc3 = nn.Linear(64, 10)         # 64 -> 10 个数字类别
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)            # 展平：(B, 1, 28, 28) -> (B, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)                      # 原始分数（logit）；CrossEntropyLoss 内部加 softmax
        return x
