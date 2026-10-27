import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from common import set_seed, get_device, get_mnist_loaders
from model import SimpleNet

# 本节课刻意展示完整的训练循环。从第 02 课开始我们复用 common.engine.fit，
# 这样你不用每次都读这 40 行代码。

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

set_seed(1337)
device = get_device()
print(f'使用设备: {device}')

# 这里故意设 val_size=0：第 01 课只有训练 + 测试。
# 第 03 课引入验证集并解释为什么需要它。
train_loader, _, test_loader = get_mnist_loaders(
    batch_size=BATCH_SIZE, val_size=0, augment=False, num_workers=2)

model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()                 # log-softmax + NLL 合在一个模块里
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train():
    for epoch in range(EPOCHS):
        model.train()                             # 启用 dropout/batchnorm 更新
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()                 # 清掉上一批的梯度
            outputs = model(images)               # 前向传播
            loss = criterion(outputs, labels)
            loss.backward()                       # 反向传播：填充 .grad
            optimizer.step()                      # 更新参数
            running_loss += loss.item()
        print(f'第 [{epoch+1}/{EPOCHS}] 轮, 损失: {running_loss/len(train_loader):.4f}')


def test():
    model.eval()                                  # 关闭 dropout，冻结 batchnorm
    correct = total = 0
    with torch.no_grad():                         # 不算梯度 -> 省内存
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'测试准确率: {100*correct/total:.2f}%')


if __name__ == '__main__':
    train()
    test()
