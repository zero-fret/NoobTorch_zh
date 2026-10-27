import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from common import set_seed, get_device, get_cifar10_loaders, fit, evaluate

BATCH_SIZE = 64
EPOCHS = 3                 # 冻结骨干训练很快；完整微调设约 10
LR = 1e-3

set_seed(1337)
device = get_device()
print(f'使用设备: {device}')

# torchvision 的 ImageNet ResNet18 有 7x7 步长 2 卷积加 maxpool，
# 是为 ~224px 输入设计的。喂给它 32/64px 的 CIFAR 图会丢掉大部分信号，
# 浪费预训练的好处——所以这里用 224px，
# 加上骨干网络期望的 ImageNet 归一化（norm='imagenet')。
# 这里不做空间增强：本节课讲迁移学习，不是数据增强。
train_loader, val_loader, test_loader = get_cifar10_loaders(
    batch_size=BATCH_SIZE, val_size=5000, augment=None,
    resize=224, num_workers=4, norm='imagenet')

# --- 迁移学习第 1 步：用在 ImageNet 上训练的骨干网络 ----------
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# --- 第 2 步：冻结所有参数，然后替换最后一层 ----------------
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)   # 只有这新层训练
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

if __name__ == '__main__':
    print('特征提取模式：骨干网络冻结，训练新头部。')
    fit(model, train_loader, val_loader, criterion, optimizer,
        device, epochs=EPOCHS)
    _, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    print(f'测试准确率（冻结骨干）: {test_acc:.2f}%')

    torch.save(model.state_dict(), 'cifar10_resnet18.pth')
    print('模型已保存到 cifar10_resnet18.pth')

    # --- 第 3 步（可选）：完整微调 --------------------------------
    # 解冻后用小学习率重新优化整个网络几个轮次，
    # 让预训练特征被调整而不是被摧毁：
    #
    # for p in model.parameters():
    #     p.requires_grad = True
    # optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # fit(model, train_loader, val_loader, criterion, optimizer, device, epochs=5)
