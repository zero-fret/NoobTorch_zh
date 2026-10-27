import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from common import set_seed, get_device, get_cifar10_loaders, fit, evaluate
from model import BNCNN

USE_BN = True          # 翻成 False 就在无 BatchNorm 下训练同一个网络
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 5e-4

set_seed(1337)
device = get_device()
print(f'使用设备: {device}')

train_loader, val_loader, test_loader = get_cifar10_loaders(
    batch_size=BATCH_SIZE, val_size=5000, augment='basic', num_workers=4)

model = BNCNN(use_bn=USE_BN).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

if __name__ == '__main__':
    print(f'BatchNorm 已启用: {USE_BN}')
    history = fit(model, train_loader, val_loader, criterion, optimizer,
                  device, epochs=EPOCHS, scheduler=scheduler)
    _, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    print(f'最终测试准确率: {test_acc:.2f}%')

    # 健康检查：训练后激活看起来怎么样？
    probe_images, _ = next(iter(train_loader))
    print('\n激活报告（均值, 标准差, 死亡 ReLU 比例）:')
    model.activation_report(probe_images, device)
