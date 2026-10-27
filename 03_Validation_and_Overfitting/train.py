import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from common import set_seed, get_device, get_mnist_loaders, fit, evaluate
from common.metrics import confusion_matrix, per_class_precision_recall, plot_curves
from model import OverfitNet

# --- 实验旋钮（见 readme 的练习）------------------------------------
TRAIN_SUBSET = 2000   # 很小 -> 大模型会死记；试试 50000
BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001

set_seed(1337)
device = get_device()
print(f'使用设备: {device}')

# 三个集合：训练（留出验证后 5 万）/ 验证（1 万）/ 测试（1 万）。
train_loader, val_loader, test_loader = get_mnist_loaders(
    batch_size=BATCH_SIZE, val_size=10000, augment=False, num_workers=2)

# 只保留 TRAIN_SUBSET 个训练样本，让过拟合快速出现。
train_loader = torch.utils.data.DataLoader(
    Subset(train_loader.dataset, range(TRAIN_SUBSET)),
    batch_size=BATCH_SIZE, shuffle=True)

model = OverfitNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


def main():
    history = fit(model, train_loader, val_loader, criterion, optimizer,
                  device, epochs=EPOCHS)
    plot_curves(history, save_path='results.png')

    # 最终的诚实数字：测试集只用一次。
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f'\n测试准确率: {test_acc:.2f}%')

    cm = confusion_matrix(preds, labels, num_classes=10)
    precision, recall, f1 = per_class_precision_recall(cm)
    print(f'宏平均精确率 {precision.mean():.3f} | 召回率 {recall.mean():.3f} '
          f'| F1 {f1.mean():.3f}')

    # 最常见的混淆（哪些数字被误认为哪些）。
    pairs = []
    for t in range(10):
        for p in range(10):
            if t != p and cm[t, p] > 0:
                pairs.append((int(cm[t, p]), t, p))
    print('主要混淆（真实 -> 预测）:')
    for n, t, p in sorted(pairs, reverse=True)[:5]:
        print(f'  {t} -> {p}: {n} 次')


if __name__ == '__main__':
    main()
