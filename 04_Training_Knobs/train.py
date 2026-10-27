import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from common import set_seed, get_device, get_mnist_loaders, fit, evaluate
from model import KnobCNN

# ===== 四个旋钮。一次翻一个，比较验证准确率。=====
USE_AUGMENTATION = True     # 随机旋转/平移/缩放 -> 免费数据
USE_SCHEDULER = True        # 余弦学习率衰减
USE_WEIGHT_DECAY = True     # 权重上的 L2 惩罚
ACTIVATION = 'leaky_relu'   # 'relu' 或 'leaky_relu'
# =================================

BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
WEIGHT_DECAY = 1e-4 if USE_WEIGHT_DECAY else 0.0
NUM_WORKERS = 4

set_seed(1337)
device = get_device()
print(f'使用设备: {device}')

train_loader, val_loader, test_loader = get_mnist_loaders(
    batch_size=BATCH_SIZE, val_size=10000,
    augment=USE_AUGMENTATION, num_workers=NUM_WORKERS)

model = KnobCNN(activation=ACTIVATION).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS) if USE_SCHEDULER else None

if __name__ == '__main__':
    print(f'数据增强={USE_AUGMENTATION} 调度器={USE_SCHEDULER} '
          f'权重衰减={USE_WEIGHT_DECAY} 激活函数={ACTIVATION}')
    history = fit(model, train_loader, val_loader, criterion, optimizer,
                  device, epochs=EPOCHS, scheduler=scheduler)
    _, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    print(f'最终测试准确率: {test_acc:.2f}%')
