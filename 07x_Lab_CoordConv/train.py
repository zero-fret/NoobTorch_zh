import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from common import set_seed, get_device, get_mnist_loaders, fit, evaluate
from model import CoordCNN

# 两个实验变量。编辑这些行然后重跑：
#   一次 USE_COORDS=False，一次 USE_COORDS=True，比较结果。
USE_COORDS = False        # True = CoordConv（实验组），False = 对照组
ACTIVATION = 'leaky_relu' # 'relu' 或 'leaky_relu'

BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001

set_seed(1337)
device = get_device()
print(f'使用设备: {device}')

train_loader, val_loader, test_loader = get_mnist_loaders(
    batch_size=BATCH_SIZE, val_size=10000, augment=True, num_workers=4)

model = CoordCNN(use_coords=USE_COORDS, activation=ACTIVATION).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

if __name__ == '__main__':
    print(f'坐标={USE_COORDS} 激活函数={ACTIVATION}')
    fit(model, train_loader, val_loader, criterion, optimizer,
        device, epochs=EPOCHS, scheduler=scheduler)
    _, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    print(f'测试准确率: {test_acc:.2f}%')
