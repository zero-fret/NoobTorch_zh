"""迁移学习第 2 步：微调冻结的探针。

冻结头部（train.py）只达到 ~80%，因为线性头部只训练了 3 轮，
骨干的 ImageNet 特征从未适配 CIFAR。这里我们：

  1. 加载冻结探针的权重（cifar10_resnet18.pth），
  2. 保持低层入口 + layer1/layer2 冻结（边缘/纹理原样迁移），
     但解冻 layer3、layer4 和新头部，
  3. 用小的骨干学习率和大的头部学习率训练，加轻度增强。

用法：
    python finetune.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from common import set_seed, get_device, get_cifar10_loaders, fit, evaluate

BATCH_SIZE = 32
EPOCHS = 12
BACKBONE_LR = 2e-4     # 小：适配预训练特征，不要摧毁它们
HEAD_LR = 1e-3
FROZEN_PREFIXES = ('conv1', 'bn1', 'layer1', 'layer2')   # 保持这些冻结

set_seed(1337)
device = get_device()
print(f'使用设备: {device}')

# 与 train.py 相同的预处理：224px + ImageNet 统计量。既然在适配特征了，
# 加轻度翻转/裁剪增强。
train_loader, val_loader, test_loader = get_cifar10_loaders(
    batch_size=BATCH_SIZE, val_size=5000, augment='basic',
    resize=224, num_workers=4, norm='imagenet')

# 构建相同结构并加载冻结探针的检查点。
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('cifar10_resnet18.pth',
                                 map_location=device, weights_only=True))
model.to(device)

# 先冻结所有，然后解冻高层 + 头部。
for p in model.parameters():
    p.requires_grad = False
for name, p in model.named_parameters():
    if not name.startswith(FROZEN_PREFIXES):
        p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'微调可训练参数: {trainable:,} (入口/layer1/layer2 冻结)')

# 差异化学习率：预训练骨干慢，新头部快。
backbone_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith('fc.')]
head_params = [p for n, p in model.named_parameters()
               if p.requires_grad and n.startswith('fc.')]
optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': BACKBONE_LR, 'weight_decay': 1e-4},
    {'params': head_params,     'lr': HEAD_LR,     'weight_decay': 1e-4},
])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

if __name__ == '__main__':
    fit(model, train_loader, val_loader, criterion, optimizer, device,
        epochs=EPOCHS, scheduler=scheduler)
    _, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    print(f'测试准确率（微调后）: {test_acc:.2f}%')

    torch.save(model.state_dict(), 'cifar10_resnet18_finetuned.pth')
    print('模型已保存到 cifar10_resnet18_finetuned.pth')
