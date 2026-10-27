"""MNIST 和 CIFAR-10 的共享数据集构建器。

所有课程都下载到同一个共享文件夹（../data），不需要下载五次数据集。
每节课从这里导入，而不是复制粘贴 40 行 DataLoader 代码——
每节课的*新概念*才能突出显示。

每个构建器返回 (train_loader, val_loader, test_loader)：
  - train: 打乱，数据增强（可选）
  - val:   从训练集中留出，绝不用于训练，用于调参
  - test:  只在最后报告数字时碰一次
"""
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, 'data')

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
# ImageNet 统计量：在第 07 课使用 ImageNet 预训练骨干网络时，
# 它期望输入按照训练时的方式做归一化。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _split_indices(n_total, val_size, seed):
    """确定性打乱 -> 训练索引和验证索引。"""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    return perm[val_size:], perm[:val_size]


def get_mnist_loaders(batch_size=64, val_size=10000, augment=False,
                      num_workers=2, data_root=DATA_ROOT, seed=1337):
    train_tf = [
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ] if augment else []
    train_tf = transforms.Compose(train_tf + [
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])

    train_full = torchvision.datasets.MNIST(data_root, train=True, download=True, transform=train_tf)
    val_base = torchvision.datasets.MNIST(data_root, train=True, download=True, transform=eval_tf)
    test_ds = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=eval_tf)

    if val_size and val_size > 0:
        train_idx, val_idx = _split_indices(len(train_full), val_size, seed)
        train_ds, val_ds = Subset(train_full, train_idx), Subset(val_base, val_idx)
    else:
        train_ds, val_ds = train_full, None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def get_cifar10_loaders(batch_size=128, val_size=5000, augment='basic',
                        num_workers=4, resize=None, data_root=DATA_ROOT, seed=1337,
                        norm='cifar'):
    """augment: None、'basic'（翻转 + 裁剪）或 'strong'（+ RandAugment、RandomErasing）。
    norm: 'cifar' 用于从零训练的网络，'imagenet' 用于 ImageNet 预训练骨干网络。
    """
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if norm == 'imagenet' else (CIFAR_MEAN, CIFAR_STD)
    crop = resize if resize is not None else 32
    train_steps = []
    if augment == 'strong':
        train_steps += [
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(crop, padding=4),
        ]
    elif augment == 'basic':
        train_steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(crop, padding=4),
        ]
    train_steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if augment == 'strong':
        train_steps += [transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3))]

    eval_steps = []
    if resize is not None:
        eval_steps.append(transforms.Resize(resize))
    eval_steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    train_tf = transforms.Compose(([transforms.Resize(resize)] if resize is not None else []) + train_steps)
    eval_tf = transforms.Compose(eval_steps)

    train_full = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
    val_base = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=eval_tf)
    test_ds = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=eval_tf)

    if val_size and val_size > 0:
        train_idx, val_idx = _split_indices(len(train_full), val_size, seed)
        train_ds, val_ds = Subset(train_full, train_idx), Subset(val_base, val_idx)
    else:
        train_ds, val_ds = train_full, None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
