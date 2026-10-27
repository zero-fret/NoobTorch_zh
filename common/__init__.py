"""NoobTorch 1.0 的共享工具模块。每节课导入这些函数，而不是复制粘贴样板代码，
这样每节课的新概念才能突出显示。"""
from .seeds import set_seed, get_device
from .data import get_mnist_loaders, get_cifar10_loaders
from .engine import train_one_epoch, evaluate, fit
from .metrics import confusion_matrix, per_class_precision_recall, plot_curves
