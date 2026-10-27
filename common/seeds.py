"""可复现性工具，每节课共享。

相同的随机种子 -> 相同的数据顺序 / 权重初始化 -> 每次运行结果一致。
在每个训练脚本开头调用一次 set_seed(1337)。
"""
import random
import numpy as np
import torch


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """有 CUDA 就用 GPU，否则用 CPU。"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
