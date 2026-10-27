"""加载训练好的模型并在数据上运行——这就是部署的本质。

编辑下面两个常量：
    IMAGE_PATH = None   -> 分类第一张 CIFAR 测试图
    IMAGE_PATH = 'cat.jpg' -> 分类你自己的图片
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
from common.data import IMAGENET_MEAN, IMAGENET_STD, DATA_ROOT

# 指向你自己的图片，或留 None 来分类一张 CIFAR 测试图。
IMAGE_PATH = None
WEIGHTS_PATH = 'cifar10_resnet18.pth'   # 或 'cifar10_resnet18_finetuned.pth'

CLASSES = ['飞机', '汽车', '鸟', '猫', '鹿',
           '狗', '青蛙', '马', '船', '卡车']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(weights_path):
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval().to(device)
    return model


def predict(model, pil_image):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tf(pil_image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    conf, idx = probs.max(0)
    return CLASSES[idx], conf.item(), probs


if __name__ == '__main__':
    model = build_model(WEIGHTS_PATH)

    if IMAGE_PATH is not None:
        img = Image.open(IMAGE_PATH)
    else:
        test_ds = torchvision.datasets.CIFAR10(
            DATA_ROOT, train=False, download=True)
        img, label = test_ds[0]
        print(f'(CIFAR 测试图，真实标签: {CLASSES[label]})')

    name, conf, probs = predict(model, img)
    print(f'预测: {name} ({conf:.1%})')
    top3 = probs.topk(3)
    for p, i in zip(top3.values, top3.indices):
        print(f'  {CLASSES[i]:>12}: {p:.1%}')
