import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utils import CircleDataset, S, decode_predictions, nms, compute_ap, decode_gt
from model import SimpleYOLO_DSConv

BATCH = 32
EPOCHS = 60
LR = 0.001
COORD_W = 10.0      # 每张图 4 个目标；强调定位


def forward_loss(model, mse, imgs, targets, masks):
    out = model(imgs)
    pred_tx = torch.sigmoid(out[:, 0:1])
    pred_ty = torch.sigmoid(out[:, 1:2])
    pred_tw, pred_th = out[:, 2:3], out[:, 3:4]
    conf_logits = out[:, 4:5]
    conf_target = targets[:, 4:5]

    coord = mse(pred_tx, targets[:, 0:1]) + mse(pred_ty, targets[:, 1:2]) \
          + mse(pred_tw, targets[:, 2:3]) + mse(pred_th, targets[:, 3:4])
    mask = masks.unsqueeze(1)
    coord = (coord * mask.float()).sum() / (mask.sum() + 1e-6) * COORD_W

    # 平衡严重不平衡：每个目标 1 个正样本单元 vs 约 396 个负样本。
    num_pos = mask.sum().clamp(min=1)
    num_neg = float(S * S * imgs.size(0)) - num_pos
    pos_weight = torch.tensor([(num_neg / num_pos).item()], device=imgs.device)
    conf_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(conf_logits, conf_target)
    return coord + conf_loss


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    preds, gts = [], []
    for b, (imgs, targets, masks) in enumerate(val_loader):
        imgs = imgs.to(device)
        for i in range(imgs.size(0)):
            img_id = b * imgs.size(0) + i
            boxes = decode_predictions(model(imgs[i:i+1]), S, conf_thresh=0.1)
            for box in nms(boxes, 0.5):
                preds.append((img_id, box[:4], box[4], 0))
            gts.extend(decode_gt(targets[i], masks[i], img_id, S))
    if preds and gts:
        ap50 = compute_ap(preds, gts, 0.5)
        ap5095 = np.mean([compute_ap(preds, gts, t) for t in np.arange(0.5, 1.0, 0.05)])
        print(f'  验证 AP50={ap50:.3f} mAP50-95={ap5095:.3f}')


def main():
    # 在 Windows 上，DataLoader 工作进程（num_workers>0）会重新导入此模块；
    # 一次性设置和循环放在 __main__ 守卫下。
    random.seed(1337); np.random.seed(1337); torch.manual_seed(1337)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    train_loader = DataLoader(CircleDataset('train'), batch_size=BATCH, shuffle=True,
                              num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(CircleDataset('val'), batch_size=BATCH,
                            shuffle=False, num_workers=4)

    model = SimpleYOLO_DSConv(S=S).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)
    mse = nn.MSELoss(reduction='none')

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for imgs, targets, masks in train_loader:
            imgs, targets, masks = imgs.to(device), targets.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = forward_loss(model, mse, imgs, targets, masks)
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        print(f'第 {epoch+1}/{EPOCHS} 轮 loss={total/len(train_loader)*100:.3f}')
        if (epoch + 1) % 10 == 0:
            validate(model, val_loader, device)

    torch.save(model.state_dict(), 'model.pth')
    print('已保存 model.pth')


if __name__ == '__main__':
    main()
