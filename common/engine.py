"""训练循环，只写一次。

每个分类课都调用 fit()，这样 readme 可以聚焦于单个新概念，
而不用反复解释同样的 40 行样板代码。
"""
import torch


def train_one_epoch(model, loader, criterion, optimizer, device):
    """返回一轮遍历数据后的 (平均损失, 准确率百分比)。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """返回 (平均损失, 准确率百分比, 预测, 标签)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, targets = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        pred = outputs.argmax(1)
        correct += (pred == labels).sum().item()
        total += images.size(0)
        preds.append(pred.cpu())
        targets.append(labels.cpu())
    return total_loss / total, 100.0 * correct / total, torch.cat(preds), torch.cat(targets)


def fit(model, train_loader, val_loader, criterion, optimizer, device,
        epochs, scheduler=None, verbose=True):
    """训练 epochs 轮，返回包含各曲线的历史字典。

    history = {'train_loss': [...], 'train_acc': [...],
               'val_loss':   [...], 'val_acc':   [...], 'lr': [...]}
    """
    history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr')}
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()
        for k, v in (('train_loss', tr_loss), ('train_acc', tr_acc),
                     ('val_loss', va_loss), ('val_acc', va_acc), ('lr', lr)):
            history[k].append(v)
        if verbose:
            print(f'第 {epoch:3d}/{epochs} 轮 | '
                  f'训练损失 {tr_loss:.4f} 准确率 {tr_acc:5.2f}% | '
                  f'验证损失 {va_loss:.4f} 准确率 {va_acc:5.2f}% | 学习率 {lr:.5f}')
    return history
