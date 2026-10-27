"""超越单一准确率数字的评估工具：
混淆矩阵、逐类精确率/召回率、学习曲线绘图。
"""
import torch


def confusion_matrix(preds, labels, num_classes):
    """cm[真实标签, 预测标签] = 计数。"""
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(labels.tolist(), preds.tolist()):
        cm[t, p] += 1
    return cm


def per_class_precision_recall(cm):
    """返回每个类别的 (精确率, 召回率, F1) 张量。"""
    tp = cm.diag().float()
    precision = tp / (cm.sum(dim=0).float() + 1e-12)
    recall = tp / (cm.sum(dim=1).float() + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1


def plot_curves(history, save_path='results.png'):
    """保存损失/准确率曲线（训练 vs 验证）。需要 matplotlib；
    如果没安装就打印提示而不是崩溃。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('未安装 matplotlib（运行 `pip install matplotlib`）——跳过绘图。')
        return

    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history['train_loss'], label='训练')
    ax1.plot(epochs, history['val_loss'], label='验证')
    ax1.set_title('损失'); ax1.set_xlabel('轮次'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history['train_acc'], label='训练')
    ax2.plot(epochs, history['val_acc'], label='验证')
    ax2.set_title('准确率 (%)'); ax2.set_xlabel('轮次'); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f'学习曲线已保存到 {save_path}')
