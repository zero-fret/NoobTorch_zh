"""检测的平均精度（单类，不训练）。

predictions: [(img_id, [x1, y1, x2, y2], score), ...]
groundtruths: [(img_id, [x1, y1, x2, y2]), ...]
"""
import numpy as np


def average_precision(predictions, groundtruths, iou_thresh=0.5):
    """按置信度排序预测；如果它在同一图上匹配到一个尚未匹配的真实框
    （IoU 超过阈值），则为 TP，否则为 FP。
    AP 是精确率-召回率曲线下的面积（VOC 风格单调插值），
    在召回率直接跳到 1 时也稳健。"""
    if not predictions or not groundtruths:
        return 0.0

    preds = sorted(predictions, key=lambda p: p[2], reverse=True)
    from boxes import iou
    matched = {i: False for i in range(len(groundtruths))}
    tp, fp = [], []

    for img_id, box, _ in preds:
        best_iou, best_idx = 0.0, -1
        for i, (g_img, g_box) in enumerate(groundtruths):
            if g_img == img_id:
                v = iou(box, g_box)
                if v > best_iou:
                    best_iou, best_idx = v, i
        if best_iou >= iou_thresh and best_idx >= 0 and not matched[best_idx]:
            tp.append(1); fp.append(0); matched[best_idx] = True
        else:
            tp.append(0); fp.append(1)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / len(groundtruths)
    return _ap_from_curve(precision, recall)


def _ap_from_curve(precision, recall):
    """VOC 风格 PR 曲线下面积：在 recall=0 处前置哨兵，
    让精确率单调不增，然后在召回率步长上求和。"""
    mrec = np.concatenate(([0.0], recall))
    mpre = np.concatenate(([1.0], precision))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))


def map_50_95(predictions, groundtruths):
    """COCO 风格，在 IoU 阈值 0.5..0.95 上取平均。"""
    thresholds = np.arange(0.5, 1.0, 0.05)
    return float(np.mean([average_precision(predictions, groundtruths, t)
                          for t in thresholds]))
