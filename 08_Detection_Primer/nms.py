"""非极大值抑制：每个物体只保留一个框。"""


def nms(boxes, iou_thresh):
    """boxes: [x1, y1, x2, y2, score] 列表。
    贪心保留最高分框，删除所有与它重叠超过 iou_thresh 的低分框
    （同一物体的重复框）。
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    from boxes import iou
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if iou(best[:4], b[:4]) < iou_thresh]
    return keep
