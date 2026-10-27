"""目标检测的边界框几何。这里没有神经网络。"""

# 本课使用的框格式：
#  YOLO 标签（归一化）：(cx, cy, w, h)，都在 [0,1]，图像尺寸的比例
#  像素框（xyxy）：      (x1, y1, x2, y2)，左上角/右下角
#  像素框（cxcywh）：    (cx, cy, w, h)，中心点和宽高


def cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def normalized_to_pixels(box, img_size=640):
    """(cx, cy, w, h) 归一化 -> (x1, y1, x2, y2) 像素坐标。"""
    cx, cy, w, h = box
    return cxcywh_to_xyxy((cx * img_size, cy * img_size,
                           w * img_size, h * img_size))


def iou(box_a, box_b):
    """两个 xyxy 框的交并比。1 = 完全相同，0 = 不重叠。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def grid_cell_of(cx, cy, S):
    """哪个网格单元拥有归一化中心 (cx, cy)。返回 (gx, gy)。"""
    gx = min(int(cx * S), S - 1)
    gy = min(int(cy * S), S - 1)
    return gx, gy


def cell_offset(cx, cy, S):
    """中心相对于其单元左上角的偏移，在 [0, 1) 中。
    这正是 YOLO 头部学习的 tx, ty。"""
    gx, gy = grid_cell_of(cx, cy, S)
    return cx * S - gx, cy * S - gy


if __name__ == '__main__':
    # 一个中心在 (320, 320)、宽高 40px 的框，在 640 图像上。
    label = (0.5, 0.5, 40 / 640, 40 / 640)
    xyxy = normalized_to_pixels(label)
    print('归一化 -> 像素:', xyxy)

    # 相同框 -> IoU 1；偏移 10px 的孪生 -> IoU 更小。
    twin = (xyxy[0] + 10, xyxy[1], xyxy[2] + 10, xyxy[3])
    print('IoU 相同:', round(iou(xyxy, xyxy), 3))
    print('IoU 偏移 10px:', round(iou(xyxy, twin), 3))

    # 网格分配。
    gx, gy = grid_cell_of(0.5, 0.5, S=20)
    off_x, off_y = cell_offset(0.5, 0.5, S=20)
    print(f'中心 (0.5,0.5) -> 单元 ({gx},{gy}), 偏移 ({off_x:.2f},{off_y:.2f})')
