"""用虚构的框跑一遍每个检测概念。
没有网络，没有数据集——先理解输出格式。

运行：  python demo.py
"""
from boxes import iou, normalized_to_pixels, grid_cell_of, cell_offset
from nms import nms
from eval import average_precision, map_50_95

S = 20          # 20x20 网格，和简易 YOLO 课程一样
IMG = 640

# --- 1. 真实标注：图 0 里一个圆（YOLO 标签）-------------------
gt_label = (0.30, 0.30, 40 / IMG, 40 / IMG)   # cx, cy, w, h，归一化
gt_box = normalized_to_pixels(gt_label, IMG)
print('真实框（像素）:', [round(v, 1) for v in gt_box])

# --- 2. 网络对同一物体的原始预测 --------------------
# 一个好框、一个近乎重复的框（同一物体，相邻单元），
# 以及一个低置信度误报。
good = [gt_box[0] + 2, gt_box[1] + 2, gt_box[2] + 2, gt_box[3] + 2, 0.94]
dup = [gt_box[0] + 8, gt_box[1] + 6, gt_box[2] + 8, gt_box[3] + 6, 0.78]
fp = [500, 500, 540, 540, 0.25]
raw = [good, dup, fp]

print('\n好框 vs 真实 IoU:', round(iou(good[:4], gt_box), 3))
print('重复框 vs 真实 IoU:', round(iou(dup[:4], gt_box), 3))

# --- 3. 置信度阈值 + NMS -----------------------------------------
CONF = 0.5
before_nms = [b for b in raw if b[4] > CONF]
kept = nms(before_nms, iou_thresh=0.5)
print(f'\n超过 {CONF} 的原始框: {len(before_nms)} -> NMS 后: {len(kept)}')
print('保留框分数:', kept[0][4])

# --- 4. 平均精度 ---------------------------------------------------
preds = [(0, b[:4], b[4]) for b in raw]
gts = [(0, gt_box)]
print('\nAP@0.5:', round(average_precision(preds, gts, 0.5), 3))
print('mAP@0.5:0.95:', round(map_50_95(preds, gts), 3),
      '（低：误报和弱重复框拉低了它）')

# --- 5. 框在 SxS 网格中的位置 ---------------------------
gx, gy = grid_cell_of(gt_label[0], gt_label[1], S)
ox, oy = cell_offset(gt_label[0], gt_label[1], S)
print(f'\n拥有中心的网格单元: ({gx},{gy}), 单元内偏移 ({ox:.2f},{oy:.2f})')
print('=> 这正是第 09 课 YOLO 头部学习的 (tx, ty)。')

# --- 6. 可选图片 ----------------------------------------------------
try:
    import cv2
    import numpy as np
    bg = cv2.imread('../assets/backgrounds/000000001685.jpg')
    canvas = cv2.resize(bg, (IMG, IMG)) if bg is not None else np.full((IMG, IMG, 3), 30, dtype='uint8')
    for x in range(0, IMG, IMG // S):
        cv2.line(canvas, (x, 0), (x, IMG), (40, 40, 40), 1)
        cv2.line(canvas, (0, x), (IMG, x), (40, 40, 40), 1)
    def draw(box, color, label):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    draw(gt_box, (0, 255, 0), '真实')
    draw(kept[0][:4], (0, 255, 255), f"预测 {kept[0][4]:.2f}")
    cv2.imwrite('primer_demo.png', canvas)
    print('\n已保存 primer_demo.png（绿=真实，黄=预测，淡网格=SxS）。')
except Exception as e:
    print(f'（跳过绘制演示图片: {e}）')
