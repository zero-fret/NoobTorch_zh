# 10 — 高效 YOLO（深度可分离卷积）
与第 09 课相同的任务，现在是多目标和更小更快的网络。

## 核心概念
一个标准的 3×3 卷积，32→64 通道，需要 `3×3×32×64 = 18,432` 个权重。**深度可分离卷积**把它拆成：
- **深度卷积**——每个通道一个 3×3 滤波器：`3×3×32 = 288` 个权重；
- **逐点卷积**——一个 1×1 卷积混合通道：`32×64 = 2,048` 个权重。

总共 `2,336` 对比 `18,432`——少了约 **87%** 的权重，在简单任务上几乎不损失精度。这是 MobileNet 和现代 YOLO 变体的构建块。

## 其他不同
- **每张图四个圆**（每个象限一个），所以数据集和损失现在处理多目标——每个一个正样本单元。
- 置信度不平衡用 `BCEWithLogitsLoss(pos_weight=num_neg/num_pos)` 处理，而不是固定的手工权重——和 focal loss 同一个想法，但更简单。
- 宽高直接回归（不用 sigmoid）；解码器把负值截断到零。

## 与第 09 课的区别
唯一的架构变化是卷积块（`DepthwiseSeparableConv`）。编码、NMS、AP、数据管道都复用。

## 运行
```
python dataset_generator.py
python train.py
python test.py
```

## 基准测试（这节课的重点）
在相同数据上比较两个检测器：
```
python -c "from model import SimpleYOLO_DSConv as M; print(sum(p.numel() for p in M().parameters()))"
```
然后在第 09 课对 `SimpleYOLO` 做同样的事，并在 `(1,3,640,640)` 张量上计一次前向传播时间：
```python
import time, torch
x = torch.randn(1, 3, 640, 640)
t = time.time(); model(x); print(time.time() - t)
```
记录两者的参数量和 AP——效率永远是权衡，没有免费的午餐。

## 检查点 10
- [ ] 我能计算深度卷积和逐点卷积的参数量。
- [ ] 我能解释为什么成本下降约 87%。
- [ ] 我能在参数量、速度和精度上比较两个模型。

→ 下一节：**10x_Lab_CoordConv_YOLO**（选学）。我们在分类上证明了消融方法；现在在检测器上跑同一个实验。
