import torch

# --- 张量 + 自动微分 -------------------------------------------------
x = torch.tensor([2.0], requires_grad=True)  # requires_grad=True：PyTorch 将梯度记录到 x.grad。x 必须是浮点数。
print(x.grad)  # None——还没调用 backward()

y = x ** 2 + 3 * x + 1  # y = f(x)，这里的每个操作都被记录
y.backward()            # 自动求导：x.grad = dy/dx
print(x.grad)           # dy/dx = 2*x + 3 = 7

x.grad.zero_()          # 清零梯度（原地操作；注意尾部下划线）
print(x.grad)           # tensor([0.])

# --- 保存 / 加载 --------------------------------------------------------
# 保存训练好的模型就是把它的参数（state_dict）保存到磁盘。
# 这里还没有模型，所以我们只保存一个张量——对模型的调用方式完全相同：
# torch.save(model.state_dict(), 'model.pth')
torch.save({'x': x, 'grad': x.grad}, 'demo.pth')
loaded = torch.load('demo.pth', weights_only=False)
print('loaded:', loaded)

# 预期输出：
# None
# tensor([7.])
# tensor([0.])
# loaded: {'x': tensor([2.], requires_grad=True), 'grad': tensor([0.])}
