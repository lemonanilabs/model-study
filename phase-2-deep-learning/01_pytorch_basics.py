"""
=============================================================================
Phase 2-1: PyTorch 基礎 — Tensor / Autograd / Device
=============================================================================

PyTorch 是什麼？
───────────────
  NumPy  + GPU 加速 + 自動微分 (Autograd) = PyTorch Tensor

  你在 Phase 0 學的 NumPy 操作，全部都能用 PyTorch 做，而且：
  1. 可以丟到 GPU 上跑（快 10-100 倍）
  2. 自動幫你算梯度（不用手寫反向傳播）

本檔涵蓋：
  1. Tensor 建立與基本操作
  2. Tensor vs NumPy 對比
  3. GPU 加速
  4. Autograd 基礎
"""

import numpy as np
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# 1. Tensor 建立
# ============================================================================
print("\n" + "=" * 60)
print("1. Tensor 建立 — 和 NumPy 幾乎一樣")
print("=" * 60)

# --- 從 list 建立 ---
t = torch.tensor([1, 2, 3])
print(f"從 list: {t}")

# --- 從 NumPy 轉換 ---
np_arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
t_from_np = torch.from_numpy(np_arr)       # 共享記憶體！
t_to_np = t_from_np.numpy()                # 轉回 NumPy
print(f"NumPy → Tensor: {t_from_np}")

# --- 常用建立函式 ---
zeros = torch.zeros(2, 3)                   # 全 0
ones = torch.ones(2, 3)                     # 全 1
rand = torch.randn(2, 3)                    # 標準常態分佈
arange = torch.arange(0, 10, 2)             # [0, 2, 4, 6, 8]
eye = torch.eye(3)                          # 3×3 單位矩陣

print(f"\nzeros(2,3):\n{zeros}")
print(f"randn(2,3):\n{rand}")

# --- 重要屬性（和 NumPy 一樣！）---
t = torch.randn(3, 4, 5)
print(f"\n形狀 shape:  {t.shape}")           # torch.Size([3, 4, 5])
print(f"維度 ndim:   {t.ndim}")              # 3
print(f"資料型態:    {t.dtype}")              # torch.float32
print(f"所在裝置:    {t.device}")             # cpu (或 cuda:0)
print(f"元素數量:    {t.numel()}")            # 60


# ============================================================================
# 2. Tensor 操作 — NumPy 對照表
# ============================================================================
print("\n" + "=" * 60)
print("2. NumPy vs PyTorch 對照")
print("=" * 60)

A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])

print(f"A + B (逐元素加):\n{A + B}")
print(f"\nA * B (逐元素乘):\n{A * B}")
print(f"\nA @ B (矩陣乘法):\n{A @ B}")        # 也可以用 torch.matmul(A, B)
print(f"\nA.T (轉置):\n{A.T}")

# 對照表
print("""
NumPy vs PyTorch 對照：

  NumPy                    PyTorch
  ──────────────────────────────────────
  np.array([1,2,3])       torch.tensor([1,2,3])
  np.zeros((2,3))         torch.zeros(2,3)
  np.random.randn(2,3)    torch.randn(2,3)
  a.shape                 t.shape
  a.dtype                 t.dtype
  a.reshape(2,3)          t.reshape(2,3)  或 t.view(2,3)
  a @ b                   t @ b
  a.T                     t.T
  np.concatenate           torch.cat
  np.stack                 torch.stack
  a.sum(axis=0)           t.sum(dim=0)     ← axis → dim
  a.mean(axis=1)          t.mean(dim=1)
  np.max(a)               t.max()  或 torch.max(t)
  a.argmax()              t.argmax()
  np.exp(a)               torch.exp(t)
  np.log(a)               torch.log(t)
  a[a > 0]                t[t > 0]
  np.from_numpy           torch.from_numpy
  a = t.numpy()           (Tensor → NumPy)

  最大差異：
  - dim 取代 axis
  - view 取代 reshape（view 要求連續記憶體）
  - Tensor 有 .device 和 .requires_grad
""")


# ============================================================================
# 3. Reshape / View / Squeeze / Unsqueeze
# ============================================================================
print("=" * 60)
print("3. 維度操作 — 深度學習中最常用的操作")
print("=" * 60)

t = torch.arange(12)
print(f"原始: {t}, shape: {t.shape}")

# reshape / view
t1 = t.view(3, 4)
print(f"view(3,4):\n{t1}")

t2 = t.view(3, -1)      # -1 自動計算
print(f"view(3,-1): shape = {t2.shape}")

# squeeze: 移除大小為 1 的維度
t3 = torch.randn(1, 3, 1, 4)
print(f"\nsqueeze 前: {t3.shape}")          # [1, 3, 1, 4]
print(f"squeeze 後: {t3.squeeze().shape}")  # [3, 4]

# unsqueeze: 在指定位置加一個維度
t4 = torch.randn(3, 4)
print(f"\nunsqueeze 前: {t4.shape}")                # [3, 4]
print(f"unsqueeze(0) 後: {t4.unsqueeze(0).shape}")  # [1, 3, 4] — 加 batch 維度
print(f"unsqueeze(2) 後: {t4.unsqueeze(2).shape}")  # [3, 4, 1]

# permute: 交換維度順序
t5 = torch.randn(2, 3, 4)   # (batch, height, width)
t6 = t5.permute(0, 2, 1)    # → (batch, width, height)
print(f"\npermute(0,2,1): {t5.shape} → {t6.shape}")

print("""
常見情境：
  unsqueeze(0) → 把單一樣本加上 batch 維度，才能餵入模型
  squeeze()    → 移除多餘的維度
  view(-1)     → 展平成一維（Flatten）
  permute      → 調整維度順序（如 HWC → CHW）
""")


# ============================================================================
# 4. GPU 加速
# ============================================================================
print("=" * 60)
print("4. GPU 加速")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用裝置: {device}")

# 把 Tensor 移到 GPU
t_cpu = torch.randn(1000, 1000)
t_device = t_cpu.to(device)
print(f"CPU Tensor: {t_cpu.device}")
print(f"Device Tensor: {t_device.device}")

# GPU vs CPU 速度比較
import time

size = 5000
a = torch.randn(size, size, device='cpu')
b = torch.randn(size, size, device='cpu')

start = time.time()
c = a @ b
cpu_time = time.time() - start
print(f"\nCPU 矩陣乘法 ({size}×{size}): {cpu_time:.4f} 秒")

if torch.cuda.is_available():
    a_gpu = a.to('cuda')
    b_gpu = b.to('cuda')
    # 暖機
    _ = a_gpu @ b_gpu
    torch.cuda.synchronize()

    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU 矩陣乘法 ({size}×{size}): {gpu_time:.4f} 秒")
    print(f"GPU 加速: {cpu_time / gpu_time:.1f}x")

print("""
重要規則：
  - CPU 和 GPU 的 Tensor 不能直接運算
  - 模型和資料必須在同一個裝置上
  - 常見寫法：
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      model = model.to(device)
      data = data.to(device)
""")


# ============================================================================
# 5. Autograd 初體驗
# ============================================================================
print("=" * 60)
print("5. Autograd — 自動計算梯度")
print("=" * 60)

# requires_grad=True 告訴 PyTorch：追蹤這個 Tensor 的所有運算，以便算梯度
x = torch.tensor(3.0, requires_grad=True)

# 前向計算
y = x ** 2 + 2 * x + 1    # y = x² + 2x + 1

# 反向傳播（自動算梯度）
y.backward()

# dy/dx = 2x + 2 = 2(3) + 2 = 8
print(f"x = {x.item()}")
print(f"y = x² + 2x + 1 = {y.item()}")
print(f"dy/dx = 2x + 2 = {x.grad.item()}")
print(f"→ 自動算出來了！不用手推公式！")

# --- 多變數 ---
print("\n--- 多變數自動微分 ---")
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x_input = torch.tensor(3.0)

# 模擬一層神經網路
z = w * x_input + b       # z = 2*3 + 1 = 7
loss = (z - 10) ** 2      # loss = (7-10)² = 9

loss.backward()

print(f"z = w*x + b = {z.item()}")
print(f"loss = (z - 10)² = {loss.item()}")
print(f"∂loss/∂w = {w.grad.item()}")   # 2*(z-10)*x = 2*(-3)*3 = -18
print(f"∂loss/∂b = {b.grad.item()}")   # 2*(z-10)*1 = 2*(-3) = -6

# 手算驗證
dz = 2 * (z.item() - 10)
print(f"\n手算驗證:")
print(f"  ∂loss/∂z = 2(z-10) = {dz}")
print(f"  ∂loss/∂w = ∂loss/∂z × ∂z/∂w = {dz} × {x_input.item()} = {dz * x_input.item()}")
print(f"  ∂loss/∂b = ∂loss/∂z × ∂z/∂b = {dz} × 1 = {dz}")

# --- 梯度清零很重要！ ---
print("\n--- 梯度累積陷阱 ---")
w2 = torch.tensor(1.0, requires_grad=True)

for i in range(3):
    y = (w2 * 2).sum()
    y.backward()
    print(f"  第 {i+1} 次 backward: w2.grad = {w2.grad.item()}")
    # 梯度會累積！不清零的話越來越大

print("  → 梯度會累積！每次 backward 前要 zero_grad()")

# 正確做法
w3 = torch.tensor(1.0, requires_grad=True)
for i in range(3):
    if w3.grad is not None:
        w3.grad.zero_()       # 清零！
    y = (w3 * 2).sum()
    y.backward()
    print(f"  清零後第 {i+1} 次: w3.grad = {w3.grad.item()}")


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結")
print("=" * 60)
print("""
PyTorch Tensor 的三大超能力：

  1. 和 NumPy 幾乎相同的 API → 學習成本極低
  2. .to('cuda') → GPU 加速
  3. requires_grad=True + .backward() → 自動算梯度

記住的關鍵：
  - dim 取代 axis
  - 每次 backward() 前要 grad.zero_()
  - CPU/GPU Tensor 不能混用
  - .item() 把單元素 Tensor 轉成 Python 數字
  - .detach() 把 Tensor 從計算圖分離（不再追蹤梯度）

下一步：02_autograd_deep_dive.py — 深入理解 Autograd
""")
