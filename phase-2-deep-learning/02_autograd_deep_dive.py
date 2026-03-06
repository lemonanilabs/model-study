"""
=============================================================================
Phase 2-2: Autograd 深入 — 手刻 vs 自動微分
=============================================================================

Phase 0 你手寫了 Chain Rule 的反向傳播。
PyTorch 的 Autograd 做的就是同一件事，只是自動化了。

本檔用同一個網路，分別用 NumPy 手刻 和 PyTorch Autograd 實作，
讓你看到 Autograd 到底幫你省了什麼。

本檔涵蓋：
  1. 計算圖 (Computational Graph) 的概念
  2. 同一個模型：NumPy 手刻反向傳播 vs PyTorch Autograd
  3. 梯度檢查 (Gradient Checking)
  4. requires_grad / detach / no_grad 的用法
"""

import numpy as np
import torch

# ============================================================================
# 1. 計算圖 (Computational Graph)
# ============================================================================
print("=" * 60)
print("1. 計算圖 — Autograd 的核心資料結構")
print("=" * 60)

print("""
每次你用 Tensor 做運算，PyTorch 都在背後記錄一張「計算圖」：

  x ──→ [×w] ──→ z ──→ [ReLU] ──→ a ──→ [×v] ──→ y ──→ [MSE] ──→ loss
  w ──↗                                   v ──↗         target ──↗

  .backward() 就是沿著這張圖，從 loss 往回走，用 Chain Rule 算每個節點的梯度。

  這就是為什麼叫「計算圖 (Computational Graph)」和「反向傳播 (Backpropagation)」。
""")

# 觀察計算圖
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y * 3 + 1

print(f"x = {x}")
print(f"y = x² = {y}")
print(f"z = 3y + 1 = {z}")
print(f"y.grad_fn = {y.grad_fn}")       # PowBackward0
print(f"z.grad_fn = {z.grad_fn}")       # AddBackward0
print("→ grad_fn 記錄了「這個 Tensor 是怎麼算出來的」")


# ============================================================================
# 2. 同一個網路：NumPy vs PyTorch
# ============================================================================
print("\n" + "=" * 60)
print("2. 完整對比：NumPy 手刻 vs PyTorch Autograd")
print("=" * 60)

# 網路結構：
#   輸入 (4,) → Linear(4→8) → ReLU → Linear(8→3) → Softmax → CE Loss
# 用同一組初始權重，驗證兩者的梯度一致

np.random.seed(42)

# 初始化權重（共用）
W1_init = np.random.randn(4, 8).astype(np.float32) * 0.1
b1_init = np.zeros(8, dtype=np.float32)
W2_init = np.random.randn(8, 3).astype(np.float32) * 0.1
b2_init = np.zeros(3, dtype=np.float32)

# 輸入資料
x_data = np.random.randn(4).astype(np.float32)
target = 1   # 正確類別

# ─────────────────────────────────────────
# 方法 A：NumPy 手刻
# ─────────────────────────────────────────
print("\n--- NumPy 手刻反向傳播 ---")

# 複製權重
W1_np = W1_init.copy()
b1_np = b1_init.copy()
W2_np = W2_init.copy()
b2_np = b2_init.copy()

# 前向傳播
z1 = x_data @ W1_np + b1_np          # (8,)
a1 = np.maximum(0, z1)                # ReLU
z2 = a1 @ W2_np + b2_np              # (3,)
exp_z2 = np.exp(z2 - z2.max())
softmax = exp_z2 / exp_z2.sum()       # (3,)
loss_np = -np.log(softmax[target])

print(f"  前向: z1.shape={z1.shape}, a1.shape={a1.shape}, z2.shape={z2.shape}")
print(f"  Softmax: {softmax.round(4)}")
print(f"  Loss: {loss_np:.6f}")

# 反向傳播（手刻 Chain Rule）
# ∂L/∂z2: Softmax + CE 的梯度有個漂亮的公式
dz2 = softmax.copy()
dz2[target] -= 1                       # (3,) — softmax - one_hot

# ∂L/∂W2 = a1ᵀ × dz2
dW2_np = np.outer(a1, dz2)            # (8, 3)
db2_np = dz2                           # (3,)

# ∂L/∂a1 = dz2 × W2ᵀ
da1 = dz2 @ W2_np.T                   # (8,)

# ∂L/∂z1 = da1 × ReLU'(z1)
dz1 = da1 * (z1 > 0).astype(float)    # (8,)  ReLU 導數：z1>0 時為 1，否則為 0

# ∂L/∂W1 = xᵀ × dz1
dW1_np = np.outer(x_data, dz1)        # (4, 8)
db1_np = dz1                           # (8,)

print(f"  dW1 range: [{dW1_np.min():.6f}, {dW1_np.max():.6f}]")
print(f"  dW2 range: [{dW2_np.min():.6f}, {dW2_np.max():.6f}]")

# ─────────────────────────────────────────
# 方法 B：PyTorch Autograd
# ─────────────────────────────────────────
print("\n--- PyTorch Autograd ---")

W1_pt = torch.tensor(W1_init.copy(), requires_grad=True)
b1_pt = torch.tensor(b1_init.copy(), requires_grad=True)
W2_pt = torch.tensor(W2_init.copy(), requires_grad=True)
b2_pt = torch.tensor(b2_init.copy(), requires_grad=True)
x_pt = torch.tensor(x_data)
target_pt = torch.tensor(target)

# 前向傳播（和 NumPy 一樣的運算）
z1_pt = x_pt @ W1_pt + b1_pt
a1_pt = torch.relu(z1_pt)              # 用 PyTorch 內建 ReLU
z2_pt = a1_pt @ W2_pt + b2_pt

# PyTorch 的 CrossEntropyLoss 內建了 Softmax
loss_pt = torch.nn.functional.cross_entropy(z2_pt.unsqueeze(0), target_pt.unsqueeze(0))

print(f"  Loss: {loss_pt.item():.6f}")

# 反向傳播 — 一行搞定！
loss_pt.backward()

print(f"  dW1 range: [{W1_pt.grad.min().item():.6f}, {W1_pt.grad.max().item():.6f}]")
print(f"  dW2 range: [{W2_pt.grad.min().item():.6f}, {W2_pt.grad.max().item():.6f}]")

# ─────────────────────────────────────────
# 比較兩者
# ─────────────────────────────────────────
print("\n--- 比較結果 ---")
print(f"  Loss 差異:  {abs(loss_np - loss_pt.item()):.8f}")
print(f"  dW1 差異:   {np.max(np.abs(dW1_np - W1_pt.grad.numpy())):.8f}")
print(f"  db1 差異:   {np.max(np.abs(db1_np - b1_pt.grad.numpy())):.8f}")
print(f"  dW2 差異:   {np.max(np.abs(dW2_np - W2_pt.grad.numpy())):.8f}")
print(f"  db2 差異:   {np.max(np.abs(db2_np - b2_pt.grad.numpy())):.8f}")
print("  → 完全一致！Autograd 做的就是你手寫的 Chain Rule")

print(f"""
對比：
  NumPy 手刻:  前向 ~10 行 + 反向 ~15 行（每層都要自己推導數）
  PyTorch:     前向 ~5 行 + loss.backward() 一行

  模型越大，手寫反向傳播越不現實。
  Autograd 讓你只需要定義前向傳播，梯度自動算好。
""")


# ============================================================================
# 3. 梯度檢查 (Gradient Checking)
# ============================================================================
print("=" * 60)
print("3. 梯度檢查 — 驗證你的梯度是否正確")
print("=" * 60)

# 如果你自己寫了自定義的層或損失函數，可以用數值梯度來驗證

def numerical_gradient(f, x, eps=1e-5):
    """用數值方法計算梯度（和 Phase 0 學的一樣）"""
    grad = torch.zeros_like(x)
    for i in range(x.numel()):
        x_flat = x.view(-1)
        old_val = x_flat[i].item()

        x_flat[i] = old_val + eps
        loss_plus = f(x)

        x_flat[i] = old_val - eps
        loss_minus = f(x)

        grad.view(-1)[i] = (loss_plus - loss_minus) / (2 * eps)
        x_flat[i] = old_val

    return grad


# 測試
w_check = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

def my_loss(w):
    return (w ** 3).sum()    # L = w₁³ + w₂³ + w₃³

# Autograd 梯度
loss = my_loss(w_check)
loss.backward()
autograd_grad = w_check.grad.clone()

# 數值梯度
w_check2 = torch.tensor([1.0, 2.0, 3.0])
numerical_grad = numerical_gradient(my_loss, w_check2)

print(f"Autograd 梯度: {autograd_grad}")        # [3, 12, 27] = 3w²
print(f"數值梯度:     {numerical_grad}")
print(f"差異: {(autograd_grad - numerical_grad).abs().max().item():.8f}")
print("→ 差異極小，Autograd 正確")


# ============================================================================
# 4. requires_grad / detach / no_grad
# ============================================================================
print("\n" + "=" * 60)
print("4. 控制梯度追蹤")
print("=" * 60)

# --- requires_grad ---
# 只有 requires_grad=True 的 Tensor 才會被追蹤
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0)    # 預設 False
c = a * b
print(f"a.requires_grad = {a.requires_grad}")  # True
print(f"b.requires_grad = {b.requires_grad}")  # False
print(f"c.requires_grad = {c.requires_grad}")  # True (因為 a 需要梯度)

# --- detach() ---
# 把 Tensor 從計算圖中分離
d = c.detach()
print(f"\nc.requires_grad = {c.requires_grad}")  # True
print(f"d = c.detach(), d.requires_grad = {d.requires_grad}")  # False
# 用途：凍結模型的部分參數、從計算圖取出數值

# --- torch.no_grad() ---
# 在這個區塊內，所有運算都不追蹤梯度（省記憶體、加速）
print("\n--- torch.no_grad() ---")
x = torch.tensor(1.0, requires_grad=True)

# 正常情況
y = x * 2
print(f"正常: y.requires_grad = {y.requires_grad}")

# no_grad 區塊
with torch.no_grad():
    y_no = x * 2
    print(f"no_grad: y.requires_grad = {y_no.requires_grad}")

print("""
三者的使用場景：

  requires_grad=True   → 模型的權重（需要算梯度來更新）
  requires_grad=False  → 輸入資料（不需要對資料算梯度）

  .detach()           → 從計算圖中分離，用於：
                         - 凍結 pre-trained 層
                         - 把中間結果拿出來做別的用途

  torch.no_grad()     → 推論(inference)時用，不記錄計算圖：
                         - 驗證/測試時
                         - 節省記憶體和加速
""")


# ============================================================================
# 小結
# ============================================================================
print("=" * 60)
print("小結")
print("=" * 60)
print("""
Autograd 的核心理解：

  前向傳播：每次運算都在建立「計算圖」
  backward()：沿著計算圖反向走，用 Chain Rule 算每個節點的梯度

  你在 Phase 0 手推的 Chain Rule，就是 Autograd 在做的事。
  差別是：你不用再手推了。

  但理解原理很重要，因為：
  1. Debug 時你需要知道梯度為什麼是 NaN 或 0
  2. 自定義 Loss 或層時需要確認梯度正確
  3. 理解梯度累積、梯度爆炸等問題

下一步：03_neural_network_from_scratch.py — 用純 NumPy 手刻完整的神經網路
""")
