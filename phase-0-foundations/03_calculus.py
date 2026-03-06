"""
=============================================================================
Phase 0-3: 微積分 — 模型是怎麼「學習」的
=============================================================================

核心問題：
─────────
神經網路有幾百萬個權重參數，要怎麼找到「最好的」那組權重？
答案：梯度下降 (Gradient Descent)

而梯度下降的數學基礎，就是微積分中的「導數」和「鏈鎖律」。

本檔涵蓋：
  1. 導數的直覺：函數在某一點的「變化率」
  2. 用數值方法近似導數
  3. 偏微分：多變數函數的導數
  4. 梯度：所有偏微分組成的向量
  5. 梯度下降法：用梯度來最小化函數
  6. Chain Rule（鏈鎖律）：反向傳播的數學核心
"""

import numpy as np

# ============================================================================
# 1. 導數 (Derivative) 的直覺
# ============================================================================
print("=" * 60)
print("1. 導數 = 函數的變化率")
print("=" * 60)

# 導數的定義：f'(x) = lim(h→0) [f(x+h) - f(x)] / h
# 直覺：如果我稍微改變 x，f(x) 會變多少？

# 例：f(x) = x²
def f(x):
    return x ** 2

# 數學上，f(x) = x² 的導數是 f'(x) = 2x
# 用數值方法來驗證：

x = 3.0
h = 0.00001  # 很小的變化量

numerical_derivative = (f(x + h) - f(x)) / h
analytical_derivative = 2 * x   # 數學解

print(f"f(x) = x², 在 x = {x} 處：")
print(f"  數值導數:  {numerical_derivative:.6f}")
print(f"  解析導數:  {analytical_derivative:.6f}")
print(f"  幾乎相等！")

# 導數的意義：
#   f'(3) = 6 → 在 x=3 時，x 每增加 1，f(x) 大約增加 6
#   f'(3) > 0 → 函數在這裡是「上升」的
#   f'(3) < 0 → 函數在這裡是「下降」的
#   f'(x) = 0 → 函數在這裡是「平的」→ 可能是最大值或最小值！

# 為什麼重要？
#   Loss 函數的導數 = 告訴我們「往哪個方向調整權重，Loss 會變小」


# ============================================================================
# 2. 用程式驗證各種導數
# ============================================================================
print("\n" + "=" * 60)
print("2. 用程式驗證導數公式")
print("=" * 60)

def numerical_grad(func, x, h=1e-5):
    """計算函數在 x 處的數值導數"""
    return (func(x + h) - func(x - h)) / (2 * h)  # 中央差分，更精確

# 常見函數的導數
functions = [
    ("f(x) = x²",       lambda x: x**2,        lambda x: 2*x),
    ("f(x) = x³",       lambda x: x**3,        lambda x: 3*x**2),
    ("f(x) = sin(x)",   lambda x: np.sin(x),   lambda x: np.cos(x)),
    ("f(x) = eˣ",       lambda x: np.exp(x),   lambda x: np.exp(x)),   # e^x 的導數是自己！
    ("f(x) = ln(x)",    lambda x: np.log(x),   lambda x: 1/x),
]

x = 2.0
for name, func, derivative in functions:
    numerical = numerical_grad(func, x)
    analytical = derivative(x)
    print(f"{name:18s}  數值: {numerical:8.4f}  解析: {analytical:8.4f}")


# ============================================================================
# 3. 偏微分 (Partial Derivative)
# ============================================================================
print("\n" + "=" * 60)
print("3. 偏微分 — 多變數函數的導數")
print("=" * 60)

# 當函數有多個變數時，對每個變數分別求導就是偏微分
# f(x, y) = x² + 2xy + y²
# ∂f/∂x = 2x + 2y   （把 y 當常數，對 x 求導）
# ∂f/∂y = 2x + 2y   （把 x 當常數，對 y 求導）

def f_xy(x, y):
    return x**2 + 2*x*y + y**2

x, y = 3.0, 2.0
h = 1e-5

# 對 x 的偏微分（固定 y，只動 x）
df_dx = (f_xy(x + h, y) - f_xy(x - h, y)) / (2 * h)
# 對 y 的偏微分（固定 x，只動 y）
df_dy = (f_xy(x, y + h) - f_xy(x, y - h)) / (2 * h)

print(f"f(x,y) = x² + 2xy + y², 在 (x={x}, y={y}) 處：")
print(f"  ∂f/∂x = 2x + 2y = {2*x + 2*y:.1f} (數值: {df_dx:.4f})")
print(f"  ∂f/∂y = 2x + 2y = {2*x + 2*y:.1f} (數值: {df_dy:.4f})")

# 在 ML 中：
#   Loss 函數有幾百萬個變數（每個權重就是一個變數）
#   我們需要算 Loss 對每一個權重的偏微分


# ============================================================================
# 4. 梯度 (Gradient) = 所有偏微分組成的向量
# ============================================================================
print("\n" + "=" * 60)
print("4. 梯度 = 偏微分組成的向量")
print("=" * 60)

# 梯度 ∇f = [∂f/∂x, ∂f/∂y]
# 梯度指向函數「上升最快」的方向
# 負梯度指向函數「下降最快」的方向 → 這就是梯度下降的方向！

gradient = np.array([df_dx, df_dy])
print(f"梯度 ∇f = [{df_dx:.4f}, {df_dy:.4f}]")
print(f"梯度的方向 = 函數上升最快的方向")
print(f"負梯度的方向 = 函數下降最快的方向（我們要往這裡走！）")


# ============================================================================
# 5. 梯度下降法 (Gradient Descent) — 完整實作
# ============================================================================
print("\n" + "=" * 60)
print("5. 梯度下降法 — 模型訓練的核心")
print("=" * 60)

# 目標：找到 f(x) = (x-3)² + 1 的最小值
# 我們知道答案是 x=3, f(3)=1，但讓電腦自己找到

def loss(x):
    return (x - 3) ** 2 + 1

def loss_gradient(x):
    return 2 * (x - 3)

# 梯度下降
x = 10.0              # 隨機起點
learning_rate = 0.1    # 學習率：每次走多大步

print(f"目標：找到 f(x) = (x-3)² + 1 的最小值")
print(f"起始位置: x = {x}, f(x) = {loss(x)}")
print(f"學習率: {learning_rate}")
print(f"\n{'步驟':>4s}  {'x':>8s}  {'f(x)':>8s}  {'梯度':>8s}")
print("-" * 36)

for step in range(15):
    grad = loss_gradient(x)
    print(f"{step:4d}  {x:8.4f}  {loss(x):8.4f}  {grad:8.4f}")
    x = x - learning_rate * grad       # 核心公式：往負梯度方向走一步
    #     ↑ 新位置 = 舊位置 - 學習率 × 梯度

print(f"\n最終: x = {x:.6f}, f(x) = {loss(x):.6f}")
print(f"真實最小值: x = 3, f(3) = 1")

# 學習率的影響：
#   太大 → 跳過最小值，不收斂
#   太小 → 收斂太慢
#   剛好 → 穩定收斂到最小值


# ============================================================================
# 5.5 二維梯度下降（更接近真實情況）
# ============================================================================
print("\n" + "=" * 60)
print("5.5 二維梯度下降 — 同時優化多個參數")
print("=" * 60)

# f(w, b) = (w*2 + b - 5)² → 模擬 loss = (prediction - target)²
# 其中 input=2, target=5, prediction = w*input + b
# 我們要找最好的 w 和 b

def loss_2d(w, b):
    prediction = w * 2 + b
    target = 5.0
    return (prediction - target) ** 2

def loss_2d_grad(w, b):
    prediction = w * 2 + b
    target = 5.0
    error = 2 * (prediction - target)
    dw = error * 2       # ∂loss/∂w = ∂loss/∂pred * ∂pred/∂w = error * input
    db = error * 1       # ∂loss/∂b = ∂loss/∂pred * ∂pred/∂b = error * 1
    return dw, db

w, b = 0.0, 0.0       # 隨機初始化
lr = 0.01

print(f"任務：找 w, b 使得 w*2 + b ≈ 5")
print(f"起始: w={w}, b={b}, prediction={w*2+b}, loss={loss_2d(w,b)}")
print()

for step in range(50):
    dw, db = loss_2d_grad(w, b)
    w = w - lr * dw
    b = b - lr * db
    if step % 10 == 0 or step == 49:
        pred = w * 2 + b
        print(f"Step {step:3d}: w={w:.4f}, b={b:.4f}, "
              f"pred={pred:.4f}, loss={loss_2d(w,b):.6f}")

print(f"\n最終: {w:.4f} * 2 + {b:.4f} = {w*2+b:.4f} (目標: 5)")


# ============================================================================
# 6. Chain Rule（鏈鎖律）— 反向傳播的數學核心
# ============================================================================
print("\n" + "=" * 60)
print("6. Chain Rule — 反向傳播的數學核心")
print("=" * 60)

# Chain Rule：如果 y = f(g(x))，那麼 dy/dx = f'(g(x)) * g'(x)
# 直覺：「連鎖效應」— x 改變 → g(x) 改變 → y 改變

# 例：y = (2x + 1)³
# 分解：u = 2x + 1,  y = u³
# dy/dx = dy/du * du/dx = 3u² * 2 = 6(2x+1)²

x = 2.0
# 數值驗證
h = 1e-5
numerical = ((2*(x+h)+1)**3 - (2*(x-h)+1)**3) / (2*h)
analytical = 6 * (2*x+1)**2

print(f"y = (2x+1)³, 在 x={x} 處：")
print(f"  Chain Rule: dy/du * du/dx = 3u² * 2 = 6(2x+1)²")
print(f"  數值導數: {numerical:.4f}")
print(f"  解析導數: {analytical:.4f}")

# ============================================================================
# 6.5 用 Chain Rule 理解反向傳播
# ============================================================================
print("\n" + "-" * 60)
print("6.5 反向傳播 = Chain Rule 的直接應用")
print("-" * 60)

# 模擬一個簡單的計算圖：
#   x → [×w] → z → [ReLU] → a → [×v] → y → [Loss]
#
#   z = w * x
#   a = max(0, z)    (ReLU)
#   y = v * a
#   L = (y - target)²

# 前向傳播
x_val = 2.0
w = 0.5
v = 1.5
target = 3.0

z = w * x_val               # z = 0.5 * 2 = 1.0
a = max(0, z)                # a = max(0, 1) = 1.0  (ReLU)
y = v * a                   # y = 1.5 * 1 = 1.5
L = (y - target) ** 2       # L = (1.5 - 3)² = 2.25

print(f"前向傳播:")
print(f"  x={x_val}, w={w}, v={v}, target={target}")
print(f"  z = w*x = {z}")
print(f"  a = ReLU(z) = {a}")
print(f"  y = v*a = {y}")
print(f"  Loss = (y-target)² = {L}")

# 反向傳播（Chain Rule 從後往前算）
print(f"\n反向傳播（用 Chain Rule 從後往前）:")

dL_dy = 2 * (y - target)             # ∂L/∂y = 2(y-target) = -3.0
print(f"  ∂L/∂y = 2(y-target) = {dL_dy}")

dL_dv = dL_dy * a                    # ∂L/∂v = ∂L/∂y × ∂y/∂v = dL_dy × a
print(f"  ∂L/∂v = ∂L/∂y × ∂y/∂v = {dL_dy} × {a} = {dL_dv}")

dL_da = dL_dy * v                    # ∂L/∂a = ∂L/∂y × ∂y/∂a = dL_dy × v
print(f"  ∂L/∂a = ∂L/∂y × ∂y/∂a = {dL_dy} × {v} = {dL_da}")

dL_dz = dL_da * (1 if z > 0 else 0) # ∂L/∂z = ∂L/∂a × ∂a/∂z (ReLU 導數)
print(f"  ∂L/∂z = ∂L/∂a × ReLU'(z) = {dL_da} × {1 if z > 0 else 0} = {dL_dz}")

dL_dw = dL_dz * x_val                # ∂L/∂w = ∂L/∂z × ∂z/∂w = dL_dz × x
print(f"  ∂L/∂w = ∂L/∂z × ∂z/∂w = {dL_dz} × {x_val} = {dL_dw}")

# 更新權重
lr = 0.01
w_new = w - lr * dL_dw
v_new = v - lr * dL_dv
print(f"\n更新權重 (lr={lr}):")
print(f"  w: {w} → {w_new:.4f}")
print(f"  v: {v} → {v_new:.4f}")

# 驗證新權重確實讓 Loss 變小
y_new = v_new * max(0, w_new * x_val)
L_new = (y_new - target) ** 2
print(f"\n驗證: 舊 Loss = {L:.4f}, 新 Loss = {L_new:.4f}")
print(f"Loss {'下降了' if L_new < L else '沒有下降'}！")


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結")
print("=" * 60)
print("""
微積分在 ML 中的角色：

  概念              →  ML 對應
  ─────────────────────────────────────────
  導數              →  Loss 對單一權重的變化率
  偏微分            →  Loss 對其中一個權重的變化率
  梯度              →  Loss 對所有權重的偏微分（向量）
  梯度下降          →  訓練模型的核心演算法
  Chain Rule        →  反向傳播（Backpropagation）

訓練模型的完整流程：
  1. 前向傳播：用當前權重算出預測值
  2. 計算 Loss：預測值和真實值的差距
  3. 反向傳播：用 Chain Rule 算出每個權重的梯度
  4. 更新權重：w = w - lr * gradient
  5. 重複 1-4，直到 Loss 夠小

這就是所有深度學習框架（PyTorch, TensorFlow）在做的事。
框架幫你自動做了第 3 步（Autograd），但理解原理很重要。

下一步：04_probability.py — 理解損失函數背後的機率意義
""")
