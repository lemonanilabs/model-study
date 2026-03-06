"""
=============================================================================
Phase 0-1: NumPy 基礎 — 一切的起點
=============================================================================

為什麼先學 NumPy？
─────────────────
神經網路的所有運算，底層都是「矩陣（張量）運算」。
PyTorch 的 Tensor 和 NumPy 的 ndarray 幾乎一模一樣。
先搞懂 NumPy，後面學 PyTorch 就是換個名字而已。

本檔涵蓋：
  1. ndarray 的建立與基本屬性
  2. 索引與切片
  3. 矩陣運算（加減乘、點積）
  4. Broadcasting（最重要的概念之一）
  5. 向量化操作 vs 迴圈
  6. 常用函數速查
"""

import numpy as np

# ============================================================================
# 1. ndarray 的建立與基本屬性
# ============================================================================
print("=" * 60)
print("1. ndarray 基本概念")
print("=" * 60)

# --- 建立陣列 ---
a = np.array([1, 2, 3])              # 從 list 建立一維陣列（向量）
b = np.array([[1, 2, 3],
              [4, 5, 6]])            # 二維陣列（矩陣）

print(f"一維陣列 a = {a}")
print(f"二維陣列 b =\n{b}")

# --- 重要屬性 ---
# 在深度學習中，你會無時無刻查看這三個屬性
print(f"\nb 的形狀 (shape): {b.shape}")    # (2, 3) → 2 列 3 行
print(f"b 的維度 (ndim):  {b.ndim}")       # 2 → 二維
print(f"b 的資料型態 (dtype): {b.dtype}")   # int64

# shape 的意義：
#   (2, 3)     → 2 列 3 行的矩陣
#   (32, 3, 224, 224) → 一個批次 32 張、3 通道、224x224 的圖片
#   shape 是深度學習中最常 debug 的東西，形狀不對一切都會爆炸

# --- 常用建立方式 ---
zeros = np.zeros((3, 4))           # 全 0，常用來初始化
ones = np.ones((2, 3))             # 全 1
rand = np.random.randn(3, 4)      # 標準常態分佈隨機數（平均 0，標準差 1）
                                   # 神經網路的初始權重通常這樣生成
arange = np.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)   # [0, 0.25, 0.5, 0.75, 1.0] 均等分割

print(f"\n全零矩陣 (3x4):\n{zeros}")
print(f"\n隨機矩陣 (3x4):\n{rand}")


# ============================================================================
# 2. 索引與切片
# ============================================================================
print("\n" + "=" * 60)
print("2. 索引與切片")
print("=" * 60)

m = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])

print(f"矩陣 m =\n{m}")
print(f"\nm[0, 1]     = {m[0, 1]}")       # 第 0 列、第 1 行 → 20
print(f"m[1]        = {m[1]}")             # 第 1 列整列 → [40, 50, 60]
print(f"m[:, 0]     = {m[:, 0]}")          # 所有列的第 0 行 → [10, 40, 70]
print(f"m[0:2, 1:]  =\n{m[0:2, 1:]}")     # 前 2 列、第 1 行之後

# 重要觀念：
#   m[列, 行] — 先列再行（row, column）
#   : 表示「全部」
#   0:2 表示 index 0 和 1（不含 2）

# --- Boolean Indexing（條件篩選）---
print(f"\nm > 50 的元素: {m[m > 50]}")     # [60, 70, 80, 90]
# 這在資料處理中非常常用：篩選出符合條件的樣本


# ============================================================================
# 3. 矩陣運算
# ============================================================================
print("\n" + "=" * 60)
print("3. 矩陣運算")
print("=" * 60)

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# --- 逐元素運算 (Element-wise) ---
print(f"A + B (逐元素加) =\n{A + B}")
print(f"\nA * B (逐元素乘) =\n{A * B}")     # 注意：這不是矩陣乘法！

# --- 矩陣乘法 (Dot Product / Matrix Multiplication) ---
# 這是神經網路最核心的運算
# 規則：(m, n) @ (n, p) → (m, p)
# 左矩陣的「行數」必須等於右矩陣的「列數」
C = A @ B           # 等同 np.dot(A, B) 或 np.matmul(A, B)
print(f"\nA @ B (矩陣乘法) =\n{C}")

# 手動驗算 A @ B：
# [1*5+2*7, 1*6+2*8] = [19, 22]
# [3*5+4*7, 3*6+4*8] = [43, 50]
print(f"手算驗證 [0,0]: 1*5 + 2*7 = {1*5 + 2*7}")

# --- 為什麼矩陣乘法重要？---
# 神經網路的一層就是： output = input @ weights + bias
# 例如：10 個特徵的輸入，通過一層 10→5 的網路
input_data = np.random.randn(1, 10)     # 1 個樣本，10 個特徵
weights = np.random.randn(10, 5)        # 權重：10 輸入 → 5 輸出
bias = np.random.randn(1, 5)            # 偏差

output = input_data @ weights + bias     # 這就是一層神經網路的前向傳播！
print(f"\n--- 模擬神經網路的一層 ---")
print(f"輸入 shape: {input_data.shape}")   # (1, 10)
print(f"權重 shape: {weights.shape}")       # (10, 5)
print(f"輸出 shape: {output.shape}")        # (1, 5)

# --- 轉置 (Transpose) ---
print(f"\nA 的轉置 =\n{A.T}")              # 列與行互換

# --- Reshape（改變形狀）---
x = np.arange(12)
print(f"\n原始 x: {x}, shape: {x.shape}")
x_reshaped = x.reshape(3, 4)
print(f"reshape(3,4):\n{x_reshaped}")
x_auto = x.reshape(3, -1)               # -1 表示自動計算
print(f"reshape(3,-1) 和上面一樣:\n{x_auto}")


# ============================================================================
# 4. Broadcasting（廣播機制）
# ============================================================================
print("\n" + "=" * 60)
print("4. Broadcasting — NumPy 最神奇也最容易搞混的功能")
print("=" * 60)

# Broadcasting 讓不同形狀的陣列可以做運算
# 規則：從最右邊的維度開始比較，兩者相等或其中一個是 1，就可以廣播

# 例 1：矩陣 + 向量
M = np.array([[1, 2, 3],
              [4, 5, 6]])            # shape: (2, 3)
v = np.array([10, 20, 30])           # shape: (3,)

result = M + v
print(f"M (2,3) + v (3,) =\n{result}")
# v 被自動「複製」成 [[10,20,30], [10,20,30]] 再相加
# 實際上不會真的複製，效能很好

# 例 2：每一列減去該列的平均值（資料標準化的基礎）
data = np.array([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]])
row_mean = data.mean(axis=1, keepdims=True)   # shape: (2, 1)
centered = data - row_mean                     # (2, 3) - (2, 1) → broadcasting
print(f"\n原始資料:\n{data}")
print(f"每列平均: {row_mean.flatten()}")
print(f"去中心化:\n{centered}")

# keepdims=True 的重要性：
#   沒有 keepdims：mean shape = (2,)，無法和 (2,3) 廣播
#   有 keepdims：  mean shape = (2,1)，可以和 (2,3) 廣播

# 例 3：在深度學習中，bias 的加法就是 broadcasting
#   output shape: (batch_size, features) = (32, 128)
#   bias shape:   (128,) 或 (1, 128)
#   → bias 會自動廣播到每一個樣本


# ============================================================================
# 5. 向量化操作 vs 迴圈
# ============================================================================
print("\n" + "=" * 60)
print("5. 向量化 vs 迴圈 — 為什麼不要寫 for loop")
print("=" * 60)

import time

size = 1_000_000
a = np.random.randn(size)
b = np.random.randn(size)

# 方法 1：Python for 迴圈
start = time.time()
result_loop = np.zeros(size)
for i in range(size):
    result_loop[i] = a[i] * b[i]
loop_time = time.time() - start

# 方法 2：NumPy 向量化操作
start = time.time()
result_vec = a * b
vec_time = time.time() - start

print(f"For 迴圈耗時: {loop_time:.4f} 秒")
print(f"向量化耗時:   {vec_time:.6f} 秒")
print(f"向量化快了約 {loop_time / vec_time:.0f} 倍")
print(f"結果相同: {np.allclose(result_loop, result_vec)}")

# allclose(a,b,) = ∣a−b∣≤atol+rtol⋅∣b∣
# 假設（NumPy 預設）
# rtol = 1e-5
# atol = 1e-8
# 怎麼調 rtol/atol？
# 如果你比較在意「小數附近要很準」：調小 atol
# 如果你比較在意「比例誤差」：調小 rtol
# 做 ML/線代通常 allclose 預設夠用；但做單元測試（尤其小量級）常要調 atol

# 為什麼向量化這麼快？
#   1. NumPy 底層用 C 語言實作，繞過 Python 的慢速迴圈
#   2. 利用 CPU 的 SIMD 指令做平行計算
#   3. 深度學習框架 (PyTorch) 也是同樣原理，再加上 GPU 加速


# ============================================================================
# 6. 常用函數速查
# ============================================================================
print("\n" + "=" * 60)
print("6. 常用函數")
print("=" * 60)

x = np.array([[1, 2, 3],
              [4, 5, 6]])

# --- 聚合函數 ---
print(f"x.sum()          = {x.sum()}")           # 全部加總 = 21
print(f"x.sum(axis=0)    = {x.sum(axis=0)}")     # 沿列方向加 = [5, 7, 9]
print(f"x.sum(axis=1)    = {x.sum(axis=1)}")     # 沿行方向加 = [6, 15]
print(f"x.mean()         = {x.mean()}")           # 平均
print(f"x.std()          = {x.std():.4f}")        # 標準差
print(f"x.max()          = {x.max()}")
print(f"x.argmax()       = {x.argmax()}")         # 最大值的 index（分類模型取預測結果用）
print(f"x.argmax(axis=1) = {x.argmax(axis=1)}")   # 每列最大值的 index

# axis 的理解方式：
#   axis=0 → 沿著「列」的方向操作（結果少了列的維度）
#   axis=1 → 沿著「行」的方向操作（結果少了行的維度）
#   想像把該軸「壓扁」

# --- 數學函數 ---
v = np.array([-2, -1, 0, 1, 2])
print(f"\nnp.abs(v)    = {np.abs(v)}")
print(f"np.exp(v)    = {np.exp(v).round(4)}")      # e^x，Softmax 的基礎
print(f"np.log(v[3:])= {np.log(v[3:]).round(4)}")  # ln(x)，Cross-Entropy 的基礎

# --- 特別重要：np.exp 和 np.log ---
# Softmax 函數（把任意數字轉成機率分佈）：
logits = np.array([2.0, 1.0, 0.1])
exp_logits = np.exp(logits)
softmax = exp_logits / exp_logits.sum()
print(f"\nSoftmax 示範:")
print(f"  原始分數 (logits): {logits}")
print(f"  exp(logits):       {exp_logits.round(4)}")
print(f"  softmax 結果:      {softmax.round(4)}")
print(f"  總和 = {softmax.sum():.4f}")   # 一定是 1.0


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結：NumPy 在深度學習中的角色")
print("=" * 60)
print("""
你剛學到的東西，直接對應到深度學習：

  NumPy 概念          →  深度學習對應
  ─────────────────────────────────────
  ndarray             →  Tensor（PyTorch）
  shape               →  模型每一層的輸入/輸出維度
  矩陣乘法 (@)        →  神經網路的前向傳播
  Broadcasting        →  Bias 加法、標準化
  向量化操作           →  GPU 平行計算的基礎
  np.exp / np.log     →  Softmax / Cross-Entropy
  argmax              →  取得模型的預測類別

下一步：02_linear_algebra.py — 深入理解矩陣運算的幾何意義
""")
