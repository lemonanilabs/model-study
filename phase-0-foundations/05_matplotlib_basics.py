"""
=============================================================================
Phase 0-5: Matplotlib 資料視覺化
=============================================================================

為什麼學視覺化？
───────────────
  - 看 Loss 曲線判斷訓練狀況（過擬合？學習率太大？）
  - 看資料分佈決定前處理方式
  - 看模型預測結果做 Error Analysis

本檔涵蓋：
  1. 基本折線圖（Loss 曲線）
  2. 散點圖（資料分佈）
  3. 直方圖（特徵分佈）
  4. 子圖排版
  5. 實戰：模擬完整的訓練監控
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. 基本折線圖 — 畫 Loss 曲線
# ============================================================================
print("1. 繪製 Loss 曲線...")

# 模擬訓練過程中的 Loss 變化
np.random.seed(42)
epochs = np.arange(1, 51)
train_loss = 2.0 * np.exp(-0.08 * epochs) + 0.1 + np.random.normal(0, 0.02, 50)
val_loss = 2.0 * np.exp(-0.06 * epochs) + 0.2 + np.random.normal(0, 0.03, 50)
# 模擬過擬合：驗證 loss 在後期開始上升
val_loss[35:] += np.linspace(0, 0.3, 15)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', color='red', linewidth=2, linestyle='--')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training vs Validation Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# 標記過擬合開始的位置
plt.axvline(x=35, color='green', linestyle=':', alpha=0.7, label='Overfitting starts')
plt.annotate('Overfitting\nstarts here', xy=(35, val_loss[34]), xytext=(40, 0.8),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=11, color='green')

plt.tight_layout()
plt.savefig('phase-0-foundations/plots/01_loss_curve.png', dpi=100)
plt.close()
print("  → 儲存至 plots/01_loss_curve.png")
print("  觀察：Train Loss 持續下降，但 Val Loss 在 epoch 35 後上升 = 過擬合")


# ============================================================================
# 2. 散點圖 — 資料分佈
# ============================================================================
print("\n2. 繪製分類資料的散點圖...")

# 生成兩類資料
np.random.seed(42)
class_0 = np.random.randn(100, 2) + np.array([1, 1])
class_1 = np.random.randn(100, 2) + np.array([4, 4])

plt.figure(figsize=(8, 8))
plt.scatter(class_0[:, 0], class_0[:, 1], c='blue', alpha=0.6, label='Class 0', s=50)
plt.scatter(class_1[:, 0], class_1[:, 1], c='red', alpha=0.6, label='Class 1', s=50)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Two-Class Classification Data', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('phase-0-foundations/plots/02_scatter.png', dpi=100)
plt.close()
print("  → 儲存至 plots/02_scatter.png")
print("  觀察：兩類資料有部分重疊，分類器不可能 100% 正確")


# ============================================================================
# 3. 直方圖 — 特徵分佈
# ============================================================================
print("\n3. 繪製特徵分佈直方圖...")

np.random.seed(42)
data = np.random.normal(loc=5, scale=2, size=1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=40, color='steelblue', alpha=0.7, edgecolor='white', density=True)

# 疊加理論的常態分佈曲線
x = np.linspace(data.min(), data.max(), 100)
pdf = (1 / (2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - 5) / 2) ** 2)
plt.plot(x, pdf, 'r-', linewidth=2, label=f'N(μ=5, σ=2)')

plt.xlabel('Value', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Feature Distribution (Histogram + PDF)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phase-0-foundations/plots/03_histogram.png', dpi=100)
plt.close()
print("  → 儲存至 plots/03_histogram.png")


# ============================================================================
# 4. 子圖排版 — 一次看多個圖
# ============================================================================
print("\n4. 繪製子圖排版...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 左上：激活函數比較
x = np.linspace(-5, 5, 200)
axes[0, 0].plot(x, np.maximum(0, x), label='ReLU', linewidth=2)
axes[0, 0].plot(x, 1 / (1 + np.exp(-x)), label='Sigmoid', linewidth=2)
axes[0, 0].plot(x, np.tanh(x), label='Tanh', linewidth=2)
axes[0, 0].set_title('Activation Functions')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(-5, 5)

# 右上：不同學習率的 Loss 曲線
epochs = np.arange(50)
for lr, color in [(0.001, 'blue'), (0.01, 'green'), (0.1, 'red')]:
    loss = 2.0 * np.exp(-lr * 20 * epochs) + 0.1
    if lr == 0.1:
        loss += np.random.normal(0, 0.3, 50)   # 學習率太大 → 震盪
    axes[0, 1].plot(epochs, loss, label=f'lr={lr}', color=color, linewidth=2)
axes[0, 1].set_title('Effect of Learning Rate')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 左下：梯度下降的路徑（等高線圖）
w = np.linspace(-3, 5, 100)
b = np.linspace(-3, 5, 100)
W, B = np.meshgrid(w, b)
Loss = (W - 2) ** 2 + (B - 1) ** 2     # 碗狀 Loss 曲面

axes[1, 0].contour(W, B, Loss, levels=20, cmap='viridis')
# 模擬梯度下降路徑
path_w, path_b = [4.0], [4.0]
lr = 0.1
for _ in range(20):
    gw = 2 * (path_w[-1] - 2)
    gb = 2 * (path_b[-1] - 1)
    path_w.append(path_w[-1] - lr * gw)
    path_b.append(path_b[-1] - lr * gb)
axes[1, 0].plot(path_w, path_b, 'ro-', markersize=4, linewidth=1.5)
axes[1, 0].plot(2, 1, 'g*', markersize=15, label='Minimum')
axes[1, 0].set_title('Gradient Descent Path')
axes[1, 0].set_xlabel('w')
axes[1, 0].set_ylabel('b')
axes[1, 0].legend()

# 右下：Softmax 輸出隨 temperature 變化
logits = np.array([2.0, 1.0, 0.5])
temps = [0.5, 1.0, 2.0, 5.0]
x_pos = np.arange(len(logits))
width = 0.2

for i, temp in enumerate(temps):
    softmax = np.exp(logits / temp) / np.exp(logits / temp).sum()
    axes[1, 1].bar(x_pos + i * width, softmax, width, label=f'T={temp}', alpha=0.8)

axes[1, 1].set_title('Softmax with Temperature')
axes[1, 1].set_xlabel('Class')
axes[1, 1].set_ylabel('Probability')
axes[1, 1].set_xticks(x_pos + 1.5 * width)
axes[1, 1].set_xticklabels(['Class 0', 'Class 1', 'Class 2'])
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('ML Visualization Essentials', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('phase-0-foundations/plots/04_subplots.png', dpi=100)
plt.close()
print("  → 儲存至 plots/04_subplots.png")


# ============================================================================
# 5. 實戰：完整的訓練監控面板
# ============================================================================
print("\n5. 繪製訓練監控面板...")

np.random.seed(42)
epochs = 100

# 模擬訓練數據
t = np.arange(1, epochs + 1)
t_loss = 2.5 * np.exp(-0.05 * t) + 0.15 + np.random.normal(0, 0.02, epochs)
v_loss = 2.5 * np.exp(-0.04 * t) + 0.25 + np.random.normal(0, 0.03, epochs)
t_acc = 1 - 0.6 * np.exp(-0.06 * t) + np.random.normal(0, 0.01, epochs)
v_acc = 1 - 0.65 * np.exp(-0.05 * t) + np.random.normal(0, 0.015, epochs)
lr_schedule = 0.01 * 0.95 ** (t // 10)     # 每 10 epoch 衰減 5%

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(t, t_loss, 'b-', alpha=0.7, label='Train')
axes[0].plot(t, v_loss, 'r-', alpha=0.7, label='Val')
axes[0].set_title('Loss', fontsize=13)
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(t, np.clip(t_acc, 0, 1), 'b-', alpha=0.7, label='Train')
axes[1].plot(t, np.clip(v_acc, 0, 1), 'r-', alpha=0.7, label='Val')
axes[1].set_title('Accuracy', fontsize=13)
axes[1].set_xlabel('Epoch')
axes[1].set_ylim(0, 1.05)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Learning Rate
axes[2].plot(t, lr_schedule, 'g-', linewidth=2)
axes[2].set_title('Learning Rate Schedule', fontsize=13)
axes[2].set_xlabel('Epoch')
axes[2].set_yscale('log')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Training Dashboard', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('phase-0-foundations/plots/05_training_dashboard.png', dpi=100)
plt.close()
print("  → 儲存至 plots/05_training_dashboard.png")


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結")
print("=" * 60)
print("""
視覺化在 ML 中的用途：

  圖表類型         →  用途
  ──────────────────────────────────────
  折線圖 (plot)    →  Loss/Accuracy 曲線、學習率變化
  散點圖 (scatter) →  資料分佈、Embedding 視覺化
  直方圖 (hist)    →  特徵分佈、權重分佈
  等高線 (contour) →  Loss 曲面、梯度下降路徑
  長條圖 (bar)     →  模型預測的機率分佈
  子圖 (subplot)   →  訓練監控面板
  熱力圖 (heatmap) →  Confusion Matrix、Attention 權重

關鍵技能：
  看到 Loss 曲線能判斷：
  - Train ↓ Val ↓ → 正常訓練中
  - Train ↓ Val ↑ → 過擬合！需要 Regularization
  - Train ↑↓ 震盪  → 學習率太大
  - Train 不動     → 學習率太小 或 模型太簡單

═══════════════════════════════════════════
  Phase 0 完成！
  下一步：Phase 1 — 機器學習核心概念
  → phase-1-ml-basics/
═══════════════════════════════════════════
""")
