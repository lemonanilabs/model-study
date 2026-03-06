"""
=============================================================================
Phase 2-4: PyTorch nn.Module — 建構模型的標準方式
=============================================================================

上一個檔案你用 NumPy 手刻了完整的神經網路。
現在學 PyTorch 的標準做法：繼承 nn.Module。

nn.Module 幫你處理：
  - 自動追蹤所有參數（不用自己管 W, b）
  - 自動計算梯度（Autograd）
  - 移動到 GPU（.to(device)）
  - 儲存/載入模型

本檔涵蓋：
  1. nn.Linear 解析
  2. 三種建模方式：Sequential / 手寫 forward / ModuleList
  3. 查看參數 / 計算參數量
  4. 模型儲存與載入
"""

import torch
import torch.nn as nn
import numpy as np

# ============================================================================
# 1. nn.Linear — 最基本的層
# ============================================================================
print("=" * 60)
print("1. nn.Linear — 全連接層")
print("=" * 60)

# nn.Linear(in_features, out_features) 等於手刻的 z = X @ W.T + b
layer = nn.Linear(4, 3)    # 4 個輸入特徵 → 3 個輸出

print(f"層: {layer}")
print(f"權重 shape: {layer.weight.shape}")     # (3, 4) — 注意是轉置的！
print(f"偏差 shape: {layer.bias.shape}")        # (3,)
print(f"權重值:\n{layer.weight.data}")

# 前向傳播
x = torch.randn(5, 4)     # 5 個樣本, 4 個特徵
out = layer(x)             # 等於 x @ weight.T + bias
print(f"\n輸入 shape:  {x.shape}")
print(f"輸出 shape:  {out.shape}")        # (5, 3)

# 驗證等價性
manual_out = x @ layer.weight.T + layer.bias
print(f"手算 vs nn.Linear 相同: {torch.allclose(out, manual_out, atol=1e-6)}")


# ============================================================================
# 2. 三種建模方式
# ============================================================================
print("\n" + "=" * 60)
print("2. 三種建模方式")
print("=" * 60)

# ─────────────────────────────────────────
# 方式 A：nn.Sequential（最簡潔）
# ─────────────────────────────────────────
print("\n--- 方式 A: nn.Sequential ---")

model_a = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
print(model_a)

# ─────────────────────────────────────────
# 方式 B：自定義 forward（最常用、最靈活）
# ─────────────────────────────────────────
print("\n--- 方式 B: 自定義 forward ---")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()                    # 一定要呼叫 super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))           # 第 1 層 + ReLU
        x = self.relu(self.fc2(x))           # 第 2 層 + ReLU
        x = self.fc3(x)                      # 第 3 層（不加激活，交給 Loss）
        return x

model_b = MLP(784, 256, 10)
print(model_b)

# 你也可以在 forward 裡加任何 Python 邏輯
class ConditionalMLP(nn.Module):
    """可以根據條件走不同路徑"""
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(784, 256)
        self.branch_a = nn.Linear(256, 10)
        self.branch_b = nn.Linear(256, 10)

    def forward(self, x, use_branch_a=True):
        x = torch.relu(self.shared(x))
        if use_branch_a:
            return self.branch_a(x)
        else:
            return self.branch_b(x)

# ─────────────────────────────────────────
# 方式 C：nn.ModuleList（動態層數）
# ─────────────────────────────────────────
print("\n--- 方式 C: nn.ModuleList ---")

class DynamicMLP(nn.Module):
    def __init__(self, dims):
        """dims: [input, hidden1, hidden2, ..., output]"""
        super().__init__()
        self.layers = nn.ModuleList()        # 不能用普通 list！
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:     # 最後一層不加 ReLU
                x = torch.relu(x)
        return x

model_c = DynamicMLP([784, 512, 256, 128, 10])
print(model_c)

print("""
重要：為什麼不能用普通 list？
  self.layers = [nn.Linear(...)]        ← 錯！PyTorch 不知道這些層的存在
  self.layers = nn.ModuleList([...])    ← 對！PyTorch 會追蹤這些層的參數
""")


# ============================================================================
# 3. 查看參數
# ============================================================================
print("=" * 60)
print("3. 查看模型參數")
print("=" * 60)

model = MLP(784, 256, 10)

# 列出所有參數
print("所有參數 (name, shape):")
for name, param in model.named_parameters():
    print(f"  {name:15s}  shape={str(param.shape):15s}  "
          f"requires_grad={param.requires_grad}")

# 計算總參數量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n總參數量:     {total_params:,}")
print(f"可訓練參數:   {trainable_params:,}")

# 拆解參數量的計算
print(f"""
參數量計算：
  fc1: 784 × 256 + 256 = {784*256 + 256:>8,}  (W: 784×256, b: 256)
  fc2: 256 × 256 + 256 = {256*256 + 256:>8,}
  fc3: 256 × 10  + 10  = {256*10 + 10:>8,}
  ─────────────────────
  總計:                   {784*256+256 + 256*256+256 + 256*10+10:>8,}
""")

# 凍結某些層的參數
print("--- 凍結參數 ---")
for param in model.fc1.parameters():
    param.requires_grad = False

trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"凍結 fc1 後: {trainable_after:,} 可訓練 (少了 {total_params - trainable_after:,})")

# 恢復
for param in model.fc1.parameters():
    param.requires_grad = True


# ============================================================================
# 4. 模型儲存與載入
# ============================================================================
print("\n" + "=" * 60)
print("4. 模型儲存與載入")
print("=" * 60)

import os
os.makedirs('phase-2-deep-learning/checkpoints', exist_ok=True)

model = MLP(784, 256, 10)

# --- 方式 A：只存參數（推薦）---
save_path = 'phase-2-deep-learning/checkpoints/model_weights.pth'
torch.save(model.state_dict(), save_path)
print(f"方式 A: 儲存權重到 {save_path}")

# 載入
model_loaded = MLP(784, 256, 10)                # 先建立相同結構
model_loaded.load_state_dict(torch.load(save_path, weights_only=True))
print(f"方式 A: 載入完成")

# 驗證
x_test = torch.randn(1, 784)
with torch.no_grad():
    out1 = model(x_test)
    out2 = model_loaded(x_test)
print(f"載入後結果相同: {torch.allclose(out1, out2)}")

# --- 方式 B：存完整 checkpoint（訓練中斷恢復用）---
checkpoint_path = 'phase-2-deep-learning/checkpoints/checkpoint.pth'
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

torch.save({
    'epoch': 50,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.123,
}, checkpoint_path)
print(f"\n方式 B: 儲存完整 checkpoint（含 optimizer 狀態和 epoch）")

# 載入 checkpoint
checkpoint = torch.load(checkpoint_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
print(f"方式 B: 從 epoch {start_epoch} 繼續訓練")

print("""
儲存策略：
  state_dict()           → 部署、分享模型
  完整 checkpoint         → 訓練中斷恢復

  不要用 torch.save(model, path)!
  它會把整個 Python class 序列化，不同版本會出問題。
  永遠存 state_dict()。
""")


# ============================================================================
# 5. 完整範例：用 nn.Module 訓練 Iris
# ============================================================================
print("=" * 60)
print("5. 完整範例：nn.Module 訓練 Iris")
print("=" * 60)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 準備資料
iris = load_iris()
X, y = iris.data, iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# 模型
model = MLP(input_dim=4, hidden_dim=32, output_dim=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 訓練
print(f"{'Epoch':>5} {'Loss':>8} {'Train':>8} {'Test':>8}")
print("-" * 34)

for epoch in range(200):
    # 前向
    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)

    # 反向
    optimizer.zero_grad()    # 清除梯度
    loss.backward()          # 計算梯度
    optimizer.step()         # 更新權重

    if epoch % 25 == 0 or epoch == 199:
        with torch.no_grad():
            train_acc = (model(X_train_t).argmax(1) == y_train_t).float().mean()
            test_acc = (model(X_test_t).argmax(1) == y_test_t).float().mean()
        print(f"{epoch:5d} {loss.item():8.4f} {train_acc:8.2%} {test_acc:8.2%}")


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結")
print("=" * 60)
print("""
nn.Module 的核心模式：

  class MyModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.layer = nn.Linear(...)    # 定義層

      def forward(self, x):
          return self.layer(x)           # 定義前向計算

  model = MyModel()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # 訓練一步
  logits = model(x)                  # 前向
  loss = criterion(logits, y)        # 算 Loss
  optimizer.zero_grad()              # 清梯度
  loss.backward()                    # 算梯度
  optimizer.step()                   # 更新權重

  這五行就是深度學習訓練的標準流程，永遠都是這個模式。

下一步：05_activation_functions.py — 激活函數全解
""")
