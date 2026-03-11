"""
=============================================================================
Phase 3-1: 資料處理 — 模型好不好，資料佔一半
=============================================================================

"Garbage in, garbage out"
模型再好，資料爛也沒用。資料處理是 ML 工程中最花時間的部分。

本檔涵蓋：
  1. PyTorch Dataset & DataLoader 進階
  2. 資料增強 (Data Augmentation)
  3. 資料標準化策略
  4. 處理不平衡資料
  5. 自定義 Dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np

# ============================================================================
# 1. 自定義 Dataset
# ============================================================================
print("=" * 60)
print("1. 自定義 Dataset — PyTorch 資料的標準介面")
print("=" * 60)

print("""
所有 Dataset 只需要實作三個方法：
  __init__:     載入/初始化資料
  __len__:      回傳資料總數
  __getitem__:  回傳第 i 筆資料（支援 dataset[i] 語法）
""")

class CustomDataset(Dataset):
    """自定義 Dataset 範例"""

    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


# 建立範例資料
np.random.seed(42)
X_data = np.random.randn(500, 10).astype(np.float32)
y_data = np.random.randint(0, 3, size=500)

dataset = CustomDataset(X_data, y_data)
print(f"Dataset 大小: {len(dataset)}")
print(f"第 0 筆: X.shape={dataset[0][0].shape}, y={dataset[0][1]}")

# DataLoader 進階選項
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,              # 每個 epoch 打亂
    num_workers=0,             # 多線程載入（0=主線程）
    drop_last=True,            # 丟棄不滿一個 batch 的尾巴
    pin_memory=True,           # 加速 CPU→GPU 傳輸
)

print(f"\nDataLoader:")
print(f"  batch_size=32, total batches={len(loader)}")

for batch_x, batch_y in loader:
    print(f"  一個 batch: X={batch_x.shape}, y={batch_y.shape}")
    break

print("""
DataLoader 重要參數：
  shuffle=True      → 訓練時必須打亂，測試時不用
  num_workers=4     → 多線程預載資料（大資料集加速明顯）
  pin_memory=True   → 使用 pinned memory，加速 GPU 傳輸
  drop_last=True    → BatchNorm 需要 batch > 1，避免最後一個只有 1 筆
""")


# ============================================================================
# 2. 資料增強 (Data Augmentation)
# ============================================================================
print("=" * 60)
print("2. 資料增強 — 用變換產生更多訓練資料")
print("=" * 60)

print("""
核心思想：
  「模型應該要認出翻轉、旋轉、裁切後的貓，還是貓」
  不是真的增加資料量，而是在訓練時隨機變換輸入。

常見增強（圖像）：
  - 隨機水平翻轉
  - 隨機裁切
  - 色彩抖動（亮度、對比、飽和度）
  - 隨機旋轉
  - CutOut / MixUp / CutMix
""")

import torchvision.transforms as T

# 圖像增強 pipeline
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),   # 隨機裁切到 224×224
    T.RandomHorizontalFlip(p=0.5),                 # 50% 機率水平翻轉
    T.ColorJitter(brightness=0.2, contrast=0.2),   # 色彩抖動
    T.ToTensor(),                                   # PIL → Tensor, [0,255] → [0,1]
    T.Normalize(mean=[0.485, 0.456, 0.406],        # ImageNet 標準化
                std=[0.229, 0.224, 0.225]),
])

# 測試時不做隨機增強
test_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

print("訓練 transform:")
for t in train_transform.transforms:
    print(f"  {t}")

print(f"\n測試 transform:")
for t in test_transform.transforms:
    print(f"  {t}")

# 對 Tensor 類型資料做增強（非圖像）
print("\n--- 對表格/數值資料的增強 ---")

class GaussianNoise:
    """加入高斯噪音"""
    def __init__(self, std=0.01):
        self.std = std
    def __call__(self, x):
        return x + torch.randn_like(x) * self.std

class RandomMask:
    """隨機遮蔽部分特徵"""
    def __init__(self, p=0.1):
        self.p = p
    def __call__(self, x):
        mask = torch.rand_like(x) > self.p
        return x * mask

# 使用自定義增強
dataset_aug = CustomDataset(X_data, y_data, transform=GaussianNoise(std=0.05))
x_orig = dataset.X[0]
x_aug1 = dataset_aug[0][0]
x_aug2 = dataset_aug[0][0]
print(f"原始:   {x_orig[:5].numpy().round(3)}")
print(f"增強 1: {x_aug1[:5].numpy().round(3)}")
print(f"增強 2: {x_aug2[:5].numpy().round(3)}")
print("→ 每次取同一筆資料會得到不同的增強結果")


# ============================================================================
# 3. 資料標準化策略
# ============================================================================
print("\n" + "=" * 60)
print("3. 資料標準化策略")
print("=" * 60)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

X_sample = np.random.randn(100, 4) * [1, 10, 100, 1000] + [0, 5, 50, 500]

print(f"原始資料各特徵範圍:")
for i in range(4):
    print(f"  Feature {i}: [{X_sample[:, i].min():.1f}, {X_sample[:, i].max():.1f}]")

# StandardScaler: (x - mean) / std → 平均 0，標準差 1
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X_sample)

# MinMaxScaler: (x - min) / (max - min) → [0, 1]
scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X_sample)

print(f"\nStandardScaler 後:")
for i in range(4):
    print(f"  Feature {i}: mean={X_std[:, i].mean():.4f}, std={X_std[:, i].std():.4f}")

print(f"\nMinMaxScaler 後:")
for i in range(4):
    print(f"  Feature {i}: [{X_mm[:, i].min():.4f}, {X_mm[:, i].max():.4f}]")

print("""
選擇建議：
  StandardScaler → 大多數情況（梯度下降類演算法）
  MinMaxScaler   → 需要固定範圍時（如 [0,1] 輸入）
  Log Transform  → 長尾分佈的資料（如收入、瀏覽次數）

  重要：用訓練集 fit，再 transform 測試集
  scaler.fit(X_train)
  X_train_scaled = scaler.transform(X_train)
  X_test_scaled = scaler.transform(X_test)    # 不能 fit 測試集！
""")


# ============================================================================
# 4. 處理不平衡資料
# ============================================================================
print("=" * 60)
print("4. 處理不平衡資料")
print("=" * 60)

# 製造不平衡資料：類別 0 有 900 筆，類別 1 只有 100 筆
np.random.seed(42)
X_imb = np.random.randn(1000, 5).astype(np.float32)
y_imb = np.array([0] * 900 + [1] * 100)

print(f"不平衡資料: Class 0 = {(y_imb == 0).sum()}, Class 1 = {(y_imb == 1).sum()}")

# --- 方法 1: Weighted Loss ---
print("\n--- 方法 1: Weighted Loss ---")
class_counts = np.bincount(y_imb)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

criterion_weighted = nn.CrossEntropyLoss(weight=weights_tensor)
print(f"  類別權重: {weights_tensor.numpy().round(4)}")
print(f"  少數類別的權重更大，迫使模型關注它")

# --- 方法 2: WeightedRandomSampler ---
print("\n--- 方法 2: WeightedRandomSampler ---")
sample_weights = np.array([1.0 / class_counts[label] for label in y_imb])
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(y_imb),
    replacement=True,
)

dataset_imb = CustomDataset(X_imb, y_imb)
loader_balanced = DataLoader(dataset_imb, batch_size=64, sampler=sampler)

# 檢查一個 epoch 的實際類別分佈
all_labels = []
for _, labels in loader_balanced:
    all_labels.extend(labels.numpy())
all_labels = np.array(all_labels)
print(f"  Sampler 後每 epoch 的類別分佈:")
print(f"  Class 0: {(all_labels == 0).sum()}, Class 1: {(all_labels == 1).sum()}")
print(f"  → 現在大致平衡了！")

# --- 方法 3: Oversampling / Undersampling ---
print("\n--- 方法 3: Over/Under Sampling ---")
print("  Oversampling: 複製少數類別的樣本")
print("  Undersampling: 隨機刪除多數類別的樣本")
print("  SMOTE: 在少數類別的樣本之間插值生成新樣本")

print("""
方法選擇：
  輕微不平衡 (3:1)     → Weighted Loss 就夠了
  嚴重不平衡 (100:1)   → WeightedRandomSampler + Weighted Loss
  極度不平衡 (10000:1)  → Focal Loss + 特殊採樣策略

  醫療/詐騙偵測通常是嚴重不平衡場景
""")


# ============================================================================
# 5. 實戰：完整的資料管線
# ============================================================================
print("=" * 60)
print("5. 實戰：完整的資料管線模板")
print("=" * 60)

class RealWorldDataset(Dataset):
    """真實世界資料集模板"""

    def __init__(self, data_path=None, split='train', transform=None):
        # 模擬資料（真實場景從 data_path 載入）
        np.random.seed(42 if split == 'train' else 0)
        n = 800 if split == 'train' else 200
        self.X = np.random.randn(n, 20).astype(np.float32)
        self.y = np.random.randint(0, 5, n)
        self.transform = transform
        self.split = split

        # 標準化（只用訓練集的統計量）
        if split == 'train':
            self.mean = self.X.mean(axis=0)
            self.std = self.X.std(axis=0)
        # 測試集要從外部傳入訓練集的 mean/std

    def set_normalization(self, mean, std):
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mean) / (self.std + 1e-8)
        x = torch.tensor(x)
        y = torch.tensor(self.y[idx], dtype=torch.long)

        if self.transform:
            x = self.transform(x)

        return x, y


# 使用
train_ds = RealWorldDataset(split='train', transform=GaussianNoise(0.01))
test_ds = RealWorldDataset(split='test')
test_ds.set_normalization(train_ds.mean, train_ds.std)   # 用訓練集統計量

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

print(f"訓練集: {len(train_ds)} 樣本, {len(train_loader)} batches")
print(f"測試集: {len(test_ds)} 樣本, {len(test_loader)} batches")

for x, y in train_loader:
    print(f"一個 batch: x={x.shape}, y={y.shape}")
    print(f"x 範圍: [{x.min():.2f}, {x.max():.2f}]")
    break


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結")
print("=" * 60)
print("""
資料處理檢查清單：

  □ 資料標準化（用訓練集 fit，再 transform 測試集）
  □ 訓練時有資料增強，測試時沒有
  □ 不平衡資料有處理（Weighted Loss 或 Sampler）
  □ DataLoader: shuffle=True (train), False (test)
  □ 資料沒有 leak（測試集資訊沒有流入訓練集）
  □ num_workers > 0（大資料集加速載入）

下一步：02_gpu_acceleration.py — GPU 加速與混合精度
""")
