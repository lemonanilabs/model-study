"""
=============================================================================
Phase 3-3: 實驗管理 — TensorBoard / 超參數搜尋
=============================================================================

跑了 100 次實驗，哪次最好？用了什麼參數？
沒有實驗管理，你會在一週內迷失。

本檔涵蓋：
  1. TensorBoard 基礎
  2. 超參數搜尋（Grid / Random / Bayesian）
  3. 完整的實驗管理模板
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
from datetime import datetime

# ============================================================================
# 1. TensorBoard
# ============================================================================
print("=" * 60)
print("1. TensorBoard — 視覺化訓練過程")
print("=" * 60)

print("""
TensorBoard 能記錄：
  - Loss / Accuracy 曲線
  - 學習率變化
  - 模型的計算圖
  - 權重/梯度的分佈
  - 圖片/音訊/文字等

安裝：pip install tensorboard
啟動：tensorboard --logdir=runs
""")

from torch.utils.tensorboard import SummaryWriter

# 建立 writer
log_dir = f"phase-3-training/runs/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir)
print(f"TensorBoard log 目錄: {log_dir}")

# 模擬訓練並記錄
np.random.seed(42)
for epoch in range(50):
    train_loss = 2.0 * np.exp(-0.05 * epoch) + 0.1 + np.random.normal(0, 0.02)
    val_loss = 2.0 * np.exp(-0.04 * epoch) + 0.2 + np.random.normal(0, 0.03)
    train_acc = 1 - 0.6 * np.exp(-0.06 * epoch) + np.random.normal(0, 0.01)
    lr = 0.001 * 0.95 ** epoch

    # 記錄 scalar
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Learning_Rate', lr, epoch)

# 記錄模型結構
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
writer.add_graph(model, torch.randn(1, 784))

# 記錄超參數和最終結果
writer.add_hparams(
    {'lr': 0.001, 'batch_size': 32, 'hidden_dim': 256},
    {'hparam/accuracy': 0.95, 'hparam/loss': 0.15}
)

writer.close()
print(f"記錄完成！")
print(f"啟動方式: tensorboard --logdir=phase-3-training/runs")


# ============================================================================
# 2. 超參數搜尋
# ============================================================================
print("\n" + "=" * 60)
print("2. 超參數搜尋")
print("=" * 60)

print("""
常見需要調的超參數：
  - 學習率 (lr): 最重要！
  - Batch size: 16, 32, 64, 128, 256
  - 隱藏層大小: 64, 128, 256, 512
  - Dropout 率: 0.1, 0.2, 0.3, 0.5
  - Weight decay: 1e-4, 1e-3, 1e-2
  - 優化器: Adam vs AdamW vs SGD
""")

# 準備資料
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

digits = load_digits()
scaler = StandardScaler()
X = torch.tensor(scaler.fit_transform(digits.data), dtype=torch.float32)
y = torch.tensor(digits.target, dtype=torch.long)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(config):
    """用給定的超參數訓練並回傳結果"""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(64, config['hidden_dim']),
        nn.ReLU(),
        nn.Dropout(config['dropout']),
        nn.Linear(config['hidden_dim'], 10),
    )
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_tr, y_tr),
                        batch_size=config['batch_size'], shuffle=True)

    for epoch in range(config['epochs']):
        model.train()
        for bx, by in loader:
            loss = criterion(model(bx), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_acc = (model(X_te).argmax(1) == y_te).float().mean().item()
    return test_acc


# --- Grid Search ---
print("\n--- Grid Search（窮舉搜尋）---")
grid = {
    'lr': [0.001, 0.01],
    'hidden_dim': [64, 128],
    'dropout': [0.1],
    'weight_decay': [0.0],
    'batch_size': [32],
    'epochs': [50],
}

import itertools
keys = grid.keys()
values = grid.values()
configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"共 {len(configs)} 種組合")
results = []
for i, config in enumerate(configs):
    acc = train_and_evaluate(config)
    results.append((acc, config))
    print(f"  [{i+1}/{len(configs)}] lr={config['lr']}, hidden={config['hidden_dim']} "
          f"→ Acc={acc:.2%}")

best_acc, best_config = max(results, key=lambda x: x[0])
print(f"\n最佳: Acc={best_acc:.2%}, config={best_config}")


# --- Random Search ---
print("\n--- Random Search（隨機搜尋，通常更有效率）---")
np.random.seed(42)
n_trials = 8

print(f"隨機嘗試 {n_trials} 組")
random_results = []
for i in range(n_trials):
    config = {
        'lr': 10 ** np.random.uniform(-4, -2),         # log 均勻: 1e-4 ~ 1e-2
        'hidden_dim': int(np.random.choice([64, 128, 256])),
        'dropout': np.random.uniform(0.0, 0.5),
        'weight_decay': 10 ** np.random.uniform(-5, -2),
        'batch_size': int(np.random.choice([16, 32, 64])),
        'epochs': 50,
    }
    acc = train_and_evaluate(config)
    random_results.append((acc, config))
    print(f"  [{i+1}/{n_trials}] lr={config['lr']:.5f}, "
          f"hidden={config['hidden_dim']}, drop={config['dropout']:.2f} "
          f"→ Acc={acc:.2%}")

best_acc_r, best_config_r = max(random_results, key=lambda x: x[0])
print(f"\n最佳: Acc={best_acc_r:.2%}")

print("""
Grid vs Random Search：
  Grid:   窮舉所有組合 → 參數少時可以用
  Random: 隨機抽樣 → 高維空間更有效率
  Bayesian (Optuna): 用過去結果指導下一次嘗試 → 最聰明

推薦工具：
  - Optuna: Python 超參數搜尋框架
  - W&B (Weights & Biases): 實驗追蹤 + Sweep
  - Ray Tune: 分散式超參數搜尋
""")


# ============================================================================
# 3. 實驗管理模板
# ============================================================================
print("=" * 60)
print("3. 實驗管理模板")
print("=" * 60)

class ExperimentTracker:
    """簡單的實驗記錄器"""

    def __init__(self, experiment_name, log_dir='phase-3-training/experiments'):
        self.name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics = {}
        self.config = {}

    def log_config(self, config):
        self.config = config
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def log_metric(self, name, value, step):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({'step': step, 'value': value})

    def save(self):
        with open(os.path.join(self.log_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"實驗儲存至: {self.log_dir}")

    def summary(self):
        print(f"\n實驗: {self.name}")
        for name, values in self.metrics.items():
            last = values[-1]['value']
            best = min(values, key=lambda x: x['value'])['value'] if 'loss' in name.lower() \
                else max(values, key=lambda x: x['value'])['value']
            print(f"  {name}: last={last:.4f}, best={best:.4f}")


# 使用
tracker = ExperimentTracker('demo_experiment')
tracker.log_config({'lr': 0.001, 'hidden_dim': 128, 'epochs': 50})

for epoch in range(50):
    fake_loss = 2.0 * np.exp(-0.05 * epoch) + 0.1
    fake_acc = 1 - 0.6 * np.exp(-0.06 * epoch)
    tracker.log_metric('train_loss', fake_loss, epoch)
    tracker.log_metric('train_acc', fake_acc, epoch)

tracker.save()
tracker.summary()


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結")
print("=" * 60)
print("""
實驗管理的核心原則：

  1. 每次實驗都要記錄：超參數、Loss 曲線、最終結果
  2. 能重現：記錄 random seed、程式碼版本
  3. 能比較：統一的格式和指標

  工具選擇：
    個人小專案  → TensorBoard + JSON 記錄
    團隊/大專案 → Weights & Biases (W&B)
    超參數搜尋  → Optuna

  最重要的超參數排名：
    1. 學習率（最關鍵）
    2. Batch size
    3. 模型大小
    4. 正則化（dropout, weight decay）

下一步：04_distributed_training.py — 分散式訓練概念
""")
