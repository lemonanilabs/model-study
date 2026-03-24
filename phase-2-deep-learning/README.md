# Phase 2 — 深度學習基礎

## 目標
從「一個神經元」建構到完整的神經網路，用 PyTorch 訓練第一個深度學習模型。

## 檔案順序

| 順序 | 檔案 | 主題 |
|------|------|------|
| 1 | `01_pytorch_basics.py` | PyTorch 基礎：Tensor / Autograd / Device |
| 2 | `02_autograd_deep_dive.py` | 自動微分：手刻 vs PyTorch 對比 |
| 3 | `03_neural_network_from_scratch.py` | 用純 NumPy 手刻完整的神經網路 |
| 4 | `04_pytorch_nn_module.py` | nn.Module / nn.Linear / 建構模型 |
| 5 | `05_activation_functions.py` | 激活函數全解：ReLU / Sigmoid / GELU ... |
| 6 | `06_loss_and_optimizers.py` | 損失函數 + 優化器原理與比較 |
| 7 | `07_training_loop.py` | 完整訓練流程：MNIST 手寫辨識 |
| 8 | `08_techniques.py` | 實用技巧：BatchNorm / Dropout / LR Schedule |
| 9 | `09_representation_learning.py` | 表示學習：嵌入空間、度量學習、Triplet Loss |

## 學習路線

```
01 PyTorch Tensor (= NumPy + GPU + Autograd)
      ↓
02 Autograd (自動算梯度，不用手寫反向傳播)
      ↓
03 手刻神經網路 (NumPy，徹底理解每一步)
      ↓
04 nn.Module (用 PyTorch 重寫，對比手刻版)
      ↓
05-06 激活函數 + 損失函數 + 優化器
      ↓
07 完整訓練 MNIST（整合所有概念）
      ↓
08 進階技巧（讓模型訓練得更好）
      ↓
09 表示學習（理解模型如何學會「看世界」）
```
