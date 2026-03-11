"""
=============================================================================
Phase 3-2: GPU 加速 — 混合精度 / 梯度累積 / 效能優化
=============================================================================

為什麼 GPU 比 CPU 快？
  CPU: 幾個強大的核心，擅長複雜的邏輯
  GPU: 幾千個小核心，擅長平行的矩陣運算
  神經網路 = 大量矩陣乘法 → 完美適合 GPU

本檔涵蓋：
  1. GPU 記憶體管理
  2. 混合精度訓練 (AMP)
  3. 梯度累積（小 GPU 也能訓練大 batch）
  4. 效能分析與優化技巧
"""

import torch
import torch.nn as nn
import time
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"裝置: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"顯存: {total_mem:.1f} GB")


# ============================================================================
# 1. GPU 記憶體管理
# ============================================================================
print("\n" + "=" * 60)
print("1. GPU 記憶體管理")
print("=" * 60)

print("""
GPU 記憶體被什麼佔用？

  1. 模型參數:  W, b 的值
  2. 梯度:      每個參數的 .grad（和參數一樣大）
  3. 優化器狀態: Adam 需要 2 份額外的動量 (m, v)
  4. 前向活化值: 反向傳播時需要的中間結果（最大佔用！）
  5. 輸入資料:  一個 batch 的資料

  估算（fp32, 4 bytes per param）：
    1M 參數的模型 ≈ 4 MB (模型) + 4 MB (梯度) + 8 MB (Adam) = 16 MB
    實際還有活化值，取決於 batch size 和模型結構

  顯存不夠怎麼辦？
    1. 減小 batch size
    2. 用混合精度 (AMP) — 省一半記憶體
    3. 用梯度累積 — 小 batch 模擬大 batch
    4. 用梯度 checkpoint — 用時間換空間
""")

if torch.cuda.is_available():
    print("當前 GPU 記憶體使用：")
    print(f"  已用: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"  快取: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

    # 建立一個模型看記憶體變化
    model = nn.Sequential(
        nn.Linear(1000, 2000), nn.ReLU(),
        nn.Linear(2000, 2000), nn.ReLU(),
        nn.Linear(2000, 10),
    ).to(device)

    print(f"\n載入模型後:")
    print(f"  已用: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    x = torch.randn(128, 1000, device=device)
    y = model(x)

    print(f"前向傳播後:")
    print(f"  已用: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    # 清理
    del model, x, y
    torch.cuda.empty_cache()
else:
    print("(無 GPU，以下概念用 CPU 模擬)")


# ============================================================================
# 2. 混合精度訓練 (AMP — Automatic Mixed Precision)
# ============================================================================
print("\n" + "=" * 60)
print("2. 混合精度訓練 (AMP)")
print("=" * 60)

print("""
數值精度：
  FP32 (float32): 32 bit，標準精度
  FP16 (float16): 16 bit，半精度
  BF16 (bfloat16): 16 bit，和 FP32 一樣的指數範圍

AMP 做什麼：
  前向/反向: 用 FP16（快、省記憶體）
  權重更新: 用 FP32（保持精度）

好處：
  - 記憶體省一半 → 可以用更大的 batch
  - 速度快 ~2 倍（在支援的 GPU 上）
  - 幾乎不影響精度
""")

# AMP 的標準寫法
model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 10),
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# GradScaler: 處理 FP16 的梯度 underflow 問題
scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

# 模擬一步訓練
x = torch.randn(64, 784, device=device)
y = torch.randint(0, 10, (64,), device=device)

# --- 不用 AMP ---
start = time.time()
for _ in range(100):
    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
normal_time = time.time() - start

# --- 用 AMP ---
start = time.time()
for _ in range(100):
    with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
        logits = model(x)
        loss = criterion(logits, y)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
amp_time = time.time() - start

print(f"不用 AMP: {normal_time:.3f} 秒 (100 steps)")
print(f"用 AMP:   {amp_time:.3f} 秒 (100 steps)")
if torch.cuda.is_available():
    print(f"加速比:   {normal_time / amp_time:.2f}x")

print("""
AMP 標準模板：
  scaler = torch.amp.GradScaler()

  for batch in loader:
      with torch.amp.autocast(device_type='cuda'):
          logits = model(x)
          loss = criterion(logits, y)

      optimizer.zero_grad()
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
""")


# ============================================================================
# 3. 梯度累積 (Gradient Accumulation)
# ============================================================================
print("=" * 60)
print("3. 梯度累積 — 小 GPU 也能訓練大 Batch")
print("=" * 60)

print("""
問題：想用 batch_size=256 但 GPU 顯存只夠放 batch_size=32

解法：跑 8 次 batch_size=32，累積梯度後才更新一次
  等效 batch = 32 × 8 = 256

  數學上完全等價：
  ∇L(batch=256) ≈ (1/8) × Σ ∇L(batch_i=32)
""")

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

ACCUMULATION_STEPS = 4   # 累積 4 步
real_batch_size = 16      # 實際每步的 batch size
effective_batch = real_batch_size * ACCUMULATION_STEPS   # 等效 batch = 64

# 模擬訓練
print(f"實際 batch: {real_batch_size}")
print(f"累積步數:   {ACCUMULATION_STEPS}")
print(f"等效 batch: {effective_batch}")
print(f"\n模擬 2 個 effective batch 的訓練:")

optimizer.zero_grad()   # 開始前清零

for step in range(ACCUMULATION_STEPS * 2):   # 模擬 8 個 mini-batch
    x = torch.randn(real_batch_size, 784, device=device)
    y = torch.randint(0, 10, (real_batch_size,), device=device)

    logits = model(x)
    loss = criterion(logits, y) / ACCUMULATION_STEPS   # 重要！除以累積步數

    loss.backward()   # 梯度會累積（不 zero_grad）

    # 每 ACCUMULATION_STEPS 步更新一次
    if (step + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
        print(f"  Step {step + 1}: 更新權重 (accumulated {ACCUMULATION_STEPS} batches)")

print("""
關鍵注意：
  loss = loss / ACCUMULATION_STEPS  ← 要除！不然梯度偏大
  不 zero_grad                      ← 讓梯度累積
  每 N 步才 step + zero_grad        ← 累積完才更新
""")


# ============================================================================
# 4. 效能優化技巧
# ============================================================================
print("=" * 60)
print("4. 效能優化技巧")
print("=" * 60)

print("""
常見效能瓶頸與解法：

  瓶頸                        解法
  ─────────────────────────────────────────────────
  資料載入太慢               num_workers=4~8, pin_memory=True
  GPU 利用率低               增大 batch size, 減少 CPU/GPU 傳輸
  模型太大放不下             AMP (半精度), 梯度累積, 梯度 checkpoint
  訓練速度慢                 AMP, 更大 batch, 編譯模型

  torch.compile（PyTorch 2.0+）：
    model = torch.compile(model)
    → 自動把 Python 模型編譯成優化的機器碼
    → 可加速 10-40%
""")

# torch.compile 示範
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
x = torch.randn(256, 784, device=device)

try:
    compiled_model = torch.compile(model)

    # 暖機
    _ = model(x)
    _ = compiled_model(x)

    # 比較速度
    start = time.time()
    for _ in range(200):
        _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    normal_time = time.time() - start

    start = time.time()
    for _ in range(200):
        _ = compiled_model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    compiled_time = time.time() - start

    print(f"\n普通模型:    {normal_time:.3f} 秒 (200 次前向)")
    print(f"torch.compile: {compiled_time:.3f} 秒")
except Exception as e:
    print(f"\ntorch.compile 在此環境不可用: {type(e).__name__}")
    print("（需要 Python 3.12+ 對應版本的 PyTorch，或不同環境）")
    print("概念：torch.compile(model) 可加速 10-40%")

# 梯度 Checkpoint（用時間換空間）
print("\n--- Gradient Checkpointing ---")
print("""
原理：前向時不保存中間結果，反向時重新計算
  好處：大幅減少記憶體（只存少數 checkpoint 點）
  代價：反向傳播變慢（約 1.3 倍）

用法：
  from torch.utils.checkpoint import checkpoint

  class BigModel(nn.Module):
      def forward(self, x):
          x = checkpoint(self.layer1, x)   # 不保存 layer1 的中間結果
          x = checkpoint(self.layer2, x)
          return self.layer3(x)
""")

# Benchmarking 工具
print("\n--- 量測 GPU 時間 ---")
if torch.cuda.is_available():
    model = nn.Linear(1000, 1000).cuda()
    x = torch.randn(512, 1000).cuda()

    # 用 torch.cuda.Event 精確量測
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(100):
        _ = model(x)
    end_event.record()
    torch.cuda.synchronize()

    print(f"  100 次前向: {start_event.elapsed_time(end_event):.2f} ms")
else:
    print("  (無 GPU，跳過 GPU 計時)")


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結")
print("=" * 60)
print("""
GPU 加速工具箱：

  技巧                省記憶體    加速     難度
  ──────────────────────────────────────────
  AMP (混合精度)       ~50%     ~2x     低（加 3 行程式碼）
  梯度累積             N/A      N/A     低（邏輯簡單）
  torch.compile        N/A     10-40%   低（一行程式碼）
  梯度 checkpoint      ~60%    -30%     中
  DataLoader 優化      N/A     顯著     低
  分散式訓練           N/A     ~Nx     高

  最佳實踐組合：
  AMP + 梯度累積 + torch.compile + num_workers > 0

下一步：03_experiment_tracking.py — 實驗管理
""")
