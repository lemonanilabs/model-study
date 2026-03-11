"""
=============================================================================
Phase 3-4: 分散式訓練 — 多 GPU / 多機訓練概念
=============================================================================

一張 GPU 不夠怎麼辦？用多張。

本檔涵蓋：
  1. 資料平行 vs 模型平行
  2. DataParallel (DP) — 最簡單但有瓶頸
  3. DistributedDataParallel (DDP) — 實務首選
  4. FSDP / DeepSpeed — 訓練超大模型
  5. 完整 DDP 訓練腳本模板
"""

import torch
import torch.nn as nn

# ============================================================================
# 1. 平行策略概覽
# ============================================================================
print("=" * 60)
print("1. 平行策略概覽")
print("=" * 60)

print("""
兩種基本策略：

  ┌─────────────────────────────────────────────────┐
  │  資料平行 (Data Parallelism)                      │
  │  每張 GPU 都有完整的模型副本                        │
  │  資料切成 N 份，每張 GPU 處理一份                    │
  │  梯度匯總後更新                                    │
  │                                                   │
  │  GPU 0: model(batch_0) → grad_0 ─┐                │
  │  GPU 1: model(batch_1) → grad_1 ─┤→ 平均 → 更新   │
  │  GPU 2: model(batch_2) → grad_2 ─┘                │
  │                                                   │
  │  適合：模型放得下一張 GPU                           │
  └─────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────┐
  │  模型平行 (Model Parallelism)                      │
  │  模型太大，一張 GPU 放不下                          │
  │  把模型切成幾段，放在不同 GPU 上                     │
  │                                                   │
  │  GPU 0: Layer 1-10  →  output ─→ GPU 1            │
  │  GPU 1: Layer 11-20 →  output ─→ GPU 2            │
  │  GPU 2: Layer 21-30 →  final output               │
  │                                                   │
  │  適合：超大模型 (LLM, 數十億參數)                   │
  └─────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────┐
  │  FSDP (Fully Sharded Data Parallel)              │
  │  資料平行 + 模型也切分                             │
  │  每張 GPU 只存一部分模型參數                        │
  │  需要時才從其他 GPU 收集                            │
  │                                                   │
  │  適合：大模型 + 多 GPU                             │
  └─────────────────────────────────────────────────┘
""")


# ============================================================================
# 2. DataParallel (DP) — 最簡單
# ============================================================================
print("=" * 60)
print("2. DataParallel — 一行搞定（但有瓶頸）")
print("=" * 60)

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"使用 {torch.cuda.device_count()} 張 GPU (DataParallel)")
else:
    print(f"只有 {max(torch.cuda.device_count(), 1)} 張 GPU，DP 無效")

print("""
DataParallel 的問題：
  1. GPU 0 是瓶頸：收集所有結果、計算 Loss、分發梯度
  2. GIL 限制：Python 的 GIL 影響多線程效率
  3. 通訊開銷：每次 forward 都要複製模型到其他 GPU

  → 實務中幾乎不用 DP，用 DDP
""")


# ============================================================================
# 3. DistributedDataParallel (DDP) — 實務首選
# ============================================================================
print("\n" + "=" * 60)
print("3. DistributedDataParallel (DDP)")
print("=" * 60)

print("""
DDP 的改進：
  - 每張 GPU 跑一個獨立的 process（不受 GIL 影響）
  - 梯度用 All-Reduce 同步（每張 GPU 平等，沒有瓶頸）
  - 前向時不需要複製模型（每個 process 有自己的副本）

  啟動方式：
    torchrun --nproc_per_node=4 train.py

  效率：接近線性加速（4 GPU ≈ 3.8x 速度）
""")

# DDP 訓練腳本模板（需要用 torchrun 啟動，這裡展示結構）
ddp_template = '''
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def main():
    # 1. 初始化分散式環境
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # 2. 建立模型（DDP 包裝）
    model = MyModel().to(device)
    model = DDP(model, device_ids=[local_rank])

    # 3. DistributedSampler（確保每張 GPU 拿到不同的資料）
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,      # 取代 shuffle=True
        num_workers=4,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 4. 訓練迴圈（和單 GPU 幾乎一樣）
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)     # 重要！讓每個 epoch 打亂不同

        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()             # DDP 自動同步梯度
            optimizer.step()

        # 5. 只在 rank 0 儲存/記錄
        if local_rank == 0:
            torch.save(model.module.state_dict(), 'model.pth')
            print(f'Epoch {epoch}: loss={loss.item():.4f}')

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
'''
print("DDP 訓練腳本模板：")
print(ddp_template)

print("""
DDP 重點：
  1. dist.init_process_group('nccl')   → 初始化通訊
  2. DDP(model)                         → 包裝模型
  3. DistributedSampler                 → 資料分片
  4. sampler.set_epoch(epoch)           → 每 epoch 重新打亂
  5. model.module.state_dict()          → 存模型時要用 .module
  6. if rank == 0: save/log             → 只有主 process 儲存

  啟動：torchrun --nproc_per_node=4 train.py
""")


# ============================================================================
# 4. FSDP / DeepSpeed — 超大模型
# ============================================================================
print("=" * 60)
print("4. FSDP / DeepSpeed — 訓練數十億參數的模型")
print("=" * 60)

print("""
當模型大到一張 GPU 放不下時：

  DDP: 每張 GPU 存完整模型 → 4 張 24GB GPU = 只能放 24GB 的模型
  FSDP: 模型切片分散存放 → 4 張 24GB GPU = 可以放 ~96GB 的模型

  FSDP (Fully Sharded Data Parallel):
    - PyTorch 原生支援
    - 把模型參數、梯度、優化器狀態都切片
    - 需要時才從其他 GPU 收集

  DeepSpeed (Microsoft):
    - ZeRO Stage 1: 切分優化器狀態
    - ZeRO Stage 2: + 切分梯度
    - ZeRO Stage 3: + 切分模型參數（= FSDP）
    - ZeRO-Offload: 把部分資料放到 CPU 記憶體

  實際的大模型訓練通常結合多種策略：
    - Tensor Parallelism（把每一層切分到多張 GPU）
    - Pipeline Parallelism（把不同層放到不同 GPU）
    - Data Parallelism（多組上述配置平行處理不同資料）
""")

# FSDP 範例
fsdp_template = '''
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = MyLargeModel().to(device)
model = FSDP(
    model,
    auto_wrap_policy=size_based_auto_wrap_policy,
    mixed_precision=MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    ),
)
'''
print("FSDP 範例：")
print(fsdp_template)


# ============================================================================
# 5. 策略選擇指南
# ============================================================================
print("=" * 60)
print("5. 策略選擇指南")
print("=" * 60)

print("""
  情境                        推薦策略
  ──────────────────────────────────────────────
  1 張 GPU                   單 GPU + AMP
  2-8 張 GPU, 模型放得下      DDP + AMP
  2-8 張 GPU, 模型放不下      FSDP / DeepSpeed ZeRO-3
  多機多卡                    DDP / FSDP + Slurm
  訓練 LLM (7B+)             FSDP + AMP + Gradient Checkpoint

  大部分人的情況：
    個人 1-2 張 GPU → 單 GPU + AMP + 梯度累積 就夠了
    公司 4-8 張 GPU → DDP

═══════════════════════════════════════════
  Phase 3 完成！
  你現在知道模型訓練的實務技巧了。
  下一步：Phase 4 — 選一個領域深入 (CV / NLP)
═══════════════════════════════════════════
""")
