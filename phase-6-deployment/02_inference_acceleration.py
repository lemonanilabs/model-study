"""
=============================================================================
Phase 6-2: 推論加速 — 讓模型跑得更快
=============================================================================

模型訓練好了，但推論太慢怎麼辦？

  速度瓶頸在哪？
  - 計算量：矩陣乘法太多
  - 記憶體頻寬：資料搬移太慢
  - 框架開銷：Python / PyTorch 的動態圖有額外成本

本檔涵蓋：
  1. 推論加速的全景圖
  2. ONNX — 跨框架的模型格式
  3. TorchScript — PyTorch 的編譯
  4. KV Cache — Transformer 推論的關鍵優化
  5. Speculative Decoding — 加速 LLM 生成
  6. 其他加速技術
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import io
import math

# ============================================================================
# 1. 推論加速全景圖
# ============================================================================
print("=" * 60)
print("1. 推論加速全景圖")
print("=" * 60)

print("""
  加速層級：

  ┌──────────────────────────────────────────────┐
  │ 演算法層級                                     │
  │  • KV Cache (Transformer 專用)               │
  │  • Speculative Decoding                       │
  │  • Flash Attention                            │
  │  • Continuous Batching                        │
  ├──────────────────────────────────────────────┤
  │ 模型層級                                       │
  │  • 量化 (INT8/INT4)                           │
  │  • 蒸餾 (Knowledge Distillation)              │
  │  • 剪枝 (Pruning)                            │
  │  • 算子融合 (Conv+BN+ReLU → 一個算子)          │
  ├──────────────────────────────────────────────┤
  │ 系統層級                                       │
  │  • ONNX Runtime / TensorRT / TVM             │
  │  • torch.compile                              │
  │  • TorchScript                                │
  │  • vLLM / TGI (LLM 專用)                     │
  ├──────────────────────────────────────────────┤
  │ 硬體層級                                       │
  │  • GPU (CUDA Cores / Tensor Cores)           │
  │  • NPU / TPU                                  │
  │  • CPU 優化 (AVX-512, AMX)                   │
  └──────────────────────────────────────────────┘
""")


# ============================================================================
# 2. ONNX — 跨框架模型格式
# ============================================================================
print("=" * 60)
print("2. ONNX — 一次匯出，到處推論")
print("=" * 60)

print("""
ONNX (Open Neural Network Exchange):
  - 由 Microsoft 和 Facebook 創建
  - 統一的模型格式，支援 PyTorch, TensorFlow, etc.
  - 用 ONNX Runtime 推論，通常比 PyTorch 快 2-3x

  流程：
  PyTorch Model → 匯出 ONNX → ONNX Runtime 推論
                              → TensorRT 推論
                              → 部署到邊緣裝置

  匯出範例：
  torch.onnx.export(
      model,
      dummy_input,
      "model.onnx",
      input_names=['input'],
      output_names=['output'],
      dynamic_axes={'input': {0: 'batch_size'}},
  )

  推論範例（ONNX Runtime）：
  import onnxruntime as ort
  session = ort.InferenceSession("model.onnx")
  result = session.run(None, {"input": input_data})
""")

# 匯出 ONNX 示範
model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
model.eval()
dummy = torch.randn(1, 64)

try:
    # 匯出到 bytes（不寫檔案）
    buffer = io.BytesIO()
    torch.onnx.export(model, dummy, buffer, input_names=['input'], output_names=['output'])
    onnx_size = buffer.tell()
    print(f"模型匯出為 ONNX: {onnx_size:,} bytes")
except Exception as e:
    print(f"ONNX 匯出需要額外套件: {e}")
    print("→ pip install onnxscript onnx  即可使用")

# PyTorch 推論速度
x = torch.randn(100, 64)
start = time.time()
for _ in range(1000):
    _ = model(x)
pytorch_time = time.time() - start
print(f"PyTorch 推論 (1000 次): {pytorch_time:.3f}s")

print("""
ONNX Runtime 加速原理：
  1. 圖優化：合併相鄰的算子（Conv+BN → FusedConvBN）
  2. 記憶體優化：減少不必要的資料複製
  3. 硬體專用實作：用 CPU/GPU 最優的 kernel
  4. 平行執行：獨立的算子平行計算
""")


# ============================================================================
# 3. TorchScript — PyTorch 的編譯
# ============================================================================
print("\n" + "=" * 60)
print("3. TorchScript — 消除 Python 開銷")
print("=" * 60)

print("""
PyTorch 是動態圖框架（eager mode）：
  每次 forward 都要解析 Python 程式碼 → 有開銷

TorchScript 把模型「編譯」成靜態圖：
  - 不需要 Python 直譯器
  - 可以部署到 C++ 環境
  - 通常速度提升 10-30%

兩種方式：
  1. Tracing（追蹤）：跑一次 forward，記錄所有操作
     scripted = torch.jit.trace(model, example_input)

  2. Scripting（腳本化）：直接解析 Python 程式碼
     scripted = torch.jit.script(model)

  Tracing: 簡單但不能有 if/loop（動態分支）
  Scripting: 複雜但支援動態控制流
""")

# TorchScript 示範
model = nn.Sequential(
    nn.Linear(64, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
model.eval()
x = torch.randn(100, 64)

# Trace
traced = torch.jit.trace(model, x)

# 速度比較
n_iters = 2000

start = time.time()
for _ in range(n_iters):
    _ = model(x)
eager_time = time.time() - start

start = time.time()
for _ in range(n_iters):
    _ = traced(x)
jit_time = time.time() - start

print(f"Eager mode: {eager_time:.3f}s ({n_iters} iters)")
print(f"TorchScript: {jit_time:.3f}s ({n_iters} iters)")
print(f"加速比: {eager_time / jit_time:.2f}x")

# 儲存 TorchScript 模型
buffer = io.BytesIO()
torch.jit.save(traced, buffer)
print(f"TorchScript 模型大小: {buffer.tell():,} bytes")


# ============================================================================
# 4. KV Cache — Transformer 推論的核心優化
# ============================================================================
print("\n" + "=" * 60)
print("4. KV Cache — Transformer 不重複計算")
print("=" * 60)

print("""
GPT 生成的問題：
  生成 "The cat sat on the mat"
  Step 1: 輸入 "The"           → 計算 K, V for "The"
  Step 2: 輸入 "The cat"       → 重新計算 K, V for "The" 和 "cat"
  Step 3: 輸入 "The cat sat"   → 重新計算所有的 K, V

  → "The" 的 K, V 被算了 6 次！浪費！

KV Cache 的解法：
  把已經算過的 K, V 存起來，不重複計算

  Step 1: 計算 K_1, V_1     → cache: [K_1, V_1]
  Step 2: 只算 K_2, V_2     → cache: [K_1,K_2, V_1,V_2]
  Step 3: 只算 K_3, V_3     → cache: [K_1,K_2,K_3, V_1,V_2,V_3]

  每步只需要計算新 token 的 K, V → 速度大幅提升

  但是：KV Cache 會佔用大量 GPU 記憶體
  LLaMA-7B, context=2048:
    KV Cache ≈ 2 × 32 layers × 2048 × 4096 × 2 bytes ≈ 1 GB
""")


class CausalAttentionWithCache(nn.Module):
    """帶 KV Cache 的 Attention"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # KV Cache
        if kv_cache is not None:
            K_cached, V_cached = kv_cache
            K = torch.cat([K_cached, K], dim=2)
            V = torch.cat([V_cached, V], dim=2)

        new_cache = (K, V)

        # Attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Causal mask
        seq_len = K.shape[2]
        q_len = Q.shape[2]
        mask = torch.tril(torch.ones(q_len, seq_len, device=x.device))
        mask = mask[-q_len:, :]  # 只需要最後 q_len 行
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).contiguous().view(B, -1, C)
        return self.W_o(out), new_cache


# 比較有無 KV Cache 的速度
d_model = 64
attn = CausalAttentionWithCache(d_model, num_heads=4)
attn.eval()

# 無 cache: 每次輸入完整序列
print("\n--- 無 KV Cache ---")
start = time.time()
for seq_len in range(1, 51):
    x = torch.randn(1, seq_len, d_model)
    with torch.no_grad():
        out, _ = attn(x)
no_cache_time = time.time() - start

# 有 cache: 每次只輸入最新的 token
print("--- 有 KV Cache ---")
start = time.time()
cache = None
for seq_len in range(1, 51):
    x = torch.randn(1, 1, d_model)
    with torch.no_grad():
        out, cache = attn(x, kv_cache=cache)
cache_time = time.time() - start

print(f"無 Cache: {no_cache_time:.4f}s (生成 50 tokens)")
print(f"有 Cache: {cache_time:.4f}s (生成 50 tokens)")
print(f"加速比: {no_cache_time / cache_time:.2f}x")
print("→ 序列越長，加速越明顯")


# ============================================================================
# 5. Flash Attention
# ============================================================================
print("\n" + "=" * 60)
print("5. Flash Attention — 記憶體友好的 Attention")
print("=" * 60)

print("""
標準 Attention 的問題：
  scores = Q @ K^T    → (seq_len × seq_len) 的矩陣
  seq_len = 4096 → 4096² = 16M 個元素 → 很大！
  GPU 的 HBM (顯存) 太慢，SRAM (暫存器) 太小

Flash Attention (Dao et al., 2022):
  不把完整的 attention matrix 存到 HBM
  而是分塊在 SRAM 中計算
  → 記憶體用量從 O(N²) 降到 O(N)
  → 速度快 2-4x

  PyTorch 2.0+ 已內建:
  F.scaled_dot_product_attention(Q, K, V, is_causal=True)
  → 自動使用 Flash Attention（如果硬體支援）

Flash Attention 2 (2023): 更好的 GPU 利用率
Flash Attention 3 (2024): 支援更多硬體
""")


# ============================================================================
# 6. Speculative Decoding
# ============================================================================
print("=" * 60)
print("6. Speculative Decoding — 用小模型加速大模型")
print("=" * 60)

print("""
LLM 生成的瓶頸：
  自迴歸 = 一次只生成一個 token
  每個 token 都要跑一次完整的模型 → 太慢

Speculative Decoding 的想法：
  1. 用一個「小模型」快速生成 k 個候選 token
  2. 用「大模型」一次驗證這 k 個 token
  3. 如果都對了 → 跳過 k 步
  4. 如果第 j 個不對 → 只接受前 j-1 個

  為什麼有效？
  - 大多數 token 很「簡單」（小模型也能猜對）
  - 驗證是平行的（大模型一次看 k 個）
  - 數學上保證生成的分佈和大模型完全一樣！

  加速比: 通常 2-3x
  → 越「簡單」的文字（常見用語）加速越明顯

  變體：
  - Medusa: 大模型自己加多個 head 來猜
  - EAGLE: 用特徵來預測
  - Lookahead Decoding: 用 Jacobi 迭代
""")

# 簡單模擬 Speculative Decoding
print("\n--- 模擬 Speculative Decoding ---")

np.random.seed(42)
vocab = ['the', 'cat', 'sat', 'on', 'mat', 'a', 'dog', 'ran']

# 模擬大小模型的「信心」
def large_model_generate():
    """大模型每次需要 10ms"""
    time.sleep(0.01)
    return np.random.choice(vocab)

def small_model_generate():
    """小模型每次只需要 2ms"""
    time.sleep(0.002)
    return np.random.choice(vocab)

# 標準自迴歸
n_tokens = 20
start = time.time()
tokens_standard = []
for _ in range(n_tokens):
    tokens_standard.append(large_model_generate())
standard_time = time.time() - start

# Speculative Decoding (模擬)
start = time.time()
tokens_spec = []
k = 4  # draft 4 個 token
accepted_total = 0
drafted_total = 0

while len(tokens_spec) < n_tokens:
    # 小模型 draft k 個
    drafts = [small_model_generate() for _ in range(k)]
    drafted_total += k

    # 大模型驗證（模擬：70% 的機率接受）
    time.sleep(0.01)  # 大模型跑一次
    n_accept = 0
    for d in drafts:
        if np.random.random() < 0.7 and len(tokens_spec) + n_accept < n_tokens:
            n_accept += 1
        else:
            break
    accepted_total += n_accept

    # 接受前 n_accept 個
    tokens_spec.extend(drafts[:max(1, n_accept)])
    tokens_spec = tokens_spec[:n_tokens]

spec_time = time.time() - start

print(f"標準生成 ({n_tokens} tokens): {standard_time:.3f}s")
print(f"Speculative ({n_tokens} tokens): {spec_time:.3f}s")
print(f"加速比: {standard_time / spec_time:.2f}x")
print(f"接受率: {accepted_total}/{drafted_total} = {accepted_total/max(1,drafted_total):.0%}")


# ============================================================================
# 7. 其他加速技術
# ============================================================================
print("\n" + "=" * 60)
print("7. 其他重要的加速技術")
print("=" * 60)

print("""
  ┌─────────────────────────────────────────────────┐
  │ vLLM (2023):                                     │
  │   - PagedAttention: 高效管理 KV Cache 記憶體      │
  │   - Continuous Batching: 動態合併請求              │
  │   - 比 HF Transformers 快 10-24x                 │
  │   - LLM 推論的事實標準                            │
  │   用法: vllm serve meta-llama/Llama-2-7b-hf     │
  └─────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────┐
  │ TensorRT (NVIDIA):                               │
  │   - 針對 NVIDIA GPU 深度優化                      │
  │   - 算子融合 + INT8 量化 + Kernel 選擇           │
  │   - 可以比 PyTorch 快 5-10x                      │
  │   - TensorRT-LLM: 專門優化 LLM 推論             │
  └─────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────┐
  │ Continuous Batching:                              │
  │   傳統: 一批請求同時開始、同時結束                  │
  │   問題: 短的請求等長的 → 浪費 GPU                  │
  │   Continuous: 請求可以隨時加入和離開               │
  │   → GPU 利用率大幅提升                            │
  └─────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────┐
  │ 算子融合 (Operator Fusion):                       │
  │   Conv → BN → ReLU 是三個獨立的算子               │
  │   → 三次讀寫 GPU 記憶體                           │
  │   融合成一個算子                                   │
  │   → 只讀寫一次 → 快很多                           │
  └─────────────────────────────────────────────────┘
""")


# ============================================================================
# 小結
# ============================================================================
print("=" * 60)
print("小結")
print("=" * 60)
print("""
推論加速技術總覽：

  技術              加速比    難度    適用場景
  ──────────────────────────────────────────────
  torch.compile     1.3-2x   低     通用
  TorchScript       1.1-1.3x 低     部署
  ONNX Runtime      2-3x     中     跨平台部署
  TensorRT          5-10x    高     NVIDIA GPU
  INT8 量化         3-4x     低-中  邊緣裝置
  KV Cache          3-10x    中     Transformer
  Flash Attention   2-4x     低     長序列
  Speculative Dec.  2-3x     中     LLM 生成
  vLLM             10-24x    低     LLM 服務

  LLM 推論最佳實踐：
  1. 量化: INT4 (GPTQ/AWQ) 減少記憶體
  2. KV Cache: 必須開啟
  3. Flash Attention: PyTorch 2.0+ 自動使用
  4. vLLM/TGI: 用於生產環境
  5. Continuous Batching: 提高吞吐量

下一步：03_deployment.py — 模型部署
""")
