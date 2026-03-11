"""
=============================================================================
Phase 4-NLP-3: 注意力機制 + Seq2Seq — 通往 Transformer 的橋樑
=============================================================================

RNN/LSTM 的瓶頸：
  所有資訊都壓縮到一個固定大小的向量 h
  句子越長，壓縮損失越大

Attention 的想法：
  解碼時不只看最後的 h，而是看 encoder 的「每一步」
  而且決定「現在該看哪裡」

這是 Transformer 最核心的概念！

本檔涵蓋：
  1. Seq2Seq 模型（無 Attention）
  2. Attention 機制的原理（NumPy 手刻）
  3. Seq2Seq + Attention
  4. Self-Attention — Transformer 的核心
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ============================================================================
# 1. Seq2Seq — Encoder-Decoder 架構
# ============================================================================
print("=" * 60)
print("1. Seq2Seq — 序列到序列")
print("=" * 60)

print("""
Seq2Seq 用在輸入和輸出都是序列的任務：
  - 機器翻譯: "I love cats" → "我愛貓"
  - 文字摘要: 長文 → 短文
  - 對話系統: 問題 → 回答

結構：
  Encoder: 把輸入序列壓縮成一個向量（context vector）
  Decoder: 用 context vector 生成輸出序列

  Encoder:
  "I" → [LSTM] → h1 → [LSTM] → h2 → [LSTM] → h3 = context
                                                    ↓
  Decoder:                                    [LSTM] → "我"
                                               ↓
                                             [LSTM] → "愛"
                                               ↓
                                             [LSTM] → "貓"

  問題：所有資訊都壓縮在 h3（一個向量），句子長了會丟失資訊
""")


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embeds = self.embedding(x)
        outputs, (h_n, c_n) = self.rnn(embeds)
        return outputs, h_n, c_n


class DecoderNoAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h, c):
        embeds = self.embedding(x)
        output, (h, c) = self.rnn(embeds, (h, c))
        logits = self.fc(output)
        return logits, h, c


# 測試
enc = Encoder(vocab_size=100, embed_dim=16, hidden_dim=32)
dec = DecoderNoAttention(vocab_size=50, embed_dim=16, hidden_dim=32)

src = torch.randint(0, 100, (2, 8))   # batch=2, 源語言長度=8
tgt = torch.randint(0, 50, (2, 6))    # batch=2, 目標語言長度=6

enc_outputs, h, c = enc(src)
dec_output, _, _ = dec(tgt, h, c)

print(f"Encoder 輸入: {src.shape}")
print(f"Encoder 所有 hidden: {enc_outputs.shape}")
print(f"Context vector (h): {h.shape}")
print(f"Decoder 輸出: {dec_output.shape}")


# ============================================================================
# 2. Attention 機制 — NumPy 手刻
# ============================================================================
print("\n" + "=" * 60)
print("2. Attention — 讓模型學會「看哪裡」")
print("=" * 60)

print("""
Attention 的直覺：
  翻譯 "I love cats" 的 "貓" 時
  應該主要關注 "cats"，而不是 "I" 或 "love"

三步驟：
  1. Score:  計算 decoder 的 h 和 encoder 每一步的相似度
  2. Weight: 用 softmax 轉成權重（加總=1）
  3. Context: 用權重加權求和 encoder 的 outputs

  Score 的計算方式：
  - Dot Product:  score = h_dec · h_enc
  - Scaled Dot:   score = (h_dec · h_enc) / √d
  - Additive:     score = v^T tanh(W_1 h_dec + W_2 h_enc)
""")


def attention_numpy(query, keys, values):
    """
    NumPy 手刻 Scaled Dot-Product Attention

    query:  (d,)         — decoder 的當前狀態
    keys:   (seq_len, d) — encoder 的所有 hidden states
    values: (seq_len, d) — encoder 的所有 hidden states（通常和 keys 一樣）

    return: context (d,), weights (seq_len,)
    """
    d = query.shape[0]

    # Step 1: 計算 attention scores
    scores = keys @ query / np.sqrt(d)      # (seq_len,)

    # Step 2: Softmax → 權重
    exp_scores = np.exp(scores - scores.max())  # 數值穩定
    weights = exp_scores / exp_scores.sum()     # (seq_len,)

    # Step 3: 加權求和
    context = weights @ values                   # (d,)

    return context, weights


# 測試
np.random.seed(42)
seq_len = 6
d = 8

# 模擬 encoder 輸出
encoder_outputs = np.random.randn(seq_len, d)   # 6 個位置的 hidden state
# 模擬 decoder 的當前狀態
decoder_state = np.random.randn(d)

context, weights = attention_numpy(decoder_state, encoder_outputs, encoder_outputs)

print(f"Encoder outputs: {encoder_outputs.shape}")
print(f"Decoder state: {decoder_state.shape}")
print(f"Attention weights: {weights}")
print(f"  → 權重加總: {weights.sum():.4f}")
print(f"Context vector: {context.shape}")
print(f"  → 最被關注的位置: {weights.argmax()}")

# 視覺化
fig, ax = plt.subplots(figsize=(8, 3))
words = ['I', 'love', 'cats', 'very', 'much', '<EOS>']
ax.bar(words, weights, color='steelblue')
ax.set_title('Attention Weights (where to look)')
ax.set_ylabel('Weight')
plt.tight_layout()
plt.savefig('phase-4-domains/nlp/plots/03_attention_weights.png', dpi=100)
plt.close()
print("→ 圖表儲存至 nlp/plots/03_attention_weights.png")


# ============================================================================
# 3. PyTorch Attention + Seq2Seq
# ============================================================================
print("\n" + "=" * 60)
print("3. PyTorch Attention 實作")
print("=" * 60)


class Attention(nn.Module):
    """Luong-style Dot Product Attention"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: (batch, hidden_dim)
        encoder_outputs: (batch, src_len, hidden_dim)
        """
        # (batch, src_len)
        scores = torch.bmm(
            encoder_outputs,
            decoder_hidden.unsqueeze(2)
        ).squeeze(2)

        weights = F.softmax(scores, dim=1)   # (batch, src_len)

        # 加權求和
        context = torch.bmm(
            weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)  # (batch, hidden_dim)

        return context, weights


class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, h, c, encoder_outputs):
        """一步 decode"""
        embeds = self.embedding(x)           # (batch, 1, embed_dim)

        # Attention
        attn_context, attn_weights = self.attention(
            h.squeeze(0), encoder_outputs
        )

        # 把 attention context 和 embedding 拼接
        rnn_input = torch.cat([embeds, attn_context.unsqueeze(1)], dim=2)
        output, (h, c) = self.rnn(rnn_input, (h, c))

        # 輸出
        combined = torch.cat([output.squeeze(1), attn_context], dim=1)
        logits = self.fc(combined)
        return logits, h, c, attn_weights


# 測試
dec_attn = DecoderWithAttention(vocab_size=50, embed_dim=16, hidden_dim=32)
tgt_step = torch.randint(0, 50, (2, 1))   # 一步的輸入

logits, h_new, c_new, attn_w = dec_attn(tgt_step, h, c, enc_outputs)
print(f"Decoder + Attention 輸出:")
print(f"  logits: {logits.shape}")
print(f"  attention weights: {attn_w.shape}")
print(f"  → 每一步 decode 都會產生 attention weights")
print(f"  → 可以視覺化模型在「看」哪裡")


# ============================================================================
# 4. Self-Attention — Transformer 的核心
# ============================================================================
print("\n" + "=" * 60)
print("4. Self-Attention — 通往 Transformer")
print("=" * 60)

print("""
Seq2Seq Attention: decoder 看 encoder（cross-attention）
Self-Attention: 句子自己看自己！

  "The cat sat on the mat because it was tired"
  "it" 指的是 "cat" → self-attention 可以學到這個！

三個角色（來自同一個輸入）：
  Query (Q): 「我想找什麼？」
  Key (K):   「我有什麼？」
  Value (V): 「我的內容是什麼？」

  Attention(Q, K, V) = softmax(QK^T / √d_k) × V

  步驟：
  1. 輸入 X → 線性投影得到 Q, K, V
  2. Q × K^T → attention scores（誰和誰相關）
  3. softmax → 權重
  4. 權重 × V → 輸出
""")


def self_attention_numpy(X, Wq, Wk, Wv):
    """
    NumPy 手刻 Self-Attention

    X:  (seq_len, d_model)
    Wq, Wk, Wv: (d_model, d_k)
    """
    Q = X @ Wq    # (seq_len, d_k)
    K = X @ Wk
    V = X @ Wv

    d_k = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)   # (seq_len, seq_len)

    # Softmax（每一行）
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    output = weights @ V               # (seq_len, d_k)
    return output, weights


# 測試
np.random.seed(42)
seq_len = 5
d_model = 8
d_k = 4

X = np.random.randn(seq_len, d_model)
Wq = np.random.randn(d_model, d_k) * 0.1
Wk = np.random.randn(d_model, d_k) * 0.1
Wv = np.random.randn(d_model, d_k) * 0.1

output, weights = self_attention_numpy(X, Wq, Wk, Wv)

print(f"輸入 X: {X.shape}")
print(f"Q, K, V: ({seq_len}, {d_k})")
print(f"Attention weights: {weights.shape}")
print(f"輸出: {output.shape}")

# 視覺化 attention matrix
words = ['The', 'cat', 'is', 'on', 'mat']
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(weights, cmap='Blues')
ax.set_xticks(range(len(words)))
ax.set_yticks(range(len(words)))
ax.set_xticklabels(words)
ax.set_yticklabels(words)
ax.set_xlabel('Key (being attended to)')
ax.set_ylabel('Query (doing the attending)')
ax.set_title('Self-Attention Weights')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('phase-4-domains/nlp/plots/03_self_attention.png', dpi=100)
plt.close()
print("→ 圖表儲存至 nlp/plots/03_self_attention.png")


# ============================================================================
# 5. PyTorch Self-Attention
# ============================================================================
print("\n" + "=" * 60)
print("5. PyTorch Self-Attention")
print("=" * 60)


class SelfAttention(nn.Module):
    """Single-head Self-Attention"""
    def __init__(self, d_model, d_k):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_k, bias=False)
        self.Wk = nn.Linear(d_model, d_k, bias=False)
        self.Wv = nn.Linear(d_model, d_k, bias=False)
        self.d_k = d_k

    def forward(self, x, mask=None):
        """x: (batch, seq_len, d_model)"""
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        output = torch.bmm(weights, V)
        return output, weights


# 測試
attn = SelfAttention(d_model=16, d_k=8)
x = torch.randn(2, 6, 16)  # batch=2, seq=6, d=16
out, w = attn(x)
print(f"輸入: {x.shape}")
print(f"輸出: {out.shape}")
print(f"權重: {w.shape}")

# Causal Mask（用於 decoder，不能看到未來）
print("\n--- Causal Mask (防止看到未來) ---")
seq_len = 5
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
print(f"Causal Mask:\n{causal_mask.squeeze()}")
print("→ 1 = 可以看, 0 = 看不到")
print("→ 每個位置只能看到自己和之前的位置")

x = torch.randn(1, seq_len, 16)
out, w = attn(x, mask=causal_mask)
print(f"\n加了 causal mask 的 attention weights:\n{w.squeeze().detach()}")
print("→ 上三角都是 0（看不到未來）")


# ============================================================================
# 6. Multi-Head Attention（預告）
# ============================================================================
print("\n" + "=" * 60)
print("6. Multi-Head Attention（預告）")
print("=" * 60)

print("""
一個 Attention head 只能學一種「注意力模式」
Multi-Head: 用多個 head，每個學不同的模式

  Head 1: 可能學到「主語-動詞」的關係
  Head 2: 可能學到「形容詞-名詞」的關係
  Head 3: 可能學到「代詞-指代」的關係

  MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_o
  head_i = Attention(Q × W_q^i, K × W_k^i, V × W_v^i)

PyTorch 一行搞定：
  nn.MultiheadAttention(embed_dim=512, num_heads=8)
""")

# PyTorch 的 MultiheadAttention
mha = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
x = torch.randn(2, 6, 16)
output, weights = mha(x, x, x)   # self-attention: Q=K=V=x
print(f"MultiheadAttention:")
print(f"  輸入: {x.shape}")
print(f"  輸出: {output.shape}")
print(f"  權重: {weights.shape}")


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結")
print("=" * 60)
print("""
NLP 模型的演進：

  模型           年份    處理序列方式         長距離依賴
  ──────────────────────────────────────────────────────
  RNN/LSTM      ~2015   逐步遞迴              差/中
  Seq2Seq       ~2016   Encoder-Decoder        中
  + Attention   ~2017   加權看 encoder 每步    好
  Transformer   2017    純 Attention（無 RNN）  非常好

  關鍵演進：
  1. Seq2Seq: encoder 壓縮成一個向量（瓶頸）
  2. + Attention: 每步都看 encoder 所有位置
  3. Self-Attention: 句子自己看自己
  4. Transformer: 完全用 self-attention，拋棄 RNN

  Attention 的核心公式：
    Attention(Q, K, V) = softmax(QK^T / √d_k) × V

  這就是 Transformer 的全部基礎！

Phase 4 NLP 完成！
下一步：Phase 5 — Transformer 架構
""")
