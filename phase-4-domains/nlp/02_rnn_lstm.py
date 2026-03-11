"""
=============================================================================
Phase 4-NLP-2: 序列模型 — RNN → LSTM → GRU
=============================================================================

文字是「序列」：字的順序很重要！
  "狗咬人" ≠ "人咬狗"

CNN 沒有順序概念，所以 NLP 需要專門處理序列的模型。

本檔涵蓋：
  1. RNN 原理 (NumPy 手刻)
  2. RNN 的梯度消失問題
  3. LSTM — 長期記憶的解法
  4. GRU — LSTM 的簡化版
  5. PyTorch 實作 + 文字分類
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ============================================================================
# 1. RNN — NumPy 手刻
# ============================================================================
print("=" * 60)
print("1. RNN — 有記憶的神經網路")
print("=" * 60)

print("""
RNN 的核心：每一步都維護一個「隱藏狀態」h，攜帶之前的資訊

  h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)
  y_t = W_hy @ h_t + b_y

  展開：
  x_0 → [RNN] → h_0 → [RNN] → h_1 → [RNN] → h_2 → y
              ↑ x_1          ↑ x_2

  h 就像是 RNN 的「記憶」
  每一步的 h 融合了「當前輸入 x_t」和「之前的記憶 h_{t-1}」
""")


class SimpleRNN_numpy:
    """NumPy 手刻 RNN"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        scale = 0.01
        self.Wxh = np.random.randn(hidden_dim, input_dim) * scale
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * scale
        self.Why = np.random.randn(output_dim, hidden_dim) * scale
        self.bh = np.zeros(hidden_dim)
        self.by = np.zeros(output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x_seq):
        """
        x_seq: (seq_len, input_dim)
        return: all hidden states, final output
        """
        seq_len = x_seq.shape[0]
        h = np.zeros(self.hidden_dim)
        hiddens = []

        for t in range(seq_len):
            h = np.tanh(self.Wxh @ x_seq[t] + self.Whh @ h + self.bh)
            hiddens.append(h.copy())

        y = self.Why @ h + self.by
        return np.array(hiddens), y


# 測試
np.random.seed(42)
rnn = SimpleRNN_numpy(input_dim=4, hidden_dim=8, output_dim=2)
x_seq = np.random.randn(5, 4)  # 5 步的序列，每步 4 維
hiddens, output = rnn.forward(x_seq)

print(f"輸入序列: {x_seq.shape}  (seq_len=5, input_dim=4)")
print(f"隱藏狀態: {hiddens.shape}  (seq_len=5, hidden_dim=8)")
print(f"最終輸出: {output.shape}   (output_dim=2)")
print(f"\n每步隱藏狀態的 norm（看信號傳播）:")
for t in range(5):
    print(f"  t={t}: ||h|| = {np.linalg.norm(hiddens[t]):.4f}")


# ============================================================================
# 2. RNN 的梯度消失/爆炸問題
# ============================================================================
print("\n" + "=" * 60)
print("2. RNN 的致命問題 — 梯度消失/爆炸")
print("=" * 60)

print("""
反向傳播時，梯度要從最後一步傳回第一步：

  ∂L/∂h_0 = ∂L/∂h_T × ∂h_T/∂h_{T-1} × ... × ∂h_1/∂h_0

  每一步都乘以 W_hh（的 Jacobian），連乘 T 次：
    如果 W_hh 的特徵值 > 1 → 梯度爆炸（指數增長）
    如果 W_hh 的特徵值 < 1 → 梯度消失（指數衰減）

  實際影響：
    序列長度 > 20-30 時，RNN 就記不住早期的資訊了
    → "The cat, which sat on the mat, ..." 到最後忘了主語是 cat
""")

# 視覺化梯度消失
print("模擬梯度隨步數的衰減：")
seq_lengths = [10, 30, 50, 100]
eigenvalues = [0.9, 0.5, 1.0, 1.1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 不同特徵值
for ev in eigenvalues:
    steps = np.arange(50)
    gradients = ev ** steps
    axes[0].plot(steps, gradients, label=f'λ={ev}')
axes[0].set_xlabel('Steps back in time')
axes[0].set_ylabel('Gradient magnitude')
axes[0].set_title('Gradient Flow: Vanishing & Exploding')
axes[0].legend()
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# RNN vs LSTM（概念性比較）
steps = np.arange(50)
rnn_grad = 0.9 ** steps
lstm_grad = np.ones(50) * 0.8 + np.random.normal(0, 0.05, 50)
lstm_grad = np.clip(lstm_grad, 0.3, 1.0)

axes[1].plot(steps, rnn_grad, label='RNN', color='red')
axes[1].plot(steps, lstm_grad, label='LSTM (理想)', color='blue')
axes[1].set_xlabel('Steps back in time')
axes[1].set_ylabel('Gradient magnitude')
axes[1].set_title('RNN vs LSTM Gradient Flow')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase-4-domains/nlp/plots/02_gradient_flow.png', dpi=100)
plt.close()
print("→ 圖表儲存至 nlp/plots/02_gradient_flow.png")


# ============================================================================
# 3. LSTM — 長短期記憶
# ============================================================================
print("\n" + "=" * 60)
print("3. LSTM — 解決梯度消失的經典方案")
print("=" * 60)

print("""
LSTM 的關鍵：引入「Cell State」和三個 Gate

  Cell State (c_t): 像一條傳送帶，資訊可以不被修改地流過

  三個 Gate（值在 0~1 之間，由 sigmoid 產生）：

  1. Forget Gate (f_t):  決定要「忘記」多少舊資訊
     f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)

  2. Input Gate (i_t):   決定要「記住」多少新資訊
     i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)
     c̃_t = tanh(W_c @ [h_{t-1}, x_t] + b_c)   ← 候選新記憶

  3. Output Gate (o_t):  決定要「輸出」多少
     o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)

  更新：
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t     ← cell state 更新
    h_t = o_t ⊙ tanh(c_t)                  ← 隱藏狀態

  關鍵：c_t 的更新是「加法」而非「乘法」
    → 梯度可以直接沿著 cell state 流動
    → 不會消失！
""")


class LSTMCell_numpy:
    """NumPy 手刻 LSTM Cell"""
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        scale = 0.1
        # 把 4 個 gate 的權重合在一起（更高效）
        # 順序: i, f, g(candidate), o
        total_dim = input_dim + hidden_dim
        self.W = np.random.randn(4 * hidden_dim, total_dim) * scale
        self.b = np.zeros(4 * hidden_dim)
        # Forget gate bias 設為 1（讓初始時記住更多）
        self.b[hidden_dim:2*hidden_dim] = 1.0

    def forward(self, x_seq):
        seq_len = x_seq.shape[0]
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        hiddens = []
        cells = []

        for t in range(seq_len):
            combined = np.concatenate([h, x_seq[t]])
            gates = self.W @ combined + self.b

            hd = self.hidden_dim
            i = self._sigmoid(gates[:hd])          # input gate
            f = self._sigmoid(gates[hd:2*hd])      # forget gate
            g = np.tanh(gates[2*hd:3*hd])           # candidate
            o = self._sigmoid(gates[3*hd:])          # output gate

            c = f * c + i * g                        # cell state
            h = o * np.tanh(c)                       # hidden state

            hiddens.append(h.copy())
            cells.append(c.copy())

        return np.array(hiddens), np.array(cells)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# 測試
np.random.seed(42)
lstm = LSTMCell_numpy(input_dim=4, hidden_dim=8)
x_seq = np.random.randn(10, 4)
hiddens, cells = lstm.forward(x_seq)

print(f"輸入序列: {x_seq.shape}")
print(f"隱藏狀態: {hiddens.shape}")
print(f"Cell 狀態: {cells.shape}")

print(f"\n隱藏狀態 norm（LSTM 比 RNN 穩定）:")
for t in range(10):
    print(f"  t={t}: ||h|| = {np.linalg.norm(hiddens[t]):.4f}, "
          f"||c|| = {np.linalg.norm(cells[t]):.4f}")


# ============================================================================
# 4. GRU — LSTM 的簡化版
# ============================================================================
print("\n" + "=" * 60)
print("4. GRU — 更簡單、效率更好")
print("=" * 60)

print("""
GRU 把 LSTM 的 3 個 gate 簡化成 2 個：

  Reset Gate (r_t):  決定如何結合新輸入和舊記憶
    r_t = σ(W_r @ [h_{t-1}, x_t])

  Update Gate (z_t):  決定保留多少舊記憶 vs 接受多少新資訊
    z_t = σ(W_z @ [h_{t-1}, x_t])

  候選隱藏狀態：
    h̃_t = tanh(W @ [r_t ⊙ h_{t-1}, x_t])

  更新：
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

  GRU vs LSTM：
    - GRU 沒有 cell state（少一個狀態）
    - GRU 參數更少、訓練更快
    - 效果通常差不多
    - 短序列用 GRU，長序列用 LSTM
""")


# ============================================================================
# 5. PyTorch 實作 — 文字情感分類
# ============================================================================
print("\n" + "=" * 60)
print("5. PyTorch RNN/LSTM/GRU — 情感分類")
print("=" * 60)

# 建立簡單的情感分析資料集
sentences = [
    ("the movie is great", 1),
    ("the film is wonderful", 1),
    ("i love this movie", 1),
    ("the movie is good and fun", 1),
    ("great story and acting", 1),
    ("really good film", 1),
    ("i like the movie a lot", 1),
    ("the movie is bad", 0),
    ("terrible film", 0),
    ("i hate this movie", 0),
    ("worst movie ever", 0),
    ("bad acting and story", 0),
    ("really bad and boring", 0),
    ("i do not like this film", 0),
]

# 建立詞彙表
all_words_sent = set()
for sent, _ in sentences:
    all_words_sent.update(sent.split())
sent_vocab = ['<PAD>'] + sorted(all_words_sent)
sent_w2i = {w: i for i, w in enumerate(sent_vocab)}
sent_vocab_size = len(sent_vocab)
print(f"詞彙量: {sent_vocab_size}")

# 轉成 tensor
max_len = max(len(s.split()) for s, _ in sentences)
X_data = torch.zeros(len(sentences), max_len, dtype=torch.long)
y_data = torch.zeros(len(sentences), dtype=torch.long)

for i, (sent, label) in enumerate(sentences):
    words = sent.split()
    for j, w in enumerate(words):
        X_data[i, j] = sent_w2i[w]
    y_data[i] = label

print(f"輸入 shape: {X_data.shape}  (samples, max_len)")
print(f"標籤 shape: {y_data.shape}")


class SentimentRNN(nn.Module):
    """用 RNN/LSTM/GRU 做情感分類"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, rnn_type='LSTM'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        self.rnn_type = rnn_type
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)          # (batch, seq_len, embed_dim)

        if self.rnn_type == 'LSTM':
            output, (h_n, c_n) = self.rnn(embeds)
        else:
            output, h_n = self.rnn(embeds)

        # 用最後一個隱藏狀態做分類
        last_hidden = h_n.squeeze(0)        # (batch, hidden_dim)
        return self.fc(last_hidden)


# 訓練和比較
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results = {}
for rnn_type in ['RNN', 'LSTM', 'GRU']:
    torch.manual_seed(42)
    model = SentimentRNN(sent_vocab_size, embed_dim=16, hidden_dim=32,
                          num_classes=2, rnn_type=rnn_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(100):
        model.train()
        logits = model(X_data.to(device))
        loss = criterion(logits, y_data.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        preds = model(X_data.to(device)).argmax(1)
        acc = (preds == y_data.to(device)).float().mean().item()

    params = sum(p.numel() for p in model.parameters())
    print(f"  {rnn_type:4s}: Loss={loss.item():.4f}, Acc={acc:.2%}, Params={params:,}")
    results[rnn_type] = losses

# 畫 loss 曲線
fig, ax = plt.subplots(figsize=(8, 4))
for name, losses in results.items():
    ax.plot(losses, label=name)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('RNN vs LSTM vs GRU Training Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phase-4-domains/nlp/plots/02_rnn_comparison.png', dpi=100)
plt.close()
print("\n→ 圖表儲存至 nlp/plots/02_rnn_comparison.png")


# ============================================================================
# 6. 雙向 RNN 和多層 RNN
# ============================================================================
print("\n" + "=" * 60)
print("6. 進階：Bidirectional & Multi-layer")
print("=" * 60)

print("""
Bidirectional RNN:
  正向: h_0 → h_1 → h_2 → h_3
  反向: h_0 ← h_1 ← h_2 ← h_3

  合併: [h_forward; h_backward]

  好處：每個位置都能看到「前面」和「後面」的資訊

多層 RNN:
  Layer 2:  h²_0 → h²_1 → h²_2 → h²_3
                ↑        ↑        ↑
  Layer 1:  h¹_0 → h¹_1 → h¹_2 → h¹_3
                ↑        ↑        ↑
  Input:    x_0      x_1      x_2      x_3
""")

# PyTorch 支援
bilstm = nn.LSTM(input_size=16, hidden_size=32,
                  num_layers=2, batch_first=True,
                  bidirectional=True, dropout=0.1)

x = torch.randn(4, 10, 16)  # (batch=4, seq_len=10, input=16)
output, (h_n, c_n) = bilstm(x)

print(f"Bidirectional 2-layer LSTM:")
print(f"  輸入: {x.shape}")
print(f"  輸出: {output.shape}")       # (4, 10, 64) → hidden*2
print(f"  h_n:  {h_n.shape}")          # (4, 4, 32) → layers*2
print(f"  → 輸出 hidden_size × 2（雙向）")
print(f"  → h_n 有 num_layers × 2 個")


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結")
print("=" * 60)
print("""
序列模型對照表：

  模型    Gate 數   狀態     長序列    速度    使用場景
  ──────────────────────────────────────────────────
  RNN     0        h       差       快     短序列、簡單任務
  LSTM    3 (i,f,o) h + c   好       慢     長序列、複雜任務
  GRU     2 (r,z)   h       不錯     中     平衡選擇

  PyTorch API：
    nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    nn.LSTM(...)   → 回傳 (output, (h_n, c_n))
    nn.GRU(...)    → 回傳 (output, h_n)

  實務重點：
  1. 幾乎不用 vanilla RNN（梯度消失太嚴重）
  2. LSTM 和 GRU 擇一即可（效果差不多）
  3. Bidirectional 通常更好（如果不需要即時生成）
  4. 2-3 層就夠了（太深反而不好）
  5. 現在大多數 NLP 已經用 Transformer 取代 RNN

下一步：03_attention_seq2seq.py — 注意力機制
""")
