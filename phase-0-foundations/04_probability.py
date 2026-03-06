"""
=============================================================================
Phase 0-4: 機率與統計 — 理解模型的輸出和損失函數
=============================================================================

為什麼學機率？
─────────────
模型的輸出通常是「機率」：
  - 這張圖片是貓的機率 = 0.85，是狗的機率 = 0.15
  - 下一個字是「好」的機率 = 0.3，是「的」的機率 = 0.25 ...

損失函數（衡量模型好不好）也建立在機率之上：
  - Cross-Entropy Loss 就是來自資訊理論和最大似然估計

本檔涵蓋：
  1. 機率基礎
  2. 常見分佈
  3. 條件機率與貝氏定理
  4. 期望值與變異數
  5. 最大似然估計 (MLE)
  6. Cross-Entropy：最常用的分類損失函數
"""

import numpy as np

# ============================================================================
# 1. 機率基礎
# ============================================================================
print("=" * 60)
print("1. 機率基礎")
print("=" * 60)

# 機率 = 事件發生的可能性，範圍 [0, 1]
# P(A) = 0 → 不可能發生
# P(A) = 1 → 一定發生
# 所有可能事件的機率總和 = 1

# 用程式模擬：擲骰子 10000 次
np.random.seed(42)
rolls = np.random.randint(1, 7, size=10000)   # 1~6 的隨機整數

print("擲骰子 10000 次的結果：")
for i in range(1, 7):
    count = np.sum(rolls == i)
    print(f"  {i}: {count} 次 ({count/len(rolls):.4f}) ← 理論值 {1/6:.4f}")

# 大數法則：試驗次數越多，頻率越接近真實機率


# ============================================================================
# 2. 常見分佈
# ============================================================================
print("\n" + "=" * 60)
print("2. 常見機率分佈")
print("=" * 60)

# --- 均勻分佈 (Uniform Distribution) ---
uniform = np.random.uniform(0, 1, size=10000)
print(f"均勻分佈 U(0,1): 平均={uniform.mean():.4f}, 標準差={uniform.std():.4f}")

# --- 常態分佈 / 高斯分佈 (Normal/Gaussian Distribution) ---
# 最重要的分佈！自然界很多現象服從常態分佈
# N(μ, σ²): μ=平均值, σ=標準差
normal = np.random.normal(loc=0, scale=1, size=10000)   # 標準常態分佈
print(f"常態分佈 N(0,1): 平均={normal.mean():.4f}, 標準差={normal.std():.4f}")

# 68-95-99.7 法則
within_1std = np.sum(np.abs(normal) < 1) / len(normal)
within_2std = np.sum(np.abs(normal) < 2) / len(normal)
within_3std = np.sum(np.abs(normal) < 3) / len(normal)
print(f"  1σ 內: {within_1std:.1%} (理論: 68.3%)")
print(f"  2σ 內: {within_2std:.1%} (理論: 95.4%)")
print(f"  3σ 內: {within_3std:.1%} (理論: 99.7%)")

# 在 ML 中：
#   - 權重初始化常用常態分佈：W ~ N(0, 0.01)
#   - Batch Normalization 把每層輸出調整成接近 N(0, 1)

# --- 伯努利分佈 (Bernoulli Distribution) ---
# 只有兩個結果：成功 (1) 或失敗 (0)
# 二元分類的基礎
p = 0.7   # 成功機率
bernoulli = np.random.binomial(1, p, size=10000)
print(f"\n伯努利分佈 (p={p}): 成功率={bernoulli.mean():.4f}")


# ============================================================================
# 3. 條件機率與貝氏定理
# ============================================================================
print("\n" + "=" * 60)
print("3. 條件機率與貝氏定理")
print("=" * 60)

# 條件機率 P(A|B) = 在 B 發生的條件下，A 發生的機率
# 貝氏定理：P(A|B) = P(B|A) × P(A) / P(B)

# 經典例子：垃圾郵件過濾
# P(垃圾|含「免費」) = P(含「免費」|垃圾) × P(垃圾) / P(含「免費」)

P_spam = 0.3                          # 30% 的信是垃圾郵件
P_free_given_spam = 0.8               # 垃圾郵件中 80% 含「免費」
P_free_given_not_spam = 0.1           # 正常郵件中 10% 含「免費」

# 全機率公式算 P(含「免費」)
P_free = P_free_given_spam * P_spam + P_free_given_not_spam * (1 - P_spam)

# 貝氏定理
P_spam_given_free = P_free_given_spam * P_spam / P_free

print(f"先驗機率 P(垃圾) = {P_spam}")
print(f"P(含「免費」| 垃圾) = {P_free_given_spam}")
print(f"P(含「免費」| 正常) = {P_free_given_not_spam}")
print(f"P(含「免費」) = {P_free:.4f}")
print(f"")
print(f"貝氏定理 →")
print(f"P(垃圾 | 含「免費」) = {P_spam_given_free:.4f}")
print(f"→ 含有「免費」的信，有 {P_spam_given_free:.1%} 機率是垃圾郵件")

# 在 ML 中：
#   - Naive Bayes 分類器直接用貝氏定理
#   - 貝氏思維：用新的證據（data）更新我們的信念（model）


# ============================================================================
# 4. 期望值與變異數
# ============================================================================
print("\n" + "=" * 60)
print("4. 期望值 (Expected Value) 與變異數 (Variance)")
print("=" * 60)

# 期望值 E[X] = 所有可能值 × 其機率 的加總
# 直覺：「長期平均下來的結果」

# 變異數 Var[X] = E[(X - E[X])²]
# 直覺：「資料分散的程度」

# 標準差 σ = √Var[X]

# 例：兩個投資方案
np.random.seed(42)
investment_A = np.random.normal(loc=10, scale=2, size=10000)    # 穩定：平均 10%，波動小
investment_B = np.random.normal(loc=10, scale=15, size=10000)   # 高風險：平均 10%，波動大

print(f"投資 A: 平均報酬={investment_A.mean():.2f}%, 標準差={investment_A.std():.2f}%")
print(f"投資 B: 平均報酬={investment_B.mean():.2f}%, 標準差={investment_B.std():.2f}%")
print(f"→ 平均一樣，但 B 的風險（變異數）大很多")

# 在 ML 中：
#   - Batch Normalization：讓每層的期望值 ≈ 0，變異數 ≈ 1
#   - 權重初始化：控制變異數防止梯度爆炸/消失
#   - Loss 的變異數太大 → 訓練不穩定


# ============================================================================
# 5. 最大似然估計 (MLE, Maximum Likelihood Estimation)
# ============================================================================
print("\n" + "=" * 60)
print("5. 最大似然估計 (MLE)")
print("=" * 60)

# MLE 的問題：給定觀察到的資料，哪組參數最可能產生這些資料？
# 似然函數 L(θ|data) = P(data|θ)

# 例：丟銅板 10 次，出現 7 次正面，p（正面機率）最可能是多少？

observations = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 1])  # 1=正面, 0=反面
n_heads = observations.sum()
n_total = len(observations)

# 嘗試不同的 p 值，算 Likelihood
p_values = np.linspace(0.01, 0.99, 99)
likelihoods = []

for p in p_values:
    # 每次獨立：P(data|p) = p^(正面次數) × (1-p)^(反面次數)
    likelihood = p ** n_heads * (1 - p) ** (n_total - n_heads)
    likelihoods.append(likelihood)

likelihoods = np.array(likelihoods)
best_p = p_values[likelihoods.argmax()]

print(f"觀察：{n_total} 次中有 {n_heads} 次正面")
print(f"MLE 估計的 p = {best_p:.2f}")
print(f"直覺解：p = {n_heads}/{n_total} = {n_heads/n_total:.2f}")

# 通常用 Log-Likelihood（取 log），因為連乘容易數值下溢
log_likelihoods = n_heads * np.log(p_values) + (n_total - n_heads) * np.log(1 - p_values)
best_p_log = p_values[log_likelihoods.argmax()]
print(f"Log-Likelihood 版本的 MLE: {best_p_log:.2f}")

# 關鍵連結：
#   最小化 Cross-Entropy Loss = 最大化 Log-Likelihood
#   訓練神經網路做分類 = 在做 MLE


# ============================================================================
# 6. Cross-Entropy — 最常用的分類損失函數
# ============================================================================
print("\n" + "=" * 60)
print("6. Cross-Entropy Loss — 連接機率和訓練")
print("=" * 60)

# --- 資訊量 (Information) ---
# I(x) = -log₂(P(x))
# 機率越小的事件，資訊量越大
# P = 1.0 (一定會發生) → I = 0 (沒有新資訊)
# P = 0.01 (幾乎不會) → I 很大 (發生了很驚訝)

print("資訊量 I = -log₂(P):")
for p in [1.0, 0.5, 0.1, 0.01]:
    info = -np.log2(p) if p > 0 else float('inf')
    print(f"  P = {p:.2f} → I = {info:.2f} bits")

# --- Entropy (熵) ---
# H(P) = -Σ P(x) × log(P(x))
# 衡量一個分佈的「不確定性」
# 越均勻（越不確定）→ 熵越大
# 越集中（越確定）→ 熵越小

print("\n熵 (Entropy):")
distributions = {
    "確定 [1, 0, 0]":    np.array([1.0, 0.0, 0.0]),
    "偏向 [0.8, 0.1, 0.1]": np.array([0.8, 0.1, 0.1]),
    "均勻 [0.33, 0.33, 0.33]": np.array([1/3, 1/3, 1/3]),
}

for name, p in distributions.items():
    p_safe = p[p > 0]   # 避免 log(0)
    entropy = -np.sum(p_safe * np.log(p_safe))
    print(f"  {name:30s} → H = {entropy:.4f}")

# --- Cross-Entropy Loss ---
# CE(true, pred) = -Σ true(x) × log(pred(x))
# true = 真實標籤的 one-hot 分佈
# pred = 模型預測的機率分佈
# 預測越接近真實 → CE 越小

print("\nCross-Entropy Loss 示範:")
print("假設 3 個類別，真實答案是類別 0")

true_label = np.array([1, 0, 0])    # one-hot: 類別 0

# 不同的模型預測
predictions = {
    "完美預測 [0.99, 0.005, 0.005]": np.array([0.99, 0.005, 0.005]),
    "不錯預測 [0.7, 0.2, 0.1]":     np.array([0.7, 0.2, 0.1]),
    "普通預測 [0.4, 0.3, 0.3]":     np.array([0.4, 0.3, 0.3]),
    "糟糕預測 [0.1, 0.5, 0.4]":     np.array([0.1, 0.5, 0.4]),
    "反向預測 [0.01, 0.01, 0.98]":  np.array([0.01, 0.01, 0.98]),
}

for name, pred in predictions.items():
    ce_loss = -np.sum(true_label * np.log(pred))
    print(f"  {name:35s} → CE = {ce_loss:.4f}")

print("""
觀察：
  - 預測越正確（給正確類別的機率越高）→ Loss 越小
  - 預測越錯誤 → Loss 越大
  - 完全預測錯誤 → Loss 趨近無窮大

  這就是為什麼 Cross-Entropy 比 MSE 更適合分類：
  它對「自信但錯誤」的預測懲罰非常大
""")

# --- 在實際訓練中 ---
print("--- 簡化版：在程式中通常這樣算 ---")

# 不用 one-hot，直接用類別 index
true_class = 0   # 正確答案是類別 0
pred_probs = np.array([0.7, 0.2, 0.1])

# CE = -log(模型給正確類別的機率)
ce = -np.log(pred_probs[true_class])
print(f"真實類別: {true_class}")
print(f"模型預測: {pred_probs}")
print(f"CE Loss = -log({pred_probs[true_class]}) = {ce:.4f}")

# 一批資料的平均 CE
batch_true = np.array([0, 2, 1])   # 3 個樣本的真實類別
batch_pred = np.array([
    [0.7, 0.2, 0.1],   # 樣本 0：預測類別 0 (正確)
    [0.1, 0.3, 0.6],   # 樣本 1：預測類別 2 (正確)
    [0.2, 0.5, 0.3],   # 樣本 2：預測類別 1 (正確)
])

losses = [-np.log(batch_pred[i, batch_true[i]]) for i in range(len(batch_true))]
avg_loss = np.mean(losses)
print(f"\n一批 3 個樣本的 Loss: {losses}")
print(f"平均 Loss: {avg_loss:.4f}")


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結")
print("=" * 60)
print("""
機率統計在 ML 中的角色：

  概念               →  ML 對應
  ──────────────────────────────────────────
  機率分佈           →  模型的輸出（Softmax）
  常態分佈           →  權重初始化、Normalization
  條件機率           →  P(類別|輸入特徵)
  貝氏定理           →  用資料更新模型的信念
  期望值 / 變異數    →  Loss 的穩定性、BN
  MLE               →  訓練分類模型的理論基礎
  Cross-Entropy      →  最常用的分類損失函數

最重要的連結：
  訓練分類模型
  = 最小化 Cross-Entropy Loss
  = 最大化 Log-Likelihood
  = 讓模型給正確答案的機率越大越好

下一步：05_matplotlib_basics.py — 把這些概念視覺化
""")
