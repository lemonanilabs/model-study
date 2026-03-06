"""
=============================================================================
Phase 1-7: 模型評估 — 怎麼知道模型好不好
=============================================================================

「模型在訓練集上 99% 準確率」不代表模型好。
正確的評估方法是 ML 工程中最重要的技能之一。

本檔涵蓋：
  Part A: Train/Val/Test Split 的意義
  Part B: 分類指標：Accuracy / Precision / Recall / F1
  Part C: 混淆矩陣 (Confusion Matrix)
  Part D: 交叉驗證 (Cross Validation)
  Part E: Bias-Variance Tradeoff
  Part F: 完整的模型比較流程
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Part A: 資料切分
# ============================================================================
print("=" * 60)
print("Part A: 為什麼要切分資料")
print("=" * 60)

print("""
三種資料集：
  訓練集 (Train): 模型學習用         → 考試的「課本」
  驗證集 (Val):   調超參數用         → 考試的「模擬考」
  測試集 (Test):  最終評估用，只看一次 → 考試的「正式考」

為什麼不能用訓練集評估？
  因為模型可能只是「背答案」（過擬合），
  就像學生背了整本題庫，換一題就不會了。

常見比例：Train 70% / Val 15% / Test 15%
  或簡化為：Train 80% / Test 20%
""")

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# 用乳癌資料集（二元分類：惡性 vs 良性）
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"總資料: {len(y)} 樣本 ({np.sum(y==0)} 惡性, {np.sum(y==1)} 良性)")
print(f"訓練集: {len(y_train)} 樣本")
print(f"測試集: {len(y_test)} 樣本")

# stratify=y 保證訓練集和測試集的類別比例相同
print(f"\n類別比例:")
print(f"  總資料: {np.mean(y==1):.1%} 良性")
print(f"  訓練集: {np.mean(y_train==1):.1%} 良性")
print(f"  測試集: {np.mean(y_test==1):.1%} 良性")
print(f"  → stratify 確保比例一致")


# ============================================================================
# Part B: 分類指標
# ============================================================================
print("\n" + "=" * 60)
print("Part B: 分類指標 — 不是只有 Accuracy")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = LogisticRegression(max_iter=5000)
clf.fit(X_train_s, y_train)
y_pred = clf.predict(X_test_s)

# --- 手動計算各指標 ---
# 以「惡性 (0)」為正類（我們最關心的是能不能抓到惡性腫瘤）
TP = np.sum((y_pred == 0) & (y_test == 0))   # True Positive: 預測惡性，確實惡性
FP = np.sum((y_pred == 0) & (y_test == 1))   # False Positive: 預測惡性，其實良性
FN = np.sum((y_pred == 1) & (y_test == 0))   # False Negative: 預測良性，其實惡性（最危險！）
TN = np.sum((y_pred == 1) & (y_test == 1))   # True Negative: 預測良性，確實良性

print(f"混淆矩陣元素:")
print(f"  TP (抓到惡性): {TP}")
print(f"  FP (誤報惡性): {FP}")
print(f"  FN (漏掉惡性): {FN}  ← 最危險！")
print(f"  TN (正確良性): {TN}")

accuracy = (TP + TN) / (TP + FP + FN + TN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"""
指標解讀：
  Accuracy  = {accuracy:.4f}  → 整體正確率
  Precision = {precision:.4f}  → 預測惡性中，真的是惡性的比例
                              → 高 Precision = 很少誤報
  Recall    = {recall:.4f}  → 所有惡性中，被抓到的比例
                              → 高 Recall = 很少漏掉（醫療場景最重要！）
  F1 Score  = {f1:.4f}  → Precision 和 Recall 的調和平均

什麼時候用什麼指標？
  - 醫療診斷: 重視 Recall（不能漏掉病人）
  - 垃圾郵件: 重視 Precision（不能把正常信標成垃圾）
  - 一般場景: F1 Score（平衡兩者）
  - 類別平衡: Accuracy 就夠了
""")


# ============================================================================
# Part C: 混淆矩陣 (Confusion Matrix)
# ============================================================================
print("=" * 60)
print("Part C: 混淆矩陣")
print("=" * 60)

from sklearn.metrics import (confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)

cm = confusion_matrix(y_test, y_pred)
print(f"混淆矩陣:\n{cm}")
print(f"""
     預測 0  預測 1
實際 0 [{cm[0,0]:3d}]  [{cm[0,1]:3d}]     ← 惡性
實際 1 [{cm[1,0]:3d}]  [{cm[1,1]:3d}]     ← 良性
""")

print("完整分類報告:")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# 畫混淆矩陣
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Malignant', 'Benign'])
disp.plot(ax=ax, cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('phase-1-ml-basics/plots/07_confusion_matrix.png', dpi=100)
plt.close()
print("→ 圖表儲存至 plots/07_confusion_matrix.png")


# ============================================================================
# Part D: 交叉驗證 (Cross Validation)
# ============================================================================
print("\n" + "=" * 60)
print("Part D: 交叉驗證")
print("=" * 60)

print("""
問題：只切一次 Train/Test，結果可能因為切法不同而有運氣成分。

K-Fold 交叉驗證：
  1. 把資料分成 K 份
  2. 每次用 K-1 份訓練，1 份驗證
  3. 重複 K 次，每份都當過驗證集
  4. 取 K 次的平均結果

  → 更穩定、更可靠的評估
""")

from sklearn.model_selection import cross_val_score

# --- NumPy 手刻 K-Fold ---
print("--- NumPy 手刻 5-Fold ---")

def k_fold_split(n_samples, k=5, shuffle=True, seed=42):
    idx = np.arange(n_samples)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(idx)

    fold_size = n_samples // k
    folds = []
    for i in range(k):
        val_idx = idx[i * fold_size: (i + 1) * fold_size]
        train_idx = np.concatenate([idx[:i * fold_size], idx[(i + 1) * fold_size:]])
        folds.append((train_idx, val_idx))
    return folds

scores_manual = []
folds = k_fold_split(len(X), k=5)

for i, (train_idx, val_idx) in enumerate(folds):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_tr_s, y_tr)
    acc = model.score(X_val_s, y_val)
    scores_manual.append(acc)
    print(f"  Fold {i+1}: {acc:.4f}")

print(f"  平均: {np.mean(scores_manual):.4f} ± {np.std(scores_manual):.4f}")

# --- Scikit-learn 版 ---
print("\n--- Scikit-learn cross_val_score ---")
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
scores_sk = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

for i, s in enumerate(scores_sk):
    print(f"  Fold {i+1}: {s:.4f}")
print(f"  平均: {scores_sk.mean():.4f} ± {scores_sk.std():.4f}")


# ============================================================================
# Part E: Bias-Variance Tradeoff
# ============================================================================
print("\n" + "=" * 60)
print("Part E: Bias-Variance Tradeoff")
print("=" * 60)

print("""
模型的誤差 = Bias + Variance + Noise（不可約誤差）

  Bias (偏差)：模型太簡單，抓不到規律 → 欠擬合
  Variance (變異)：模型太複雜，連噪音都學了 → 過擬合

  模型複雜度 ↑ → Bias ↓, Variance ↑
  模型複雜度 ↓ → Bias ↑, Variance ↓

  目標：找到甜蜜點
""")

# 用決策樹的不同深度來展示
from sklearn.tree import DecisionTreeClassifier

depths = range(1, 20)
train_scores = []
test_scores = []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train_s, y_train)
    train_scores.append(dt.score(X_train_s, y_train))
    test_scores.append(dt.score(X_test_s, y_test))

plt.figure(figsize=(10, 5))
plt.plot(depths, train_scores, 'bo-', label='Train Accuracy', linewidth=2)
plt.plot(depths, test_scores, 'ro-', label='Test Accuracy', linewidth=2)
plt.xlabel('Tree Depth (Model Complexity)')
plt.ylabel('Accuracy')
plt.title('Bias-Variance Tradeoff')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# 標記區域
best_depth = depths[np.argmax(test_scores)]
plt.axvline(x=best_depth, color='green', linestyle='--', alpha=0.7)
plt.annotate('Sweet Spot', xy=(best_depth, max(test_scores)),
             xytext=(best_depth + 3, max(test_scores) - 0.03),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=12, color='green')

plt.tight_layout()
plt.savefig('phase-1-ml-basics/plots/07_bias_variance.png', dpi=100)
plt.close()
print("→ 圖表儲存至 plots/07_bias_variance.png")
print(f"  最佳深度: {best_depth} (Test Acc: {max(test_scores):.2%})")


# ============================================================================
# Part F: 完整的模型比較流程
# ============================================================================
print("\n" + "=" * 60)
print("Part F: 完整模型比較")
print("=" * 60)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models = {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    'KNN (k=5)':           make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)':           make_pipeline(StandardScaler(), SVC(kernel='rbf')),
}

print(f"用 5-Fold Cross Validation 比較所有模型:\n")
print(f"{'Model':>25s}  {'Mean':>7s}  {'Std':>7s}  {'Scores'}")
print("-" * 70)

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results[name] = scores
    scores_str = ', '.join([f'{s:.3f}' for s in scores])
    print(f"{name:>25s}  {scores.mean():7.4f}  {scores.std():7.4f}  [{scores_str}]")

# 畫箱形圖比較
plt.figure(figsize=(10, 5))
plt.boxplot([results[name] for name in results],
            labels=[name.replace(' ', '\n') for name in results],
            patch_artist=True)
plt.ylabel('Accuracy')
plt.title('Model Comparison (5-Fold CV)')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('phase-1-ml-basics/plots/07_model_comparison.png', dpi=100)
plt.close()
print("\n→ 圖表儲存至 plots/07_model_comparison.png")


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結")
print("=" * 60)
print("""
模型評估教會你的核心概念：

  概念                   對應到深度學習
  ──────────────────────────────────────────────────
  Train/Val/Test Split  →  完全一樣，深度學習也這樣切
  Overfitting           →  模型太大、資料太少時必定發生
  Cross Validation      →  小資料集用 CV，大資料集直接切
  Precision / Recall    →  分類模型的關鍵指標
  Bias-Variance         →  模型選擇的核心考量

實務檢查清單：
  □ 資料有沒有 leak（測試集資訊洩漏到訓練中）
  □ Train/Test 分佈是否一致
  □ 用正確的指標（不平衡資料不要只看 Accuracy）
  □ 報告標準差，不是只有平均值
  □ 比較基準線（baseline），不是只看絕對數字

═══════════════════════════════════════════
  Phase 1 完成！
  下一步：Phase 2 — 深度學習基礎 (PyTorch)
  → phase-2-deep-learning/
═══════════════════════════════════════════
""")
