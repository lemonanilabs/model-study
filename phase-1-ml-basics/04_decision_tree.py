"""
=============================================================================
Phase 1-4: 決策樹 (Decision Tree) + 隨機森林 (Random Forest)
=============================================================================

決策樹的直覺：
  就像你在做是非題——
  「特徵 1 > 5？」→ 是 → 「特徵 2 > 3？」→ 是 → 類別 A
                   → 否 → 類別 B

隨機森林 = 很多棵決策樹一起投票 (Ensemble Learning)

本檔涵蓋：
  Part A: 決策樹的分裂準則（Gini / Entropy）
  Part B: NumPy 手刻簡單決策樹
  Part C: Scikit-learn 決策樹
  Part D: 隨機森林 — 集成學習的威力
  Part E: 特徵重要性
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Part A: 分裂準則 — Gini Impurity vs Entropy
# ============================================================================
print("=" * 60)
print("Part A: 分裂準則 — 怎麼決定「問什麼問題」")
print("=" * 60)

def gini_impurity(labels):
    """Gini Impurity: 隨機抽兩個樣本，類別不同的機率"""
    if len(labels) == 0:
        return 0
    classes, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return 1 - np.sum(probs ** 2)

def entropy(labels):
    """Entropy: 資訊的不確定性"""
    if len(labels) == 0:
        return 0
    classes, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# 不同分佈的 Gini 和 Entropy
distributions = {
    "全部 A [A,A,A,A]":          np.array([0, 0, 0, 0]),
    "平均 [A,A,B,B]":            np.array([0, 0, 1, 1]),
    "偏向 [A,A,A,B]":            np.array([0, 0, 0, 1]),
    "三類平均 [A,B,C]":          np.array([0, 1, 2]),
}

print(f"{'分佈':>25s}  {'Gini':>6s}  {'Entropy':>8s}")
print("-" * 45)
for name, labels in distributions.items():
    g = gini_impurity(labels)
    e = entropy(labels)
    print(f"{name:>25s}  {g:6.3f}  {e:8.3f}")

print("""
觀察：
  - 全部同類 → Gini=0, Entropy=0（最純，最好）
  - 平均分佈 → 值最大（最不純，最差）

  決策樹的目標：每次分裂後，讓子節點的「不純度」降最多
  Information Gain = 父節點不純度 - 子節點加權平均不純度
""")


# ============================================================================
# Part B: NumPy 手刻簡單決策樹
# ============================================================================
print("=" * 60)
print("Part B: NumPy 手刻決策樹")
print("=" * 60)

class SimpleDecisionTree:
    """
    最簡單的決策樹實作：
    - 只做二元分裂（feature > threshold）
    - 用 Gini Impurity 選最佳分裂
    - 遞迴建樹，有最大深度限制
    """

    def __init__(self, max_depth=5, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None

    def _gini(self, labels):
        if len(labels) == 0:
            return 0
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        """找到最佳的特徵和閾值來分裂"""
        best_gain = -1
        best_feature = None
        best_threshold = None

        parent_gini = self._gini(y)
        n = len(y)

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < 1 or right_mask.sum() < 1:
                    continue

                # 加權 Gini
                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                weighted_gini = (left_mask.sum() / n * left_gini +
                                 right_mask.sum() / n * right_gini)

                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build(self, X, y, depth):
        """遞迴建樹"""
        # 停止條件
        if (depth >= self.max_depth or
            len(y) < self.min_samples or
            len(np.unique(y)) == 1):
            # 葉節點：回傳多數類別
            counts = np.bincount(y)
            return {"leaf": True, "class": counts.argmax(), "samples": len(y)}

        feature, threshold, gain = self._best_split(X, y)

        if gain <= 0:
            counts = np.bincount(y)
            return {"leaf": True, "class": counts.argmax(), "samples": len(y)}

        left_mask = X[:, feature] <= threshold
        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "gain": gain,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, depth=0)

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["class"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.tree
        if node["leaf"]:
            print(f"{indent}→ Class {node['class']} (n={node['samples']})")
        else:
            print(f"{indent}Feature[{node['feature']}] <= {node['threshold']:.2f}  "
                  f"(gain={node['gain']:.4f})")
            print(f"{indent}├─ Yes:")
            self.print_tree(node["left"], indent + "│  ")
            print(f"{indent}└─ No:")
            self.print_tree(node["right"], indent + "   ")


# 測試
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

tree = SimpleDecisionTree(max_depth=3)
tree.fit(X_train, y_train)

print("手刻決策樹結構：")
tree.print_tree()

y_pred = tree.predict(X_test)
acc = np.mean(y_pred == y_test)
print(f"\n手刻決策樹準確率: {acc:.2%}")


# ============================================================================
# Part C: Scikit-learn 決策樹
# ============================================================================
print("\n" + "=" * 60)
print("Part C: Scikit-learn 決策樹")
print("=" * 60)

from sklearn.tree import DecisionTreeClassifier, export_text

# 訓練
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

print(f"sklearn 決策樹準確率: {clf.score(X_test, y_test):.2%}")

# 印出樹結構
print(f"\nsklearn 樹結構：")
print(export_text(clf, feature_names=iris.feature_names, max_depth=3))


# ============================================================================
# Part D: 隨機森林 — 集成學習
# ============================================================================
print("=" * 60)
print("Part D: 隨機森林 — 多棵樹的力量")
print("=" * 60)

print("""
隨機森林的核心思想（Bagging）：
  1. 從訓練集隨機抽樣（有放回）建立多個子集
  2. 每個子集訓練一棵決策樹
  3. 每棵樹只看「隨機選取的部分特徵」
  4. 預測時：所有樹投票，多數決

為什麼比單棵樹好？
  - 單棵樹容易過擬合（太敏感）
  - 多棵樹平均後，噪音被抵消，更穩定
  - 「三個臭皮匠，勝過一個諸葛亮」
""")

# --- NumPy 手刻簡單版隨機森林 ---
class SimpleRandomForest:
    def __init__(self, n_trees=10, max_depth=5, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.max_features == 'sqrt':
            n_select = int(np.sqrt(n_features))
        else:
            n_select = n_features

        for _ in range(self.n_trees):
            # Bootstrap sampling（有放回抽樣）
            idx = np.random.randint(0, n_samples, size=n_samples)
            X_boot = X[idx]
            y_boot = y[idx]

            # 隨機選取特徵
            feat_idx = np.random.choice(n_features, size=n_select, replace=False)
            self.feature_indices.append(feat_idx)

            # 訓練一棵樹
            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X_boot[:, feat_idx], y_boot)
            self.trees.append(tree)

    def predict(self, X):
        # 每棵樹投票
        all_preds = []
        for tree, feat_idx in zip(self.trees, self.feature_indices):
            pred = tree.predict(X[:, feat_idx])
            all_preds.append(pred)

        all_preds = np.array(all_preds)   # (n_trees, n_samples)
        # 多數決
        final = np.array([np.bincount(all_preds[:, i]).argmax()
                          for i in range(X.shape[0])])
        return final


np.random.seed(42)
rf_numpy = SimpleRandomForest(n_trees=20, max_depth=5)
rf_numpy.fit(X_train, y_train)
y_pred_rf = rf_numpy.predict(X_test)
print(f"手刻隨機森林 (20 trees) 準確率: {np.mean(y_pred_rf == y_test):.2%}")

# --- Scikit-learn 版 ---
from sklearn.ensemble import RandomForestClassifier

rf_sk = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_sk.fit(X_train, y_train)
print(f"sklearn 隨機森林 (100 trees) 準確率: {rf_sk.score(X_test, y_test):.2%}")

# 比較單棵樹 vs 隨機森林
print(f"\n--- 比較 ---")
print(f"  單棵決策樹:     {clf.score(X_test, y_test):.2%}")
print(f"  手刻隨機森林:   {np.mean(y_pred_rf == y_test):.2%}")
print(f"  sklearn 隨機森林: {rf_sk.score(X_test, y_test):.2%}")


# ============================================================================
# Part E: 特徵重要性
# ============================================================================
print("\n" + "=" * 60)
print("Part E: 特徵重要性")
print("=" * 60)

# 隨機森林可以告訴你哪些特徵最有用
importances = rf_sk.feature_importances_
indices = np.argsort(importances)[::-1]

print("特徵重要性排名 (Iris):")
for i in indices:
    print(f"  {iris.feature_names[i]:>20s}: {importances[i]:.4f} {'█' * int(importances[i] * 50)}")

plt.figure(figsize=(8, 4))
plt.bar(range(4), importances[indices], color='steelblue')
plt.xticks(range(4), [iris.feature_names[i] for i in indices], rotation=15)
plt.ylabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('phase-1-ml-basics/plots/04_feature_importance.png', dpi=100)
plt.close()
print("→ 圖表儲存至 plots/04_feature_importance.png")


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結")
print("=" * 60)
print("""
決策樹與隨機森林教會你的核心概念：

  概念               對應到深度學習
  ───────────────────────────────────────────────
  非線性分割          →  激活函數提供非線性能力
  過擬合（深度太大）   →  模型太複雜的風險
  集成學習 (Ensemble) →  Model Ensemble、多模型投票
  特徵重要性          →  Attention 權重、可解釋性
  Bagging            →  Dropout 有類似的「隨機」精神

sklearn 速查：
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.ensemble import RandomForestClassifier

  # 決策樹
  dt = DecisionTreeClassifier(max_depth=5)
  dt.fit(X_train, y_train)

  # 隨機森林
  rf = RandomForestClassifier(n_estimators=100)
  rf.fit(X_train, y_train)
  rf.feature_importances_    # 特徵重要性

下一步：05_svm.py — 支持向量機
""")
