# Phase 1 — 機器學習核心概念

## 目標
理解「模型怎麼學」，每個演算法都有 NumPy 手刻版 + Scikit-learn 版。

## 檔案順序

| 順序 | 檔案 | 主題 |
|------|------|------|
| 1 | `01_linear_regression.py` | 線性回歸：最簡單的模型，理解梯度下降 |
| 2 | `02_logistic_regression.py` | 邏輯回歸：分類問題的起點 |
| 3 | `03_knn.py` | K-近鄰：最直覺的分類方式 |
| 4 | `04_decision_tree.py` | 決策樹 + 隨機森林：非線性與集成學習 |
| 5 | `05_svm.py` | 支持向量機：理解「邊界」的概念 |
| 6 | `06_unsupervised.py` | 非監督式學習：KMeans + PCA |
| 7 | `07_model_evaluation.py` | 模型評估：指標、交叉驗證、Bias-Variance |
| 8 | `08_feature_engineering.py` | 特徵工程：轉換策略與幾何直覺 |

## 學習方式

每個檔案的結構：
```
Part A: 概念解說（這個演算法在做什麼）
Part B: NumPy 手刻版（理解內部運作）
Part C: Scikit-learn 版（實務寫法）
Part D: 比較與小結
```

先跑 NumPy 版理解原理，再看 sklearn 版知道實務怎麼用。
