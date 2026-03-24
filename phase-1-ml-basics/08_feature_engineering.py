"""
=============================================================================
Phase 1-8: 特徵工程 — 重塑資料空間 (Feature Engineering)
=============================================================================

核心問題：
  SVM 在固定的資料空間中找超平面。
  但如果換一個「座標系」來描述同一批資料，分類可能變得很簡單。
  → 這就是特徵工程的本質：換一副眼鏡看資料。

從 SVM 的 Kernel Trick 說起：
  Kernel Trick 是「隱式」地把資料映射到高維。
  特徵工程是「顯式」地設計新特徵，改變資料的幾何排列。
  兩者的目的相同：讓原本線性不可分的資料變得可分。

本檔涵蓋：
  Part A: 為什麼需要特徵轉換 — 環形資料的困境
  Part B: 四種經典轉換策略（多項式、極座標、交互特徵、對數）
  Part C: 互動式視覺化 — 切換不同轉換，觀察幾何變化
  Part D: 特徵工程的本質 — 重新定義距離的尺規
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang HK']
matplotlib.rcParams['axes.unicode_minus'] = False
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ============================================================================
# Part A: 為什麼需要特徵轉換
# ============================================================================
print("=" * 60)
print("Part A: 環形資料 — 線性分類的困境")
print("=" * 60)

print("""
問題設定：
  內圈 = 類別 0（藍色），外圈 = 類別 1（橘色）
  在原始 (x₁, x₂) 空間中，沒有任何直線能把它們分開。

  但如果我們換個角度看這些資料呢？
""")

np.random.seed(42)
X, y = make_circles(n_samples=300, noise=0.08, factor=0.35, random_state=42)

lr_original = LogisticRegression()
lr_original.fit(X, y)
acc_original = accuracy_score(y, lr_original.predict(X))
print(f"原始空間 (x₁, x₂) + Logistic Regression: {acc_original:.2%}  ← 接近隨機猜測")


# ============================================================================
# Part B: 四種特徵轉換策略 — 計算準確率
# ============================================================================
print("\n" + "=" * 60)
print("Part B: 四種特徵轉換 — 讓不可分變可分")
print("=" * 60)

# 預先計算所有轉換
r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
theta = np.arctan2(X[:, 1], X[:, 0])
r_safe = np.maximum(r, 1e-10)

# 策略 1: r²
X_r2 = np.column_stack([X, X[:, 0]**2 + X[:, 1]**2])
lr_r2 = LogisticRegression()
lr_r2.fit(X_r2, y)
acc_r2 = accuracy_score(y, lr_r2.predict(X_r2))

# 策略 2: 極座標
X_polar = np.column_stack([r, theta])
lr_polar = LogisticRegression()
lr_polar.fit(X_polar, y)
acc_polar = accuracy_score(y, lr_polar.predict(X_polar))

# 策略 3: 多項式
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
lr_poly = LogisticRegression()
lr_poly.fit(X_poly, y)
acc_poly = accuracy_score(y, lr_poly.predict(X_poly))

# 策略 4: log(r)
X_log = np.column_stack([np.log(r_safe), theta])
lr_log = LogisticRegression()
lr_log.fit(X_log, y)
acc_log = accuracy_score(y, lr_log.predict(X_log))

# SVM RBF
svm_rbf = SVC(kernel='rbf', gamma='auto')
svm_rbf.fit(X, y)
acc_rbf = svm_rbf.score(X, y)

print(f"""
準確率比較：
  原始 (x₁, x₂)         : {acc_original:.2%}  ← 不可分
  加入 r²                : {acc_r2:.2%}
  極座標 (r, θ)          : {acc_polar:.2%}
  多項式 degree=2        : {acc_poly:.2%}
  log(r), θ              : {acc_log:.2%}
  SVM RBF (隱式映射)     : {acc_rbf:.2%}

結論：同一批資料，換一個特徵描述方式，分類效果天差地別。
""")


# ============================================================================
# Part C: 互動式視覺化 — 切換不同轉換，觀察幾何變化
# ============================================================================
print("=" * 60)
print("Part C: 互動式特徵轉換視覺化")
print("=" * 60)
print("使用下方的標籤頁切換不同的特徵轉換，觀察資料幾何如何改變。\n")

# --- 建立互動式 Tab 視覺化 ---
out_fe = widgets.Output()

def draw_transform(transform_name):
    """根據選擇的轉換方式，畫出原始空間 vs 變換後空間"""
    with out_fe:
        clear_output(wait=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # 左圖：永遠是原始空間
        ax_left = axes[0]
        ax_left.scatter(X[y == 0, 0], X[y == 0, 1], c='#378ADD', s=20, alpha=0.7, label='Class 0 (內圈)')
        ax_left.scatter(X[y == 1, 0], X[y == 1, 1], c='#D85A30', s=20, alpha=0.7, label='Class 1 (外圈)')
        ax_left.set_title('原始空間 (x₁, x₂)', fontsize=12, fontweight='bold')
        ax_left.set_xlabel('x₁')
        ax_left.set_ylabel('x₂')
        ax_left.legend(fontsize=9)
        ax_left.grid(True, alpha=0.3)
        ax_left.set_aspect('equal')

        # 右圖：根據選擇的轉換
        ax_right = axes[1]

        if transform_name == '原始 (無轉換)':
            ax_right.scatter(X[y == 0, 0], X[y == 0, 1], c='#378ADD', s=20, alpha=0.7)
            ax_right.scatter(X[y == 1, 0], X[y == 1, 1], c='#D85A30', s=20, alpha=0.7)
            ax_right.set_xlabel('x₁')
            ax_right.set_ylabel('x₂')
            ax_right.set_aspect('equal')
            formula = 'f(x) = (x₁, x₂)'
            desc = '無轉換，線性不可分'
            acc = acc_original

        elif transform_name == '加入 r² = x₁²+x₂²':
            ax_right.scatter(X[y == 0, 0], X_r2[y == 0, 2], c='#378ADD', s=20, alpha=0.7)
            ax_right.scatter(X[y == 1, 0], X_r2[y == 1, 2], c='#D85A30', s=20, alpha=0.7)
            ax_right.axhline(y=0.3, color='#639922', linestyle='--', lw=2, alpha=0.8, label='分界線')
            ax_right.set_xlabel('x₁')
            ax_right.set_ylabel('r² = x₁² + x₂²')
            ax_right.legend(fontsize=9)
            formula = 'f(x) = (x₁, x₂, x₁²+x₂²)'
            desc = '距離資訊浮現 → 水平線可分'
            acc = acc_r2

        elif transform_name == '極座標 (r, θ)':
            ax_right.scatter(theta[y == 0], r[y == 0], c='#378ADD', s=20, alpha=0.7)
            ax_right.scatter(theta[y == 1], r[y == 1], c='#D85A30', s=20, alpha=0.7)
            ax_right.axhline(y=0.55, color='#639922', linestyle='--', lw=2, alpha=0.8, label='分界線')
            ax_right.set_xlabel('θ (角度)')
            ax_right.set_ylabel('r (距離)')
            ax_right.legend(fontsize=9)
            formula = '(x₁,x₂) → (r, θ)    r=√(x₁²+x₂²)'
            desc = '圓形邊界 → 水平帶狀，直線可分'
            acc = acc_polar

        elif transform_name == '多項式特徵 (degree=2)':
            ax_right.scatter(X_poly[y == 0, 2], X_poly[y == 0, 4], c='#378ADD', s=20, alpha=0.7)
            ax_right.scatter(X_poly[y == 1, 2], X_poly[y == 1, 4], c='#D85A30', s=20, alpha=0.7)
            ax_right.set_xlabel('x₁²')
            ax_right.set_ylabel('x₂²')
            formula = '[x₁, x₂] → [x₁, x₂, x₁², x₁x₂, x₂²]'
            desc = '暴力升維，不需知道資料結構'
            acc = acc_poly

        elif transform_name == 'log(r) 對數距離':
            ax_right.scatter(theta[y == 0], np.log(r_safe[y == 0]), c='#378ADD', s=20, alpha=0.7)
            ax_right.scatter(theta[y == 1], np.log(r_safe[y == 1]), c='#D85A30', s=20, alpha=0.7)
            ax_right.axhline(y=-0.6, color='#639922', linestyle='--', lw=2, alpha=0.8, label='分界線')
            ax_right.set_xlabel('θ (角度)')
            ax_right.set_ylabel('log(r)')
            ax_right.legend(fontsize=9)
            formula = '(x₁,x₂) → (log(r), θ)'
            desc = '拉開近距離差異，壓縮遠距離差異'
            acc = acc_log

        elif transform_name == 'SVM RBF (隱式映射)':
            h = 0.02
            x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
            y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = svm_rbf.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax_right.contourf(xx, yy, Z, alpha=0.2, cmap='RdBu')
            ax_right.scatter(X[y == 0, 0], X[y == 0, 1], c='#378ADD', s=20, alpha=0.7)
            ax_right.scatter(X[y == 1, 0], X[y == 1, 1], c='#D85A30', s=20, alpha=0.7)
            ax_right.set_xlabel('x₁')
            ax_right.set_ylabel('x₂')
            ax_right.set_aspect('equal')
            formula = 'K(x,z) = exp(-γ||x-z||²)'
            desc = 'Kernel 隱式映射到高維，自動找到非線性邊界'
            acc = acc_rbf

        ax_right.set_title(f'變換後空間  Acc: {acc:.0%}', fontsize=12, fontweight='bold')
        ax_right.grid(True, alpha=0.3)

        fig.suptitle(f'公式: {formula}\n{desc}', fontsize=11, y=0.02,
                     color='#555', style='italic')
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.show()

transform_tabs = widgets.ToggleButtons(
    options=['原始 (無轉換)', '加入 r² = x₁²+x₂²', '極座標 (r, θ)',
             '多項式特徵 (degree=2)', 'log(r) 對數距離', 'SVM RBF (隱式映射)'],
    value='原始 (無轉換)',
    description='特徵轉換:',
    style={'description_width': 'initial', 'button_width': 'auto'},
)

def on_tab_change(change):
    draw_transform(change['new'])

transform_tabs.observe(on_tab_change, names='value')
display(transform_tabs, out_fe)
draw_transform('原始 (無轉換)')


# ============================================================================
# Part D: 特徵工程的本質與極限
# ============================================================================
print("\n" + "=" * 60)
print("Part D: 特徵工程的本質 — 重新定義距離")
print("=" * 60)

print("""
核心洞察：
  每一種特徵轉換，都是在重新定義「什麼叫做相近」。

  原始空間:  歐式距離 d(x, y) = ||x - y||
  極座標後:  用距離原點的 r 來衡量 → 同心圓上的點「等價」
  加入 r²:   距離原點的距離成為主要特徵
  log(r):    拉開近處差異，壓縮遠處差異

  本質上就是：告訴模型「用這個視角看資料」。

特徵工程的三大來源：
  1. 數學直覺：看到圓形 → 用極座標
  2. 領域知識：房價預測中 → 加入「坪效 = 價格/面積」
  3. 統計觀察：某特徵偏態嚴重 → log 轉換

特徵工程 vs Kernel Trick vs 神經網路：

  方法            誰設計轉換？    需要什麼？         可擴展性
  ─────────────────────────────────────────────────────────
  特徵工程        人工手動        領域知識 + 直覺     受限於人的智慧
  Kernel Trick    數學公式        選擇正確的 kernel   固定的映射函數
  神經網路        自動學習        大量資料 + 算力     可學任意轉換 ←

  → 深度學習的每一層隱藏層，都在做一次「自動特徵工程」。
  → 但特徵工程並未消失——在結構化資料（表格、時序）中，
    人工設計的特徵仍然是最有效的手段，甚至優於深度學習。
""")


# ============================================================================
# 小結
# ============================================================================
print("=" * 60)
print("小結")
print("=" * 60)
print("""
特徵工程的核心：

  1. 資料在空間中的相對位置取決於「你用什麼座標系描述它」
  2. 換一個座標系（特徵轉換），就是換一副眼鏡
  3. 好的轉換讓同類自然聚攏、異類自然分開
  4. 轉換後，一條簡單的直線就能分類

  方法              本質
  ────────────────────────────────
  加入 r²          → 讓距離資訊浮現
  極座標            → 把圓形邊界變水平線
  多項式特徵        → 暴力升維，通用但不精準
  log 轉換         → 重新定義距離的尺規

與其他方法的關係：
  特徵工程：人工設計空間轉換（Phase 1 的延伸）
  表示學習：讓神經網路自動學習空間轉換（Phase 2 的核心）
  度量學習：直接把「讓同類靠近」當作訓練目標（Phase 2 進階）

下一步：09_representation_learning.py — 從人工到自動的特徵學習
""")
