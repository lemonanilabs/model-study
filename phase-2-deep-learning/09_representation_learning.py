"""
=============================================================================
Phase 2-9: 表示學習、度量學習與幾何湧現
         (Representation Learning, Metric Learning & Emergent Geometry)
=============================================================================

從特徵工程到這裡的思路演進：
  Phase 1-8 特徵工程：人工設計座標轉換，讓資料可分
  Phase 2-9 表示學習：讓神經網路自動學習座標轉換

核心問題：
  深度學習的隱藏層在做什麼？
  → 每一層 σ(Wx + b) 都是一次空間變換
  → 訓練過程透過梯度下降，把參數 W 調整到
    使得變換後的空間具有「幾何意義」
  → 這個幾何意義不是人設計的，是被 Loss 函數「逼」出來的

本檔涵蓋：
  Part A: 表示學習 — 互動觀察訓練如何改變每一層的空間
  Part B: 嵌入空間 — 互動探索詞向量的幾何關係
  Part C: 度量學習 — 拖拉 Triplet Loss 訓練過程
  Part D: 三種範式比較 — 監督式、自監督、度量學習
  Part E: 幾何湧現 — 用滑桿觀察梯度逐步雕刻空間
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang HK']
matplotlib.rcParams['axes.unicode_minus'] = False
import ipywidgets as widgets
from IPython.display import display, clear_output

# ============================================================================
# Part A: 表示學習 — 互動觀察「訓練進度」如何改變空間
# ============================================================================
print("=" * 60)
print("Part A: 表示學習 — 深度網路如何學出有意義的空間")
print("=" * 60)

print("""
核心機制：
  每一層計算: h = σ(Wx + b)

  W = 參數矩陣 → 對資料做線性變換（旋轉、縮放、投影）
  σ = 非線性激活 → 扭曲空間（ReLU 把負值折到 0）
  b = 偏移      → 平移座標原點

  N 層網路 = N 次連續的「旋轉 + 扭曲」
  把原始資料從一個幾何空間「變形」到另一個幾何空間

最關鍵的洞察：
  梯度從來沒有說「請讓貓和狗靠近」。
  它只說「讓預測更準確」(∂Loss/∂W)。
  但為了讓預測更準確，網路唯一能做的事就是
  把特徵整理成有意義的幾何結構——因為這樣分類器才能輕鬆區分。
  幾何意義是被梯度「逼」出來的，是訓練的湧現 (emergence)。

▼ 拖動下方滑桿，觀察「訓練進度」如何改變每一層的空間表示：
""")

# 生成螺旋形資料
np.random.seed(42)
N = 200
t_spiral = np.linspace(0, 2 * np.pi, N // 2)
noise = 0.3

X_spiral = np.vstack([
    np.column_stack([t_spiral * np.cos(t_spiral) + np.random.randn(N // 2) * noise,
                     t_spiral * np.sin(t_spiral) + np.random.randn(N // 2) * noise]),
    np.column_stack([t_spiral * np.cos(t_spiral + np.pi) + np.random.randn(N // 2) * noise,
                     t_spiral * np.sin(t_spiral + np.pi) + np.random.randn(N // 2) * noise])
])
y_spiral = np.array([0] * (N // 2) + [1] * (N // 2))
X_spiral = (X_spiral - X_spiral.mean(axis=0)) / X_spiral.std(axis=0)

def relu(x):
    return np.maximum(0, x)

# 隨機權重
W1_random = np.random.randn(2, 8) * 0.5
b1_random = np.random.randn(8) * 0.1
W2_random = np.random.randn(8, 4) * 0.5
b2_random = np.random.randn(4) * 0.1

# 「訓練過的」權重
W1_trained = np.array([
    [1.5, -1.5, 0.8, -0.8, 1.2, -1.2, 0.5, -0.5],
    [0.8, 0.8, -1.5, 1.5, -0.5, 0.5, 1.2, -1.2]
])
b1_trained = np.array([0.1, 0.1, -0.3, 0.3, 0.2, -0.2, -0.1, 0.1])
W2_trained = np.array([
    [1.0, 0.5, -0.3, 0.8], [-0.5, 1.0, 0.8, -0.3],
    [0.3, -0.8, 1.0, 0.5], [-0.8, 0.3, -0.5, 1.0],
    [0.6, -0.4, 0.7, -0.6], [-0.4, 0.6, -0.6, 0.7],
    [0.9, -0.2, 0.4, -0.9], [-0.2, 0.9, -0.9, 0.4]
])
b2_trained = np.array([0.0, 0.0, 0.0, 0.0])

colors_spiral = ['#378ADD' if label == 0 else '#D85A30' for label in y_spiral]

out_a = widgets.Output()

def draw_layer_transforms(training_progress=0):
    """training_progress: 0 ~ 100，模擬從隨機到訓練完成"""
    with out_a:
        clear_output(wait=True)
        t = training_progress / 100.0

        # 插值權重
        W1 = W1_random * (1 - t) + W1_trained * t
        b1 = b1_random * (1 - t) + b1_trained * t
        W2 = W2_random * (1 - t) + W2_trained * t
        b2 = b2_random * (1 - t) + b2_trained * t

        h1 = relu(X_spiral @ W1 + b1)
        h2 = relu(h1 @ W2 + b2)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].scatter(X_spiral[:, 0], X_spiral[:, 1], c=colors_spiral, s=12, alpha=0.6)
        axes[0].set_title('輸入空間\n(螺旋形，線性不可分)', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal')
        axes[0].set_xlim(-3, 3)
        axes[0].set_ylim(-3, 3)

        axes[1].scatter(h1[:, 0], h1[:, 1], c=colors_spiral, s=12, alpha=0.6)
        status1 = '混亂' if t < 0.3 else '出現結構' if t < 0.7 else '結構清晰'
        axes[1].set_title(f'Layer 1: σ(W₁x + b₁)\n({status1})', fontsize=11)
        axes[1].grid(True, alpha=0.3)

        axes[2].scatter(h2[:, 0], h2[:, 1], c=colors_spiral, s=12, alpha=0.6)
        status2 = '混亂' if t < 0.3 else '開始分開' if t < 0.7 else '兩類分離'
        axes[2].set_title(f'Layer 2: σ(W₂h₁ + b₂)\n({status2})', fontsize=11)
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(f'訓練進度: {training_progress}%  —  '
                     f'每一層 σ(Wx+b) 都在變換空間，訓練讓變換產生幾何意義',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

slider_a = widgets.IntSlider(
    value=0, min=0, max=100, step=5,
    description='訓練進度:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='600px'),
    continuous_update=False
)
play_a = widgets.Play(value=0, min=0, max=100, step=5, interval=300)
widgets.jslink((play_a, 'value'), (slider_a, 'value'))

slider_a.observe(lambda change: draw_layer_transforms(change['new']), names='value')
display(widgets.HBox([play_a, slider_a]), out_a)
draw_layer_transforms(0)


# ============================================================================
# Part B: 嵌入空間的幾何性質 — 互動探索向量算術
# ============================================================================
print("\n" + "=" * 60)
print("Part B: 嵌入空間的幾何性質 — 語意 = 幾何")
print("=" * 60)

print("""
Word2Vec 的驚人發現：
  在訓練出的嵌入空間中，語意關係對應幾何方向。

  king - man + woman ≈ queen
  Paris - France + Japan ≈ Tokyo

  這不只是把相似的東西放在一起，
  還保留了語意關係的「方向性」：
  - 「性別」是空間中的一個固定方向
  - 「首都 ↔ 國家」是另一個固定方向

  這些方向不是人設計的，是網路從大量文字中自己學到的。

▼ 選擇不同的語意關係，觀察向量算術如何在幾何上呈現：
""")

# 模擬 2D 嵌入空間
words_db = {
    'king': np.array([3.0, 4.5]), 'queen': np.array([1.0, 4.5]),
    'prince': np.array([3.0, 3.0]), 'princess': np.array([1.0, 3.0]),
    'man': np.array([3.0, 1.0]), 'woman': np.array([1.0, 1.0]),
    'France': np.array([6.0, 1.0]), 'Paris': np.array([6.0, 3.5]),
    'Japan': np.array([8.0, 1.0]), 'Tokyo': np.array([8.0, 3.5]),
    'Germany': np.array([7.0, 1.0]), 'Berlin': np.array([7.0, 3.5]),
    'walk': np.array([1.0, 7.0]), 'walking': np.array([3.0, 7.0]),
    'run': np.array([1.0, 8.5]), 'running': np.array([3.0, 8.5]),
}

analogies = {
    '性別 (king-man+woman=queen)': {
        'words': ['king', 'queen', 'prince', 'princess', 'man', 'woman'],
        'formula': ('king', 'man', 'woman', 'queen'),
        'axis_label': 'gender axis', 'axis_color': '#2ecc71',
        'xlim': (-0.5, 5), 'ylim': (-0.5, 6),
    },
    '首都 (Paris-France+Japan=Tokyo)': {
        'words': ['France', 'Paris', 'Japan', 'Tokyo', 'Germany', 'Berlin'],
        'formula': ('Paris', 'France', 'Japan', 'Tokyo'),
        'axis_label': 'capital axis', 'axis_color': '#e67e22',
        'xlim': (4.5, 10), 'ylim': (-0.5, 5),
    },
    '時態 (walking-walk+run=running)': {
        'words': ['walk', 'walking', 'run', 'running'],
        'formula': ('walking', 'walk', 'run', 'running'),
        'axis_label': 'tense axis', 'axis_color': '#9b59b6',
        'xlim': (-0.5, 5), 'ylim': (5.5, 10),
    },
}

out_b = widgets.Output()

def draw_analogy(relation):
    """畫出選定的語意關係和向量算術"""
    with out_b:
        clear_output(wait=True)
        info = analogies[relation]
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        # 畫所有相關詞
        for w in info['words']:
            v = words_db[w]
            ax.scatter(v[0], v[1], s=100, zorder=5, c='#2c3e50',
                       edgecolors='white', linewidths=1.5)
            ax.annotate(w, (v[0], v[1]), textcoords="offset points",
                        xytext=(10, 10), fontsize=13, fontweight='bold')

        # 向量算術
        a_name, b_name, c_name, d_name = info['formula']
        a, b, c, d = words_db[a_name], words_db[b_name], words_db[c_name], words_db[d_name]
        predicted = a - b + c

        # 畫平行四邊形
        ax.annotate('', xy=a, xytext=b,
                    arrowprops=dict(arrowstyle='->', color=info['axis_color'], lw=2.5))
        ax.annotate('', xy=d, xytext=c,
                    arrowprops=dict(arrowstyle='->', color=info['axis_color'], lw=2.5))
        ax.annotate('', xy=a, xytext=d,
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5, linestyle='dashed'))
        ax.annotate('', xy=b, xytext=c,
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5, linestyle='dashed'))

        # 預測點
        ax.scatter(predicted[0], predicted[1], s=150, marker='x',
                   c='#e74c3c', linewidths=3, zorder=6)
        dist = np.linalg.norm(predicted - d)
        ax.annotate(f'{a_name}-{b_name}+{c_name}\n(距{d_name}: {dist:.2f})',
                    (predicted[0], predicted[1]),
                    textcoords="offset points", xytext=(-20, -25),
                    fontsize=10, color='#e74c3c', fontweight='bold')

        ax.set_title(f'向量算術: {a_name} − {b_name} + {c_name} ≈ {d_name}\n'
                     f'語意類比 = 幾何平行四邊形  |  {info["axis_label"]}',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(info['xlim'])
        ax.set_ylim(info['ylim'])
        plt.tight_layout()
        plt.show()

analogy_toggle = widgets.ToggleButtons(
    options=list(analogies.keys()),
    description='語意關係:',
    style={'description_width': 'initial', 'button_width': 'auto'},
)
analogy_toggle.observe(lambda change: draw_analogy(change['new']), names='value')
display(analogy_toggle, out_b)
draw_analogy(list(analogies.keys())[0])


# ============================================================================
# Part C: 度量學習 — 互動式 Triplet Loss 訓練
# ============================================================================
print("\n" + "=" * 60)
print("Part C: 度量學習 — 把「讓同類靠近」當作訓練目標")
print("=" * 60)

print("""
表示學習 vs 度量學習的差異：

  表示學習：
    訓練目標 = 分類 / 重建 / 預測 (cross-entropy, MSE, ...)
    幾何結構 = 副產品（為了完成目標，自然產生的）

  度量學習：
    訓練目標 = 直接優化距離函數
    幾何結構 = 訓練目標本身

  三元組損失 (Triplet Loss)：
    L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    目標：讓 d(a, p) + margin < d(a, n)
    → 同類之間的距離比異類之間的距離小至少 margin

▼ 拖動滑桿觀察訓練過程中 Triplet Loss 如何讓同類聚攏、異類遠離：
""")

np.random.seed(123)
n_per_class = 30
class_centers_init = [np.array([0, 0]), np.array([2, 3]), np.array([4, 0])]
target_centers = [np.array([-3, 0]), np.array([3, 3]), np.array([3, -3])]

init_points = []
init_labels = []
for i, center in enumerate(class_centers_init):
    pts = center + np.random.randn(n_per_class, 2) * 1.5
    init_points.append(pts)
    init_labels.extend([i] * n_per_class)
X_metric = np.vstack(init_points)
y_metric = np.array(init_labels)

# 為每個類預生成目標位置的噪音（固定隨機種子讓動畫一致）
np.random.seed(456)
target_noise = [np.random.randn(n_per_class, 2) * 0.25 for _ in range(3)]

colors_metric = ['#378ADD', '#D85A30', '#639922']
class_names = ['Class 0', 'Class 1', 'Class 2']

out_c = widgets.Output()

def draw_metric_learning(step=0):
    """step: 0 ~ 50"""
    with out_c:
        clear_output(wait=True)
        t = min(step / 40.0, 1.0)
        t_smooth = 1 - (1 - t) ** 2  # ease-out

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 左圖：資料空間
        ax = axes[0]
        all_centers = []
        for i in range(3):
            mask = y_metric == i
            target = target_centers[i] + target_noise[i]
            current = X_metric[mask] * (1 - t_smooth) + target * t_smooth
            center = current.mean(axis=0)
            all_centers.append(center)
            ax.scatter(current[:, 0], current[:, 1], c=colors_metric[i],
                       s=25, alpha=0.7, label=class_names[i])
            ax.scatter(*center, c=colors_metric[i], s=120, marker='x',
                       linewidths=2.5, zorder=5)

        # 畫類間距離線
        for i in range(3):
            for j in range(i + 1, 3):
                ci, cj = all_centers[i], all_centers[j]
                d = np.linalg.norm(ci - cj)
                mid = (ci + cj) / 2
                ax.plot([ci[0], cj[0]], [ci[1], cj[1]],
                        '--', color='gray', alpha=0.5, lw=1)
                ax.text(mid[0], mid[1], f'{d:.1f}', fontsize=9,
                        ha='center', va='bottom', color='gray')

        ax.set_title('嵌入空間', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-6, 7)
        ax.set_ylim(-6, 7)
        ax.set_aspect('equal')

        # 右圖：Loss 和距離指標
        ax2 = axes[1]
        steps_range = np.arange(0, 51)
        t_arr = np.minimum(steps_range / 40.0, 1.0)
        t_arr_smooth = 1 - (1 - t_arr) ** 2

        intra_dists = 1.5 * (1 - t_arr_smooth) + 0.25 * t_arr_smooth
        inter_dists = 3.5 * (1 - t_arr_smooth) + 6.5 * t_arr_smooth
        margin = 1.0
        losses = np.maximum(0, intra_dists - inter_dists + margin)

        ax2.plot(steps_range, losses, '-', color='#e74c3c', lw=2.5, label='Triplet Loss')
        ax2.plot(steps_range, intra_dists, '--', color='#378ADD', lw=1.5, label='類內距離 (avg)')
        ax2.plot(steps_range, inter_dists, '--', color='#D85A30', lw=1.5, label='類間距離 (avg)')
        ax2.axhline(y=0, color='gray', lw=0.5, alpha=0.5)

        # 當前步驟指示線
        ax2.axvline(x=step, color='black', lw=1, alpha=0.5, linestyle=':')
        cur_loss = max(0, (1.5 * (1 - t_smooth) + 0.25 * t_smooth) -
                       (3.5 * (1 - t_smooth) + 6.5 * t_smooth) + margin)
        ax2.scatter([step], [cur_loss], c='#e74c3c', s=80, zorder=5, edgecolors='white')

        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Value')
        ax2.set_title(f'Step {step}  |  Triplet Loss = {cur_loss:.3f}',
                      fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 50)
        ax2.set_ylim(-1, 4)

        fig.suptitle('度量學習：L = max(0, d(a,p) − d(a,n) + margin)\n'
                     '目標 → 類內距離 < 類間距離 − margin',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

slider_c = widgets.IntSlider(
    value=0, min=0, max=50, step=1,
    description='訓練步驟:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='600px'),
    continuous_update=False
)
play_c = widgets.Play(value=0, min=0, max=50, step=1, interval=200)
widgets.jslink((play_c, 'value'), (slider_c, 'value'))

slider_c.observe(lambda change: draw_metric_learning(change['new']), names='value')
display(widgets.HBox([play_c, slider_c]), out_c)
draw_metric_learning(0)


# ============================================================================
# Part D: 三種訓練範式的比較
# ============================================================================
print("\n" + "=" * 60)
print("Part D: 三種訓練範式 — 都產生表示學習，但目標不同")
print("=" * 60)

print("""
常見混淆的澄清：
  「表示學習」不是一種訓練方式，而是一種效果。
  所有深度學習的隱藏層都在做表示學習。
  差別在於「訓練目標」不同。

三種範式比較：
  ┌─────────────┬──────────────────┬──────────────────┬──────────────────┐
  │             │ 監督式分類        │ 自監督語言模型    │ 度量學習          │
  ├─────────────┼──────────────────┼──────────────────┼──────────────────┤
  │ 代表        │ ResNet, YOLO     │ GPT, BERT        │ FaceNet, CLIP    │
  │ 訓練目標    │ Cross-Entropy    │ Next Token       │ Triplet Loss     │
  │             │ 「分對類別」      │ 「預測下一個字」  │ 「讓同類靠近」    │
  │ Loss 函數   │ -Σ y·log(ŷ)     │ -log P(xₜ|x<ₜ)  │ max(0, d⁺-d⁻+m) │
  │ 需要什麼    │ 類別標籤         │ 只需文字          │ 配對/相似性標籤   │
  │ 表示學習？  │ ✓ 副產品         │ ✓ 副產品          │ ✓ 就是目標       │
  │ 遷移學習？  │ ✓ 特徵層可遷移   │ ✓ 嵌入層超通用    │ ✓ 嵌入天生通用   │
  └─────────────┴──────────────────┴──────────────────┴──────────────────┘

常見誤解的修正：
  ✗ 「LLM 是表示學習」→ 不精確。LLM 的訓練目標是自監督語言模型，
     表示學習是其副產品（但非常豐富）。
  ✗ 「圖像分類是度量學習」→ 錯。圖像分類的 Loss 是 cross-entropy，
     從來沒有計算過兩張圖的距離。度量學習的 Loss 本身就是距離函數。
  ✓ 所有深度模型的中間層都在做表示學習——這是深度架構的天然副產品。

比喻：
  表示學習就像「所有廚師切菜都會練刀工」——副產品，不是你報名學的技能。
  度量學習就像「你明確去練習判斷兩樣東西像不像」——這才是訓練目標本身。
""")


# ============================================================================
# Part E: 幾何的湧現 — 互動式觀察梯度如何雕刻空間
# ============================================================================
print("\n" + "=" * 60)
print("Part E: 幾何的湧現 — 梯度如何雕刻表示空間")
print("=" * 60)

print("""
訓練的機制（精確描述）：

  前向傳播：
    h₁ = σ(W₁x + b₁)         ← Layer 1: 線性變換 + 非線性扭曲
    h₂ = σ(W₂h₁ + b₂)        ← Layer 2: 再一次變換
    ŷ = softmax(Wₙhₙ₋₁ + bₙ)  ← 最後一層: 輸出預測

  反向傳播：
    ∂L/∂Wₙ → ... → ∂L/∂W₁    ← 每個 ∂L/∂W 告訴該層「怎麼調」

  最關鍵的洞察：
    梯度只說「讓 Loss 更小」。
    但為了讓 Loss 更小，網路必須在中間層整理出有意義的特徵結構。
    → Loss 下降是「因」，幾何結構是「果」
    → 幾何是從訓練過程中湧現 (emerge) 的

  驗證——遷移學習為什麼有效：
    ImageNet 訓練的 ResNet 中間層學到的是「視覺特徵的幾何組織」，
    不是「貓狗的知識」。所以可以遷移到醫療影像等不同領域。

▼ 拖動滑桿觀察幾何如何從隨機初始化中逐步湧現：
""")

np.random.seed(42)
n_pts_e = 80
classes_e = 3
final_centers_e = [np.array([-2.5, 2.0]), np.array([2.5, 2.0]), np.array([0.0, -2.5])]
initial_pts_e = np.random.randn(n_pts_e * classes_e, 2) * 2.5
initial_labels_e = np.repeat(range(classes_e), n_pts_e)

# 預生成目標噪音
np.random.seed(789)
target_noise_e = [np.random.randn(n_pts_e, 2) * 0.3 for _ in range(classes_e)]

color_map_e = ['#378ADD', '#D85A30', '#639922']
class_names_e = ['Class 0', 'Class 1', 'Class 2']

out_e = widgets.Output()

def draw_emergence(epoch=0):
    """epoch: 0 ~ 500"""
    with out_e:
        clear_output(wait=True)
        t = min(epoch / 500.0, 1.0)
        t_smooth = 1 - (1 - t) ** 2

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左圖：空間快照
        ax = axes[0]
        for c in range(classes_e):
            mask = initial_labels_e == c
            init = initial_pts_e[mask]
            target = final_centers_e[c] + target_noise_e[c]
            current = init * (1 - t_smooth) + target * t_smooth
            ax.scatter(current[:, 0], current[:, 1],
                       c=color_map_e[c], s=18, alpha=0.7, label=class_names_e[c])
            # 畫群心
            center = current.mean(axis=0)
            ax.scatter(*center, c=color_map_e[c], s=100, marker='x',
                       linewidths=2.5, zorder=5)

        ax.set_title(f'Epoch {epoch}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal')
        ax.legend(fontsize=9, loc='upper left')

        if epoch < 20:
            ax.text(0, -5.5, '隨機初始化 — 無幾何結構',
                    ha='center', fontsize=10, color='gray', style='italic')
        elif epoch > 400:
            ax.text(0, -5.5, '收斂 — 幾何結構完全湧現',
                    ha='center', fontsize=10, color='#27ae60', style='italic',
                    fontweight='bold')

        # 右圖：Loss 曲線
        ax2 = axes[1]
        epochs_arr = np.arange(0, 501)
        t_arr = np.minimum(epochs_arr / 500.0, 1.0)
        t_arr_smooth = 1 - (1 - t_arr) ** 2
        loss_arr = 2.5 * (1 - t_arr_smooth) ** 2 + 0.05

        ax2.plot(epochs_arr, loss_arr, '-', color='#e74c3c', lw=2.5, label='Loss')
        ax2.axvline(x=epoch, color='black', lw=1, alpha=0.5, linestyle=':')

        cur_loss = 2.5 * (1 - t_smooth) ** 2 + 0.05
        ax2.scatter([epoch], [cur_loss], c='#e74c3c', s=80, zorder=5, edgecolors='white')

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'Loss = {cur_loss:.4f}', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.set_xlim(0, 500)
        ax2.set_ylim(0, 2.8)

        fig.suptitle('幾何湧現：梯度只說「讓 Loss 更小」→ 網路自己整理出幾何結構\n'
                     '梯度從不說「讓同類靠近」，但結構是 Loss 下降的必然副產品',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.show()

slider_e = widgets.IntSlider(
    value=0, min=0, max=500, step=10,
    description='Epoch:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='600px'),
    continuous_update=False
)
play_e = widgets.Play(value=0, min=0, max=500, step=10, interval=100)
widgets.jslink((play_e, 'value'), (slider_e, 'value'))

slider_e.observe(lambda change: draw_emergence(change['new']), names='value')
display(widgets.HBox([play_e, slider_e]), out_e)
draw_emergence(0)


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結")
print("=" * 60)
print("""
從特徵工程到表示學習的核心脈絡：

  方法          誰設計新空間？     幾何意義的來源
  ──────────────────────────────────────────────────────
  特徵工程      人工手動           人的領域知識
  監督分類      神經網路自動       Loss 函數逼出來的（副產品）
  自監督 LLM    神經網路自動       語言建模逼出來的（副產品）
  度量學習      神經網路自動       距離函數直接優化（目標本身）

五個核心觀念：

  1. 每一層 σ(Wx+b) 是一次空間變換
     → W 做旋轉/縮放，σ 做非線性扭曲

  2. 訓練 = 調整 W 使得變換後的空間讓 Loss 最小
     → ∂L/∂W 告訴每個 W 該怎麼調

  3. 幾何意義是湧現的，不是設計的
     → 梯度只管 Loss，但結構自然浮現

  4. 嵌入空間保留語意的方向性
     → 不只是同類靠近，還有平行四邊形關係

  5. 所有深度模型都做表示學習，差別在訓練目標
     → 監督、自監督、度量學習只是不同的 Loss 函數

與課程其他部分的關聯：

  Phase 1-5  SVM          → 固定空間中找邊界
  Phase 1-8  特徵工程      → 人工重塑空間
  Phase 2-9  表示學習      → 網路自動重塑空間 ← 你在這裡
  Phase 4    NLP/CV       → 具體領域的表示學習應用
  Phase 5    Transformer  → 大規模表示學習的架構
""")
