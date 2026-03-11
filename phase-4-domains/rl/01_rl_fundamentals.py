"""
=============================================================================
Phase 4-RL-1: 強化學習基礎 — MDP / 策略 / 價值函數
=============================================================================

強化學習 (RL) 和監督學習完全不同：
  監督學習：給你正確答案，照著學
  強化學習：沒有正確答案，靠「獎勵」自己摸索

  Agent 在環境中行動 → 獲得獎勵 → 學會做出更好的決策

本檔涵蓋：
  1. RL 的基本框架
  2. MDP (Markov Decision Process)
  3. 策略 (Policy) 和價值函數 (Value Function)
  4. Bellman 方程
  5. 動態規劃求解小型 MDP
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. 強化學習的框架
# ============================================================================
print("=" * 60)
print("1. 強化學習的框架")
print("=" * 60)

print("""
  ┌─────────┐   action a_t   ┌─────────────┐
  │  Agent  │ ──────────────→│ Environment │
  │ (大腦)  │                 │  (遊戲世界)  │
  │         │ ←──────────────│             │
  └─────────┘  state s_{t+1} └─────────────┘
                reward r_{t+1}

  每一步：
  1. Agent 看到 state s_t
  2. Agent 選擇 action a_t
  3. Environment 回傳：
     - 新的 state s_{t+1}
     - 獎勵 reward r_{t+1}
  4. 重複

  目標：最大化「累積獎勵」= Σ γ^t × r_t
    γ (gamma) = 折扣因子 (0~1)
    γ = 0.99: 重視長期獎勵
    γ = 0.5:  重視短期獎勵

  和監督學習的差異：
  ┌──────────────────────────────────────────────┐
  │ 監督學習:  (x, y) → 最小化 loss              │
  │ 強化學習:  (s, a, r, s') → 最大化 reward     │
  │                                               │
  │ 監督學習: 有正確答案                           │
  │ 強化學習: 只有「好不好」的信號                  │
  │                                               │
  │ 監督學習: 資料是固定的                          │
  │ 強化學習: 資料是 agent 自己產生的（邊學邊探索）  │
  └──────────────────────────────────────────────┘
""")


# ============================================================================
# 2. MDP — Markov Decision Process
# ============================================================================
print("=" * 60)
print("2. MDP — 數學框架")
print("=" * 60)

print("""
MDP 由五個元素組成：(S, A, P, R, γ)

  S: 狀態空間 (State Space)
     例：棋盤上的所有可能局面

  A: 動作空間 (Action Space)
     例：上下左右移動

  P: 轉移機率 P(s'|s, a)
     例：在狀態 s 做動作 a，到達 s' 的機率

  R: 獎勵函數 R(s, a, s')
     例：到達終點 +10，掉入陷阱 -10

  γ: 折扣因子 (discount factor)
     未來獎勵的權重

  Markov 性質：
    「未來只取決於現在，和過去無關」
    P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, s_1, ..., s_t, a_t)
""")

# 建立一個簡單的 Grid World
print("--- Grid World 範例 ---")
print("""
  簡單的 4×4 格子世界：

  ┌───┬───┬───┬───┐
  │ S │   │   │   │
  ├───┼───┼───┼───┤
  │   │ X │   │   │
  ├───┼───┼───┼───┤
  │   │   │   │ X │
  ├───┼───┼───┼───┤
  │   │   │   │ G │
  └───┴───┴───┴───┘

  S = 起點, G = 終點(+10), X = 陷阱(-5)
  每步 -1（鼓勵快速到達終點）
""")


class GridWorld:
    """簡單的 Grid World 環境"""

    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # 上下左右
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.action_names = ['上', '下', '左', '右']

        # 起點、終點、陷阱
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.traps = [(1, 1), (2, 3)]

    def state_to_idx(self, state):
        return state[0] * self.size + state[1]

    def idx_to_state(self, idx):
        return (idx // self.size, idx % self.size)

    def step(self, state, action):
        """執行動作，回傳 (next_state, reward, done)"""
        dr, dc = self.actions[action]
        nr, nc = state[0] + dr, state[1] + dc

        # 邊界檢查
        if 0 <= nr < self.size and 0 <= nc < self.size:
            next_state = (nr, nc)
        else:
            next_state = state  # 撞牆不動

        # 獎勵
        if next_state == self.goal:
            return next_state, 10.0, True
        elif next_state in self.traps:
            return next_state, -5.0, False
        else:
            return next_state, -1.0, False

    def get_transition(self, state, action):
        """取得轉移機率（確定性環境，機率=1）"""
        next_state, reward, done = self.step(state, action)
        return [(1.0, next_state, reward, done)]


env = GridWorld()
print(f"狀態空間: {env.n_states} 個狀態")
print(f"動作空間: {env.n_actions} 個動作 ({env.action_names})")

# 隨機走幾步
state = env.start
print(f"\n從 {state} 開始隨機走:")
np.random.seed(42)
for i in range(5):
    action = np.random.randint(env.n_actions)
    next_state, reward, done = env.step(state, action)
    print(f"  {state} → {env.action_names[action]} → {next_state}, reward={reward}")
    state = next_state
    if done:
        print(f"  到達終點！")
        break


# ============================================================================
# 3. 策略和價值函數
# ============================================================================
print("\n" + "=" * 60)
print("3. 策略 (Policy) 和價值函數 (Value Function)")
print("=" * 60)

print("""
策略 (Policy) π:
  π(a|s) = 在狀態 s 選擇動作 a 的機率
  確定性策略：π(s) = a（直接回傳動作）
  隨機策略：  π(a|s) = 0.7 往右, 0.3 往下

狀態價值函數 V^π(s):
  「從狀態 s 開始，跟著策略 π，期望能拿到多少累積獎勵？」
  V^π(s) = E[Σ γ^t r_{t} | s_0 = s, π]

動作價值函數 Q^π(s, a):
  「在狀態 s 做動作 a，然後跟著策略 π，期望獎勵是多少？」
  Q^π(s, a) = E[Σ γ^t r_{t} | s_0 = s, a_0 = a, π]

  Q 和 V 的關係：
  V^π(s) = Σ_a π(a|s) × Q^π(s, a)
  Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) × V^π(s')
""")


# ============================================================================
# 4. Bellman 方程和動態規劃
# ============================================================================
print("=" * 60)
print("4. Bellman 方程 — 價值函數的遞迴定義")
print("=" * 60)

print("""
Bellman 方程（RL 的核心方程式）：

  V^π(s) = Σ_a π(a|s) × [R(s,a) + γ × Σ_{s'} P(s'|s,a) × V^π(s')]

  直覺：
  當前狀態的價值 = 即時獎勵 + 折扣後的下一步價值

  最優 Bellman 方程：
  V*(s) = max_a [R(s,a) + γ × Σ_{s'} P(s'|s,a) × V*(s')]
""")


def policy_evaluation(env, policy, gamma=0.99, theta=1e-6):
    """
    策略評估：給定策略，計算每個狀態的價值

    用迭代法解 Bellman 方程
    """
    V = np.zeros(env.n_states)

    for iteration in range(1000):
        delta = 0
        for s_idx in range(env.n_states):
            state = env.idx_to_state(s_idx)
            if state == env.goal:
                continue

            v_old = V[s_idx]
            v_new = 0
            for a in range(env.n_actions):
                for prob, next_state, reward, done in env.get_transition(state, a):
                    ns_idx = env.state_to_idx(next_state)
                    v_next = 0 if done else V[ns_idx]
                    v_new += policy[s_idx, a] * prob * (reward + gamma * v_next)

            V[s_idx] = v_new
            delta = max(delta, abs(v_old - v_new))

        if delta < theta:
            break

    return V


def value_iteration(env, gamma=0.99, theta=1e-6):
    """
    價值迭代：直接找最優價值函數和最優策略

    反覆套用最優 Bellman 方程直到收斂
    """
    V = np.zeros(env.n_states)

    for iteration in range(1000):
        delta = 0
        for s_idx in range(env.n_states):
            state = env.idx_to_state(s_idx)
            if state == env.goal:
                continue

            v_old = V[s_idx]
            q_values = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                for prob, next_state, reward, done in env.get_transition(state, a):
                    ns_idx = env.state_to_idx(next_state)
                    v_next = 0 if done else V[ns_idx]
                    q_values[a] += prob * (reward + gamma * v_next)

            V[s_idx] = q_values.max()
            delta = max(delta, abs(v_old - V[s_idx]))

        if delta < theta:
            print(f"  Value Iteration 收斂於第 {iteration+1} 次迭代")
            break

    # 從 V 導出最優策略
    policy = np.zeros((env.n_states, env.n_actions))
    for s_idx in range(env.n_states):
        state = env.idx_to_state(s_idx)
        if state == env.goal:
            continue
        q_values = np.zeros(env.n_actions)
        for a in range(env.n_actions):
            for prob, next_state, reward, done in env.get_transition(state, a):
                ns_idx = env.state_to_idx(next_state)
                v_next = 0 if done else V[ns_idx]
                q_values[a] += prob * (reward + gamma * v_next)
        policy[s_idx, q_values.argmax()] = 1.0

    return V, policy


# 隨機策略的價值
random_policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
V_random = policy_evaluation(env, random_policy)

print("\n隨機策略的價值函數:")
V_grid = V_random.reshape(env.size, env.size)
for r in range(env.size):
    for c in range(env.size):
        print(f"{V_grid[r, c]:7.2f}", end="")
    print()

# 最優策略
print("\n最優策略 (Value Iteration):")
V_optimal, optimal_policy = value_iteration(env)

V_grid = V_optimal.reshape(env.size, env.size)
print("\n最優價值函數:")
for r in range(env.size):
    for c in range(env.size):
        print(f"{V_grid[r, c]:7.2f}", end="")
    print()

print("\n最優策略 (↑↓←→):")
arrows = ['↑', '↓', '←', '→']
for r in range(env.size):
    for c in range(env.size):
        s_idx = r * env.size + c
        if (r, c) == env.goal:
            print("  G  ", end="")
        elif (r, c) in env.traps:
            print("  X  ", end="")
        else:
            a = optimal_policy[s_idx].argmax()
            print(f"  {arrows[a]}  ", end="")
    print()

# 視覺化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(V_random.reshape(4, 4), cmap='RdYlGn')
axes[0].set_title('Random Policy Value')
plt.colorbar(im1, ax=axes[0])
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, f'{V_random[i*4+j]:.1f}', ha='center', va='center')

im2 = axes[1].imshow(V_optimal.reshape(4, 4), cmap='RdYlGn')
axes[1].set_title('Optimal Policy Value')
plt.colorbar(im2, ax=axes[1])
for i in range(4):
    for j in range(4):
        s_idx = i * 4 + j
        if (i, j) == env.goal:
            text = 'G'
        elif (i, j) in env.traps:
            text = 'X'
        else:
            a = optimal_policy[s_idx].argmax()
            text = f'{arrows[a]}\n{V_optimal[s_idx]:.1f}'
        axes[1].text(j, i, text, ha='center', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('phase-4-domains/rl/plots/01_grid_world.png', dpi=100)
plt.close()
print("\n→ 圖表儲存至 rl/plots/01_grid_world.png")


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結")
print("=" * 60)
print("""
強化學習基礎：

  核心概念：
  ┌──────────────────────────────────────────┐
  │ MDP = (S, A, P, R, γ)                    │
  │ Policy π(a|s): 選動作的策略               │
  │ V(s): 狀態的價值                          │
  │ Q(s,a): 狀態-動作的價值                   │
  │ Bellman: V(s) = R + γ V(s')              │
  └──────────────────────────────────────────┘

  求解方法：
    已知環境(P,R) → 動態規劃 (Policy/Value Iteration)
    未知環境      → 學習 (Q-Learning, Policy Gradient)

  實際問題中環境通常未知 → 需要 Q-Learning 或 Policy Gradient

下一步：02_q_learning_dqn.py — Q-Learning 和 DQN
""")
