"""
=============================================================================
Phase 4-RL-3: Policy Gradient → PPO → RLHF
=============================================================================

Q-Learning 學的是「每個動作的價值」，然後選最大的。
Policy Gradient 直接學「策略本身」— 直接輸出動作的機率。

本檔涵蓋：
  1. Policy Gradient 的動機
  2. REINFORCE 演算法（從零實作）
  3. Actor-Critic
  4. PPO (Proximal Policy Optimization)
  5. RLHF — 用 RL 對齊 LLM
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ============================================================================
# 1. 為什麼需要 Policy Gradient
# ============================================================================
print("=" * 60)
print("1. Policy Gradient — 直接學策略")
print("=" * 60)

print("""
Q-Learning 的限制：
  1. 只能處理離散動作（上下左右）
  2. 連續動作怎麼辦？（機器手臂的角度 = 0~360°）
     → 不可能對每個角度都算 Q 值

Policy Gradient:
  直接學一個函數 π_θ(a|s):
  - 離散: 輸出每個動作的機率
  - 連續: 輸出動作的均值和標準差 → 從中取樣

  目標: 最大化期望回報
  J(θ) = E[Σ γ^t r_t | π_θ]

  梯度:
  ∇J(θ) = E[Σ ∇log π_θ(a_t|s_t) × G_t]

  G_t = Σ γ^k r_{t+k}  (從 t 開始的累積回報)

  直覺：
  - 如果一個動作帶來好的回報（G_t 大）→ 增加該動作的機率
  - 如果一個動作帶來差的回報（G_t 小）→ 降低該動作的機率
""")


# ============================================================================
# 2. REINFORCE — 最基本的 Policy Gradient
# ============================================================================
print("=" * 60)
print("2. REINFORCE 演算法")
print("=" * 60)

# 環境
class SimpleEnv:
    """簡單的環境：走到目標"""
    def __init__(self, size=7):
        self.size = size
        self.goal = size - 1
        self.n_states = size
        self.n_actions = 2  # 左, 右

    def reset(self):
        self.pos = self.size // 2
        return self.pos

    def step(self, action):
        if action == 0:
            self.pos = max(0, self.pos - 1)
        else:
            self.pos = min(self.size - 1, self.pos + 1)

        if self.pos == self.goal:
            return self.pos, 1.0, True
        elif self.pos == 0:
            return self.pos, -1.0, True
        else:
            return self.pos, -0.01, False


class PolicyNetwork(nn.Module):
    """策略網路"""
    def __init__(self, n_states, n_actions, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state):
        return F.softmax(self.net(state), dim=-1)

    def select_action(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


def reinforce(env, episodes=500, gamma=0.99, lr=0.01):
    """REINFORCE 演算法"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = PolicyNetwork(env.n_states, env.n_actions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    reward_history = []

    for ep in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []

        # 收集一整個 episode
        for _ in range(50):
            state_t = F.one_hot(torch.tensor(state), env.n_states).float().to(device)
            action, log_prob = policy.select_action(state_t)
            next_state, reward, done = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
            if done:
                break

        # 計算每一步的累積回報 G_t
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        # 標準化（減少方差）
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 計算 policy loss
        log_probs_t = torch.stack(log_probs)
        loss = -(log_probs_t * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reward_history.append(sum(rewards))

    return policy, reward_history


env = SimpleEnv(size=7)
print("訓練 REINFORCE...")
policy, rewards = reinforce(env, episodes=500)

window = 30
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
print(f"最後 100 episodes 平均獎勵: {np.mean(rewards[-100:]):.3f}")


# ============================================================================
# 3. Actor-Critic
# ============================================================================
print("\n" + "=" * 60)
print("3. Actor-Critic — 減少方差")
print("=" * 60)

print("""
REINFORCE 的問題：方差很大
  因為 G_t 是整個 episode 的累積回報
  一個 episode 可能很好，下一個可能很差

Actor-Critic 的解法：
  Actor:  策略網路 π_θ(a|s)   → 選動作
  Critic: 價值網路 V_φ(s)     → 評估狀態好不好

  用 Advantage 取代 G_t：
  A_t = r_t + γ V(s_{t+1}) - V(s_t)

  Actor loss:  -log π(a|s) × A_t
  Critic loss: (r + γ V(s') - V(s))²

  好處：A_t 的方差比 G_t 小很多
""")


class ActorCritic(nn.Module):
    """Actor-Critic 網路"""
    def __init__(self, n_states, n_actions, hidden=32):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(n_states, hidden), nn.ReLU())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, state):
        features = self.shared(state)
        action_probs = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return action_probs, value


def actor_critic_train(env, episodes=500, gamma=0.99, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorCritic(env.n_states, env.n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reward_history = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(50):
            state_t = F.one_hot(torch.tensor(state), env.n_states).float().to(device)
            probs, value = model(state_t)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, done = env.step(action.item())
            total_reward += reward

            next_state_t = F.one_hot(torch.tensor(next_state), env.n_states).float().to(device)
            _, next_value = model(next_state_t)

            # Advantage
            td_target = reward + gamma * next_value * (1 - done)
            advantage = td_target - value

            # Losses
            actor_loss = -dist.log_prob(action) * advantage.detach()
            critic_loss = advantage.pow(2)
            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            if done:
                break

        reward_history.append(total_reward)

    return model, reward_history


print("訓練 Actor-Critic...")
ac_model, ac_rewards = actor_critic_train(env, episodes=500)
print(f"最後 100 episodes 平均獎勵: {np.mean(ac_rewards[-100:]):.3f}")

# 比較
fig, ax = plt.subplots(figsize=(10, 4))
for name, hist in [('REINFORCE', rewards), ('Actor-Critic', ac_rewards)]:
    smoothed = np.convolve(hist, np.ones(30)/30, mode='valid')
    ax.plot(smoothed, label=name)
ax.set_xlabel('Episode')
ax.set_ylabel('Reward (MA-30)')
ax.set_title('REINFORCE vs Actor-Critic')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phase-4-domains/rl/plots/03_policy_gradient.png', dpi=100)
plt.close()
print("→ 圖表儲存至 rl/plots/03_policy_gradient.png")


# ============================================================================
# 4. PPO — 實務首選
# ============================================================================
print("\n" + "=" * 60)
print("4. PPO — Proximal Policy Optimization")
print("=" * 60)

print("""
PPO (OpenAI, 2017) 是目前最常用的 RL 演算法。

問題：Policy Gradient 每次更新可能太大 → 策略崩壞

PPO 的解法：限制每次更新的幅度

  重要性比率：
  r_t(θ) = π_θ(a_t|s_t) / π_θold(a_t|s_t)

  Clipped Objective:
  L = min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)

  直覺：
  - 如果新策略比舊策略好太多（r_t > 1+ε），clip 掉
  - 如果新策略比舊策略差太多（r_t < 1-ε），clip 掉
  - 保證每次更新都是「小步走」

PPO 的優點：
  1. 簡單好實作
  2. 穩定（不容易崩壞）
  3. 一般性強（離散/連續動作都行）
  4. 是 ChatGPT/Claude RLHF 的核心演算法
""")

# PPO 簡化實作
print("--- PPO 簡化實作 ---")


def ppo_train(env, episodes=500, gamma=0.99, lr=0.01, clip_eps=0.2, epochs_per_ep=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorCritic(env.n_states, env.n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reward_history = []

    for ep in range(episodes):
        state = env.reset()
        states, actions, rewards_list, log_probs_old, values = [], [], [], [], []

        # 收集一個 episode
        for _ in range(50):
            state_t = F.one_hot(torch.tensor(state), env.n_states).float().to(device)
            with torch.no_grad():
                probs, value = model(state_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, done = env.step(action.item())

            states.append(state_t)
            actions.append(action)
            rewards_list.append(reward)
            log_probs_old.append(dist.log_prob(action))
            values.append(value.squeeze())

            state = next_state
            if done:
                break

        # 計算 returns 和 advantages
        returns = []
        G = 0
        for r in reversed(rewards_list):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        values_t = torch.stack(values)
        advantages = returns - values_t.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs_old_t = torch.stack(log_probs_old).detach()
        states_t = torch.stack(states)
        actions_t = torch.stack(actions)

        # PPO 更新（多次 epoch）
        for _ in range(epochs_per_ep):
            probs, values_new = model(states_t)
            dist = torch.distributions.Categorical(probs)
            log_probs_new = dist.log_prob(actions_t)

            # 重要性比率
            ratio = (log_probs_new - log_probs_old_t).exp()

            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(values_new.squeeze(), returns)

            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        reward_history.append(sum(rewards_list))

    return model, reward_history


print("訓練 PPO...")
ppo_model, ppo_rewards = ppo_train(env, episodes=500)
print(f"最後 100 episodes 平均獎勵: {np.mean(ppo_rewards[-100:]):.3f}")


# ============================================================================
# 5. RLHF — 對齊 LLM
# ============================================================================
print("\n" + "=" * 60)
print("5. RLHF — 讓 AI 聽話的秘密武器")
print("=" * 60)

print("""
RLHF (Reinforcement Learning from Human Feedback)
= ChatGPT / Claude 變好用的關鍵技術

三個階段：

  Stage 1: 監督式微調 (SFT)
    用人工標註的 (問題, 好的回答) 微調 LLM
    → 模型學會回答問題的格式

  Stage 2: 訓練獎勵模型 (Reward Model)
    人工比較同一個問題的多個回答，標註哪個更好
    回答 A > 回答 B > 回答 C
    → 訓練一個模型來自動評分

  Stage 3: PPO 強化學習
    把 LLM 當 Agent:
    - State: 問題 (prompt)
    - Action: 生成回答
    - Reward: 獎勵模型的評分

    用 PPO 最大化獎勵，同時加 KL 懲罰避免偏離太遠：
    Reward = RM(answer) - β × KL(π_θ || π_ref)

  DPO (Direct Preference Optimization):
    不需要 RM 和 PPO，直接從偏好資料學
    更簡單、更穩定
    loss = -log σ(β × [log π(好回答) - log π(壞回答)])

  RLHF 的演進：
    RLHF (PPO)  →  DPO  →  KTO  →  ORPO
    越來越簡單、越來越穩定
""")


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結")
print("=" * 60)
print("""
RL 方法總覽：

  方法              動作空間    核心想法
  ──────────────────────────────────────────────
  Q-Learning/DQN   離散       學 Q(s,a)，選 max
  REINFORCE        離散/連續  ∇log π × G_t
  Actor-Critic     離散/連續  Actor + Critic 互助
  PPO              離散/連續  Clipped objective，穩定更新

  RL 和 LLM 的關係：
    RLHF 用 PPO 微調 LLM → ChatGPT
    DPO 直接用偏好資料 → 更簡單的對齊方式

  想深入 RL？推薦：
  - Gymnasium (前 OpenAI Gym): 標準 RL 環境
  - Stable Baselines3: 各種 RL 演算法的實作
  - CleanRL: 乾淨的 RL 實作（學習用）

Phase 4 RL 完成！
""")
