# ML / AI 學習路線圖

> 從零開始，到能自己訓練、理解、部署模型的完整學習路徑。
> 每個階段都標註了「概念」與「程式實作」兩個面向。

---

## 總覽：學習階段

```
Phase 0  基礎數學與程式能力
   │
Phase 1  機器學習核心概念
   │
Phase 2  深度學習基礎
   │
Phase 3  模型訓練實務
   │
Phase 4  專項領域深入 (CV / NLP / Audio / RL)
   │
Phase 5  現代大模型與進階主題 (Transformer / LLM / Diffusion)
   │
Phase 6  部署與工程化
```

---

## Phase 0 — 基礎數學與程式能力

> 這些不需要精通，但需要「看到公式不害怕、能用程式驗證」的程度。

### 0.1 Python 程式基礎
- [ ] Python 語法：變數、迴圈、函式、類別
- [ ] NumPy：矩陣運算、Broadcasting、向量化操作
- [ ] Pandas：資料讀取、清洗、基本統計
- [ ] Matplotlib / Seaborn：資料視覺化

### 0.2 線性代數 (Linear Algebra)
- [ ] 向量 (Vector)、矩陣 (Matrix) 的基本運算
- [ ] 矩陣乘法的幾何意義（線性變換）
- [ ] 特徵值 (Eigenvalue) 與特徵向量 (Eigenvector)
- [ ] 奇異值分解 (SVD) — 理解降維的數學基礎
- **為什麼重要**：神經網路本質上就是一連串的矩陣運算

### 0.3 微積分 (Calculus)
- [ ] 導數 (Derivative)：函數的變化率
- [ ] 偏微分 (Partial Derivative)：多變數函數的變化率
- [ ] 鏈鎖律 (Chain Rule) — 反向傳播的數學核心
- [ ] 梯度 (Gradient)：多維空間中「最陡上升方向」
- **為什麼重要**：模型訓練 = 用梯度下降法最小化損失函數

### 0.4 機率與統計 (Probability & Statistics)
- [ ] 機率分佈：常態分佈、均勻分佈、伯努利分佈
- [ ] 條件機率與貝氏定理 (Bayes' Theorem)
- [ ] 最大似然估計 (MLE, Maximum Likelihood Estimation)
- [ ] 期望值、變異數、共變異數
- **為什麼重要**：模型的輸出通常是機率分佈；損失函數常基於 MLE

---

## Phase 1 — 機器學習核心概念

> 先理解「什麼是學習」、「模型怎麼學」，再進入深度學習。

### 1.1 核心框架：監督式學習 (Supervised Learning)
- [ ] 什麼是特徵 (Feature)、標籤 (Label)、樣本 (Sample)
- [ ] 訓練集 / 驗證集 / 測試集的切分與意義
- [ ] 損失函數 (Loss Function)：衡量預測與真實值的差距
- [ ] 梯度下降法 (Gradient Descent)：模型如何「學習」
- [ ] 過擬合 (Overfitting) vs 欠擬合 (Underfitting)
- [ ] 正則化 (Regularization)：L1 / L2 / Dropout

### 1.2 經典演算法（建立直覺）
- [ ] 線性回歸 (Linear Regression) — 最簡單的模型，理解權重與偏差
- [ ] 邏輯回歸 (Logistic Regression) — 分類問題的起點
- [ ] 決策樹 (Decision Tree) — 非線性模型的直覺
- [ ] 隨機森林 (Random Forest) — 集成學習 (Ensemble) 概念
- [ ] K-Nearest Neighbors (KNN) — 最直觀的分類方式
- [ ] Support Vector Machine (SVM) — 理解「邊界」的概念

### 1.3 非監督式學習 (Unsupervised Learning)
- [ ] K-Means 聚類
- [ ] PCA 主成分分析（降維）
- [ ] 概念：什麼時候用監督 vs 非監督

### 1.4 模型評估
- [ ] 分類指標：Accuracy / Precision / Recall / F1-Score / AUC-ROC
- [ ] 回歸指標：MSE / MAE / R²
- [ ] 交叉驗證 (Cross Validation)
- [ ] 偏差-變異數取捨 (Bias-Variance Tradeoff)

### 1.5 工具
- [ ] Scikit-learn：實作上述所有演算法
- [ ] 用真實資料集練習 (Iris, Titanic, MNIST 手寫辨識)

---

## Phase 2 — 深度學習基礎

> 從「一個神經元」開始，逐步建構到完整的神經網路。

### 2.1 神經網路的基本組成
- [ ] 感知器 (Perceptron)：一個神經元怎麼運作
- [ ] 激活函數 (Activation Function)：ReLU / Sigmoid / Tanh / Softmax
  - 為什麼需要非線性？沒有激活函數，多層網路等於一層
- [ ] 多層感知器 (MLP, Multi-Layer Perceptron)
- [ ] 前向傳播 (Forward Propagation)：輸入 → 預測
- [ ] 反向傳播 (Backpropagation)：計算梯度、更新權重
  - 這就是 Chain Rule 的直接應用

### 2.2 訓練的機制
- [ ] 損失函數選擇：
  - 回歸：MSE (Mean Squared Error)
  - 分類：Cross-Entropy Loss
- [ ] 優化器 (Optimizer)：
  - SGD → Momentum → Adam（演進脈絡）
  - Learning Rate 的重要性
- [ ] Batch / Mini-Batch / Stochastic 的差異
- [ ] Epoch 的概念

### 2.3 常見問題與技巧
- [ ] 梯度消失 (Vanishing Gradient) / 梯度爆炸 (Exploding Gradient)
- [ ] 權重初始化策略 (Xavier / He Initialization)
- [ ] Batch Normalization
- [ ] Dropout
- [ ] 學習率調度 (Learning Rate Scheduling)
- [ ] Early Stopping

### 2.4 工具
- [ ] PyTorch 基礎：Tensor / Autograd / nn.Module / DataLoader
- [ ] 或 TensorFlow/Keras（二選一即可，推薦 PyTorch）
- [ ] 用 MNIST 手寫辨識實作第一個神經網路

---

## Phase 3 — 模型訓練實務

> 從「能跑」到「跑得好」的關鍵知識。

### 3.1 資料處理
- [ ] 資料增強 (Data Augmentation)
- [ ] 特徵工程 (Feature Engineering)
- [ ] 處理不平衡資料 (Imbalanced Data)
- [ ] 資料正規化 / 標準化 (Normalization / Standardization)

### 3.2 訓練流程
- [ ] 完整的 Training Loop 手寫實作
- [ ] 驗證流程 (Validation Loop)
- [ ] 模型存儲與載入 (Checkpointing)
- [ ] TensorBoard / Weights & Biases 訓練監控
- [ ] 超參數調整 (Hyperparameter Tuning)

### 3.3 硬體與效能
- [ ] GPU 加速原理：為什麼 GPU 適合深度學習
- [ ] CUDA 基本概念
- [ ] 混合精度訓練 (Mixed Precision Training, FP16/BF16)
- [ ] 資料載入的瓶頸與多工處理 (num_workers)
- [ ] 梯度累積 (Gradient Accumulation) — 小顯卡也能訓練大模型

### 3.4 分散式訓練（進階）
- [ ] DataParallel vs DistributedDataParallel
- [ ] 模型並行 (Model Parallelism) vs 資料並行 (Data Parallelism)
- [ ] DeepSpeed / FSDP 概念

---

## Phase 4 — 專項領域深入

> 根據興趣選擇一或多個方向深入。

### 4.1 電腦視覺 (Computer Vision)
- [ ] 卷積神經網路 (CNN)：卷積層 / 池化層 / 特徵圖
- [ ] 經典架構演進：LeNet → AlexNet → VGG → ResNet → EfficientNet
- [ ] 目標偵測 (Object Detection)：YOLO / Faster R-CNN
- [ ] 語義分割 (Semantic Segmentation)：U-Net
- [ ] 圖像生成 (Image Generation)：GAN / VAE / Diffusion

### 4.2 自然語言處理 (NLP)
- [ ] 詞嵌入 (Word Embedding)：Word2Vec / GloVe
- [ ] 循環神經網路 (RNN) / LSTM / GRU
  - 為什麼 RNN 處理序列？隱藏狀態的概念
- [ ] 注意力機制 (Attention Mechanism) — 最重要的突破
- [ ] Seq2Seq 模型：Encoder-Decoder 架構
- [ ] 文字分類 / 情感分析 / 命名實體辨識 (NER)

### 4.3 語音處理 (Audio/Speech)
- [ ] 音頻特徵：Mel Spectrogram / MFCC
- [ ] 語音辨識 (ASR)：CTC / Whisper
- [ ] 語音合成 (TTS)

### 4.4 強化學習 (Reinforcement Learning)
- [ ] MDP (Markov Decision Process)
- [ ] Q-Learning / DQN
- [ ] Policy Gradient / PPO
- [ ] RLHF (Reinforcement Learning from Human Feedback)

---

## Phase 5 — 現代大模型與進階主題

> 這是 2024-2026 年的核心戰場。

### 5.1 Transformer 架構（必修）
- [ ] Self-Attention 機制：Query / Key / Value
- [ ] Multi-Head Attention
- [ ] Positional Encoding：為什麼需要位置資訊
- [ ] Encoder-Decoder 架構 (原始 Transformer)
- [ ] Layer Normalization / Residual Connection
- [ ] "Attention Is All You Need" 論文精讀

### 5.2 大語言模型 (LLM)
- [ ] GPT 系列：Decoder-Only 架構、自回歸生成
- [ ] BERT 系列：Encoder-Only、雙向理解
- [ ] T5 / BART：Encoder-Decoder
- [ ] Tokenization：BPE / SentencePiece / WordPiece
- [ ] 生成策略：Greedy / Beam Search / Top-K / Top-P / Temperature

### 5.3 大模型訓練技術
- [ ] 預訓練 (Pre-training)：語言建模目標
- [ ] 微調 (Fine-tuning)：全參數微調 vs 部分微調
- [ ] LoRA / QLoRA：低秩適應，用少量參數微調大模型
- [ ] PEFT (Parameter-Efficient Fine-Tuning)
- [ ] RLHF / DPO：對齊人類偏好
- [ ] 量化 (Quantization)：INT8 / INT4 / GPTQ / GGUF

### 5.4 Diffusion Models（圖像生成）
- [ ] 去噪擴散模型原理
- [ ] U-Net 在 Diffusion 中的角色
- [ ] Stable Diffusion 架構
- [ ] ControlNet / LoRA 在圖像生成的應用

### 5.5 多模態模型 (Multimodal)
- [ ] CLIP：圖文對齊
- [ ] Vision Transformer (ViT)
- [ ] LLaVA / GPT-4V 類型的多模態 LLM

---

## Phase 6 — 部署與工程化

### 6.1 模型部署
- [ ] ONNX：模型格式轉換
- [ ] TorchScript / TorchServe
- [ ] FastAPI / Flask 包裝推論服務
- [ ] Docker 容器化

### 6.2 推論優化
- [ ] 模型量化 (Post-Training Quantization)
- [ ] 知識蒸餾 (Knowledge Distillation)
- [ ] 模型剪枝 (Pruning)
- [ ] vLLM / TensorRT-LLM 推論加速

### 6.3 MLOps
- [ ] 實驗追蹤：MLflow / W&B
- [ ] 模型版本管理
- [ ] CI/CD for ML
- [ ] Hugging Face Hub 使用

---

## 建議的學習順序與時間分配

```
階段         建議時間    重點
─────────────────────────────────────────────
Phase 0     2-4 週     能用 NumPy 做矩陣運算，看懂梯度的意義
Phase 1     3-4 週     用 Scikit-learn 跑完經典演算法
Phase 2     3-4 週     用 PyTorch 從零寫一個神經網路
Phase 3     2-3 週     完整訓練一個有意義的模型
Phase 4     4-8 週     選一個方向深入 (推薦 NLP 或 CV)
Phase 5     持續學習    Transformer 是必修，其他依興趣
Phase 6     需要時學    實際要部署時再深入
```

---

## 本 Repo 的學習資料結構

```
Model-Study/
├── ROADMAP.md              ← 你正在看的這份路線圖
├── phase-0-foundations/    ← 數學與 Python 基礎
├── phase-1-ml-basics/      ← 機器學習核心
├── phase-2-deep-learning/  ← 深度學習基礎
├── phase-3-training/       ← 訓練實務
├── phase-4-domains/        ← 專項領域
│   ├── cv/
│   ├── nlp/
│   └── rl/
├── phase-5-modern/         ← 大模型與進階主題
│   ├── transformer/
│   ├── llm/
│   └── diffusion/
└── phase-6-deployment/     ← 部署與工程化
```

---

## 學習原則

1. **概念先行，程式驗證**：先理解「為什麼」，再用程式碼確認「怎麼做」
2. **從簡單到複雜**：每個概念都從最小可運行的例子開始
3. **手寫一次，框架用一輩子**：關鍵演算法先用 NumPy 手刻，理解後再用 PyTorch
4. **讀論文不是必須，但讀架構圖是**：至少能看懂模型的結構圖
5. **一個好的 Loss 曲線勝過千言萬語**：訓練時盯著指標看，培養直覺
