# 更新日誌 (Changelog)

> 本專案是一本循序漸進的 ML/AI 學習書。如果你已經讀到後面的章節，可以透過這份日誌快速確認前面的內容是否有更新，決定是否需要回頭複習。

---

## 2026-03-24 — 新增章節 + Notebook 清理

### 新增內容
- **Phase 1** 新增 `08_feature_engineering` — 特徵工程：四種轉換策略、準確率比較、互動式幾何視覺化
- **Phase 2** 新增 `09_representation_learning` — 表示學習：嵌入空間互動探索、向量算術、Triplet Loss 度量學習、三種訓練範式比較

### 全面清理
- **Phase 0 ~ Phase 6 所有 Notebook（42 個檔案）**：移除嵌入的執行輸出與時間戳，回到乾淨的可執行狀態。讀者重新 Run All 即可看到最新結果，也大幅縮小了檔案大小

### 維護
- 更新 Phase 1、Phase 2 的 README 加入新章節索引
- 新增 `.gitignore` 規則（`.DS_Store`、`.ipynb_checkpoints/`）
- 新增本 CHANGELOG 檔案

---

## 2026-03-13 — 修正 Matplotlib 警告

- 所有 Notebook 加入 `warnings.filterwarnings("ignore")` 抑制 Matplotlib 字體相關警告，執行時不再出現干擾訊息

## 2026-03-12 — 嵌入執行輸出與圖表

- 所有 Notebook 預先執行並嵌入輸出結果與 inline 圖表，讓讀者不需要本地執行也能閱讀

## 2026-03-11 — 新增 Phase 3 ~ Phase 6

- **Phase 3**：資料處理、GPU 加速、實驗追蹤、分散式訓練
- **Phase 4 CV**：手刻 CNN、經典架構（LeNet → ResNet）、遷移學習
- **Phase 4 NLP**：詞嵌入、RNN/LSTM、Attention 與 Seq2Seq
- **Phase 4 RL**：強化學習基礎、Q-Learning/DQN、Policy Gradient
- **Phase 5 Transformer**：Transformer 結構解析、手刻 GPT
- **Phase 5 LLM**：Tokenization、LLM 全景、LoRA 微調
- **Phase 5 Diffusion**：擴散模型原理與實作
- **Phase 6**：量化、推論加速、部署

## 2026-03-06 — 初始版本

- 建立學習路線圖與 Phase 0 ~ Phase 2 全部教材（`.py` + `.ipynb`）
