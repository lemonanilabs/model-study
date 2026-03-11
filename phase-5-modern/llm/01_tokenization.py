"""
=============================================================================
Phase 5-LLM-1: Tokenization — 把文字切成 Token
=============================================================================

LLM 不是一個字一個字處理的！
  而是用「token」— 介於字元和詞之間的單位

  "tokenization" → ["token", "ization"]
  "unhappiness"  → ["un", "happiness"] 或 ["un", "happ", "iness"]

為什麼不用字/詞？
  字元：序列太長，效率差
  整詞：詞彙表太大，遇到新詞就沒辦法

本檔涵蓋：
  1. 字元級 vs 詞級 vs 子詞級
  2. BPE (Byte-Pair Encoding) — 從零實作
  3. WordPiece（BERT 用的）
  4. SentencePiece（語言無關）
  5. 實際 tokenizer 的使用
"""

import numpy as np
from collections import Counter, defaultdict
import re

# ============================================================================
# 1. Tokenization 的三種層級
# ============================================================================
print("=" * 60)
print("1. 三種層級的 Tokenization")
print("=" * 60)

text = "The cats are playing in the garden. They love playing!"

# 字元級
char_tokens = list(text)
print(f"原文: {text}")
print(f"\n字元級 ({len(char_tokens)} tokens): {char_tokens[:20]}...")

# 詞級
word_tokens = text.split()
print(f"詞級 ({len(word_tokens)} tokens): {word_tokens}")

# 子詞級（模擬）
subword_tokens = ["The", "▁cat", "s", "▁are", "▁play", "ing", "▁in",
                   "▁the", "▁garden", ".", "▁They", "▁love", "▁play", "ing", "!"]
print(f"子詞級 ({len(subword_tokens)} tokens): {subword_tokens}")

print("""
  比較：
  方法    詞彙量        序列長度    處理新詞    語義
  ─────────────────────────────────────────────────
  字元    ~100          很長        完美       差
  詞      ~100,000+     短         無法       好
  子詞    ~30,000       適中        可以       不錯

  → 子詞 (Subword) 是最佳平衡點
""")


# ============================================================================
# 2. BPE (Byte-Pair Encoding) — 從零實作
# ============================================================================
print("=" * 60)
print("2. BPE — 目前最流行的 Tokenization 方法")
print("=" * 60)

print("""
BPE 的核心想法：
  1. 從字元開始
  2. 統計哪兩個相鄰的 token 最常一起出現
  3. 把它們合併成一個新 token
  4. 重複步驟 2-3，直到達到目標詞彙量

範例：
  語料: "low lower lowest"
  初始: ['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r', ...]

  Step 1: 'l' + 'o' 最常見 → 合併為 'lo'
  Step 2: 'lo' + 'w' 最常見 → 合併為 'low'
  Step 3: 'low' + 'e' → 'lowe'
  ...

GPT-2/3/4: 用 BPE, 詞彙量 ~50,000
LLaMA: 用 BPE (SentencePiece), 詞彙量 ~32,000
""")


class SimpleBPE:
    """從零實作 BPE Tokenizer"""

    def __init__(self):
        self.merges = {}     # (pair) → merged token
        self.vocab = {}       # token → index

    def get_pairs(self, tokens_list):
        """統計所有相鄰 pair 的頻率"""
        pairs = Counter()
        for tokens in tokens_list:
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def merge_pair(self, tokens_list, pair, merged):
        """在所有 token 序列中合併指定的 pair"""
        new_list = []
        for tokens in tokens_list:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_list.append(new_tokens)
        return new_list

    def train(self, corpus, num_merges=20):
        """訓練 BPE"""
        # 初始化：每個字元是一個 token，詞尾加 '</w>'
        tokens_list = []
        for word in corpus:
            tokens_list.append(list(word) + ['</w>'])

        print(f"初始 token 序列（前 3 個詞）:")
        for t in tokens_list[:3]:
            print(f"  {t}")

        # 迭代合併
        print(f"\nBPE 合併過程（{num_merges} 次）:")
        for step in range(num_merges):
            pairs = self.get_pairs(tokens_list)
            if not pairs:
                break

            best_pair = pairs.most_common(1)[0]
            pair, count = best_pair
            merged = pair[0] + pair[1]

            self.merges[pair] = merged
            tokens_list = self.merge_pair(tokens_list, pair, merged)

            if step < 10 or step == num_merges - 1:
                print(f"  Step {step+1:2d}: "
                      f"merge ('{pair[0]}', '{pair[1]}') → '{merged}' "
                      f"(出現 {count} 次)")

        # 建立詞彙表
        all_tokens = set()
        for tokens in tokens_list:
            all_tokens.update(tokens)
        self.vocab = {t: i for i, t in enumerate(sorted(all_tokens))}

        print(f"\n最終詞彙量: {len(self.vocab)}")
        print(f"詞彙表: {sorted(self.vocab.keys())[:20]}...")

        return tokens_list

    def encode(self, word):
        """用學到的合併規則編碼新詞"""
        tokens = list(word) + ['</w>']

        for pair, merged in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens


# 訓練 BPE
corpus_words = []
train_text = """low lower lowest new newer newest wide wider widest
low low low lower lower lowest newest wider"""
for word in train_text.split():
    corpus_words.append(word)

bpe = SimpleBPE()
result = bpe.train(corpus_words, num_merges=15)

# 測試
print("\n--- 編碼測試 ---")
test_words = ['low', 'lower', 'lowest', 'newer', 'widest', 'unknown']
for word in test_words:
    tokens = bpe.encode(word)
    print(f"  '{word}' → {tokens}")

print("\n→ BPE 可以處理沒見過的詞（拆成已知的子詞）")


# ============================================================================
# 3. WordPiece（BERT 用的）
# ============================================================================
print("\n" + "=" * 60)
print("3. WordPiece — BERT 的 Tokenizer")
print("=" * 60)

print("""
WordPiece 和 BPE 很像，但合併策略不同：

  BPE:       選「出現次數最多」的 pair
  WordPiece: 選「能最大化語言模型似然度」的 pair
             近似 = 選 pair_count / (token1_count × token2_count) 最大的

  BERT 的 WordPiece 特點：
  - 用 '##' 表示「這不是詞的開頭」
  - "playing" → ["play", "##ing"]
  - "unhappiness" → ["un", "##happ", "##iness"]
  - 詞彙量: 30,522

  GPT-2 的 BPE 特點：
  - 用 byte-level（不會遇到 unknown token）
  - "playing" → ["play", "ing"]
  - 詞彙量: 50,257
""")


# ============================================================================
# 4. 實際的 Tokenizer
# ============================================================================
print("=" * 60)
print("4. Tokenizer 實務")
print("=" * 60)

print("""
主要的 Tokenizer 實作：

  1. tiktoken (OpenAI):
     import tiktoken
     enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 用的
     tokens = enc.encode("Hello world")

  2. Hugging Face tokenizers:
     from transformers import AutoTokenizer
     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
     tokens = tokenizer.encode("Hello world")

  3. SentencePiece (Google):
     import sentencepiece as spm
     sp = spm.SentencePieceProcessor()
     sp.Load("model.model")
     tokens = sp.Encode("Hello world")
""")

# 用簡單的方式模擬 tokenizer 的行為
class SimpleTokenizer:
    """教學用的簡單 tokenizer"""
    def __init__(self):
        self.vocab = {}
        self.id2token = {}

    def build_vocab(self, texts, vocab_size=100):
        """建立詞彙表"""
        # 簡單的字元+常見子詞
        chars = set()
        for text in texts:
            chars.update(text)

        tokens = sorted(chars)
        # 加入特殊 token
        special = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        tokens = special + tokens

        self.vocab = {t: i for i, t in enumerate(tokens[:vocab_size])}
        self.id2token = {i: t for t, i in self.vocab.items()}
        return self

    def encode(self, text):
        """編碼"""
        return [self.vocab.get(c, self.vocab['<UNK>']) for c in text]

    def decode(self, ids):
        """解碼"""
        return ''.join([self.id2token.get(i, '?') for i in ids])


tokenizer = SimpleTokenizer()
tokenizer.build_vocab(["Hello, world! This is a test."])

text = "Hello!"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
print(f"原文:   '{text}'")
print(f"編碼:   {encoded}")
print(f"解碼:   '{decoded}'")


# ============================================================================
# 5. Tokenization 對模型的影響
# ============================================================================
print("\n" + "=" * 60)
print("5. Tokenization 很重要！")
print("=" * 60)

print("""
Tokenization 直接影響模型效能：

  1. 詞彙量大小：
     太小 → 序列太長，效率差
     太大 → Embedding 參數太多，罕見 token 學不好
     甜蜜點 → 32K~100K

  2. 語言公平性：
     英文 "Hello" = 1 token
     中文 "你好"  = 2-3 tokens
     → 中文用同樣的 context window 能放的內容更少
     → 成本更高

  3. 數字處理：
     "1234" 可能被切成 ["12", "34"] 或 ["1", "234"]
     → LLM 不擅長數學的一個原因

  4. Token 邊界：
     "unhappiness" 切成 ["un", "happiness"] 比 ["unha", "ppiness"] 好
     → BPE 自動學到有意義的切分

  常見的特殊 Token：
    <BOS> / <s>     : 句子開頭
    <EOS> / </s>    : 句子結尾
    <PAD>           : 填充（補齊長度）
    <UNK>           : 未知詞
    <MASK>          : 遮蔽（BERT 用）
    <|endoftext|>   : GPT 的文件分隔
""")


# ============================================================================
# 小結
# ============================================================================
print("\n" + "=" * 60)
print("小結")
print("=" * 60)
print("""
Tokenization 重點整理：

  方法         使用者              詞彙量
  ──────────────────────────────────────
  BPE          GPT-2/3/4, LLaMA   32K-100K
  WordPiece    BERT, DistilBERT    30K
  SentencePiece LLaMA, T5          32K

  核心流程：
    文字 → Tokenizer → token IDs → Embedding → 模型
    模型 → logits → token ID → Tokenizer → 文字

  重要概念：
  1. Subword 是字元和詞之間的最佳平衡
  2. BPE 通過統計頻率自動學習合併規則
  3. Tokenization 會影響模型效能和公平性
  4. 特殊 token 對模型行為很重要

下一步：02_llm_landscape.py — LLM 全景
""")
