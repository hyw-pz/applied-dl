# Model Architecture

## Two-Stage Pipeline

```
Neural Signal  (B, 1, 512 channels, T timesteps)
       │
       ▼
┌─────────────────────────────┐
│     Stage 1: Acoustic Model │
│                             │
│  ┌─────────────────────┐    │
│  │  EEGConformer v1    │    │  OR
│  │  PatchEmbedding     │    │
│  │  (2D Conv → Pool)   │    │
│  │  → PositionalEnc.   │    │
│  │  → TransformerEnc.  │    │
│  │  → PhonemeHead(CTC) │    │
│  └─────────────────────┘    │
│                             │
│  ┌─────────────────────┐    │
│  │  DBConformer v2     │    │
│  │  ┌───────────────┐  │    │
│  │  │Temporal Branch│  │    │
│  │  │Stem(Conv1d    │  │    │
│  │  │+ AvgPool1d)   │  │    │
│  │  │→ TransEnc     │  │    │
│  │  └───────────────┘  │    │
│  │  ┌───────────────┐  │    │
│  │  │Spatial Branch │  │    │
│  │  │PerChannel Conv│  │    │
│  │  │→ TransEnc     │  │    │
│  │  └───────────────┘  │    │
│  │  → CTCHead          │    │
│  └─────────────────────┘    │
└─────────────────────────────┘
       │  Phoneme sequence  (e.g. "HH EH L OW")
       │  + AM uncertainty  (−mean max log-prob)
       ▼
┌─────────────────────────────┐
│     Stage 2: Language Model │
│                             │
│  BART-base / BART-large     │  OR
│  (seq2seq fine-tuned)       │
│                             │
│  Qwen2.5-7B (QLoRA)         │
│  (causal LM, ChatML fmt)    │
│                             │
│  GPT-4o-mini (few-shot)     │
└─────────────────────────────┘
       │  English text  (e.g. "Hello")
       │  + LM uncertainty  (−beam score)
       ▼
   Final output  +  Confidence Report
```

---

## EEGConformer (v1)

| Component | Detail |
|---|---|
| Input | (B, 1, 512, T) |
| Temporal Conv | Conv2d(1→emb, kernel=(1,10)) |
| Spatial Conv | Depthwise Conv2d(emb, kernel=(512,1)) |
| Pooling | AvgPool2d((1,15), stride=(1,8)) |
| Position | Sinusoidal PositionalEncoding |
| Transformer | 6× TransformerEncoderBlock (pre-LN) |
| CTC Head | LayerNorm → Linear(128→256) → GELU → Dropout → Linear(256→41) |
| Output | (T', B, 41) log-probs |

---

## DBConformer v2

| Component | Detail |
|---|---|
| Input | (B, 1, 512, T) |
| **Temporal Branch** | |
| Stem | Conv1d(512→512) + Multi-scale Conv1d + AvgPool1d(15, stride=8) |
| Pos. Embedding | Learnable, interpolated to actual sequence length |
| Transformer | 6× TransformerEncoderBlock with padding mask |
| **Spatial Branch** | |
| Embedding | Per-channel Conv1d → AdaptiveAvgPool → Linear |
| Transformer | 6× TransformerEncoderBlock |
| CTC Head | Same two-layer FFN as above, applied to temporal output only |
| Output | (T', B, 41) log-probs |

**v2 key improvements:**
- **A** — FlashAttention (`F.scaled_dot_product_attention`)
- **B** — Pre-LN architecture
- **C** — Padding mask threaded through all Transformer layers
- **D** — Two-stage pooling: Conv1d kernel=15, AvgPool1d kernel=15/stride=8
- **G** — Two-layer CTC FFN (256-dim or 1024-dim hidden)
- **H** — Dropout reduced from 0.5 → 0.3

---

## Language Models

### BART (seq2seq)

Input tokens: ARPAbet phoneme symbols as space-separated text  
e.g. `"HH EH L OW W ER L D"`

Output: English sentence  
e.g. `"Hello world"`

Training data mixing:
| Variant | GT | Synthetic | Predicted |
|---|---|---|---|
| BART-base (simple) | 50% | 0% | 50% |
| BART-base (synthetic) | 25% | 25% | 50% |
| BART-large (simple) | 50% | 0% | 50% |
| BART-large (synthetic) | 20% | 60% | 20% |

### Qwen2.5-7B QLoRA

- 4-bit NF4 quantisation (bitsandbytes)
- LoRA: r=16, α=32, target all linear projections
- ChatML prompt format:
  ```
  <|im_start|>system
  You are an expert speech decoding system…<|im_end|>
  <|im_start|>user
  Phonemes: HH EH L OW<|im_end|>
  <|im_start|>assistant
  Hello<|im_end|>
  ```

---

## Uncertainty Estimation

### Acoustic Model (AM) Uncertainty

$$u_{AM} = -\frac{1}{T'} \sum_{t=1}^{T'} \max_c \log p(c | t)$$

Higher value → model is less confident at each timestep.

### Language Model (LM) Uncertainty

For BART:
$$u_{LM} = -\text{sequence\_score (beam normalised log-prob)}$$

For Qwen:
$$u_{LM} = -\frac{1}{L} \sum_{l=1}^{L} \log p(y_l | y_{<l}, x)$$

where L is the number of generated tokens.

### Confidence Levels

| Level | LM Uncertainty Threshold | Expected WER |
|---|---|---|
| HIGH | < 0.05 | < 10% |
| MEDIUM | 0.05 – 0.15 | 10–40% |
| LOW | 0.15 – 0.25 | 40–60% |
| VERY LOW | > 0.25 | > 60% |

---

## Phoneme Vocabulary

41 classes (index 0–40):

```
0  BLANK    (CTC blank)
1  AA       (father)
2  AE       (cat)
3  AH       (cup)
4  AO       (dog)
5  AW       (how)
6  AY       (hide)
7  B
8  CH       (cheese)
9  D
10 DH       (then)
11 EH       (bed)
12 ER       (bird)
13 EY       (bait)
14 F
15 G
16 HH       (hat)
17 IH       (bit)
18 IY       (beet)
19 JH       (joy)
20 K
21 L
22 M
23 N
24 NG       (sing)
25 OW       (boat)
26 OY       (boy)
27 P
28 R
29 S
30 SH       (shoe)
31 T
32 TH       (think)
33 UH       (book)
34 UW       (boot)
35 V
36 W
37 Y        (yes)
38 Z
39 ZH       (measure)
40 |        (word boundary)
```
