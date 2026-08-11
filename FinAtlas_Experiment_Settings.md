# FinAtlas Experiment Settings

This document records the key settings needed to reproduce the FinAtlas data construction pipeline and the experiments reported in the paper.

## 1. SFT Training

| Setting | Value | Note |
| --- | --- | --- |
| Base model | Qwen3.5-9B | https://huggingface.co/Qwen/Qwen3.5-9B |
| Training data | FinAtlas-SFT (12,500 samples; 2,500 per dimension) | 1,000 news + 1,000 tweets + 500 reports per dimension |
| Epochs | 3~10 |  |
| Batch size | 16 | per-device 8, gradient accumulation 2 |
| Learning rate | 2e-5 |  |
| Optimizer | AdamW |  |
| Sequence length | 12000 | reports need for more length |
| LoRA / full-parameter | LoRA, rank=16, alpha=32, all linear layers |  |
| Random seed | 42 |  |
| Precision | bf16 |  |
| Device | 2x A800-80GB |  |

## 2. GRPO

| Setting | Value | Note |
| --- | --- | --- |
| Initial policy | SFT model | Qwen3.5-9B after SFT training |
| Reference policy | Frozen initial policy | Qwen3.5-9B after SFT training |
| Group size `g` | 4 | Can be larger, based on device situation |
| Reward function | Format validity + label correctness; default weights: format 0.2, correctness 0.8 |  |
| KL coefficient `beta` | 0.04 |  |
| Clipping `epsilon` | 0.2 |  |
| Training steps | about 1,000 steps | Can be increased or decreased; the specific steps should be determined based on the loss |
| Sampling temperature | 0.0 | consistent with evaluation |
| Precision | bf16 |  |
| Device | 2x A800-80GB |  |

## 3. Data Pipeline Funnel
Screening thresholds (from Algorithm 1 in the paper):

- 3/3 agreement -> high confidence -> FinAtlas-SFT
- exactly 2/3 agreement -> medium confidence -> GRPO corpus
- no majority -> discarded

## 4. API Annotation Models

| Model | Temperature | Max Tokens |
| --- | --- | --- |
| DeepSeek-V4-Flash | 0.0 | 4096(tweets), 10000(news), 12000(reports) |
| Qwen3.5-Flash | 0.0 | 4096(tweets), 10000(news), 12000(reports) |
| GLM-4.7-Flash | 0.0 | 4096(tweets), 10000(news), 12000(reports) |

## 5. RQ2 Experiments

| Setting | Value | Note |
| --- | --- | --- |
| Base model | Qwen3.5-9B | https://huggingface.co/Qwen/Qwen3.5-9B |
| External datasets | Gretel (risk), FinCausal20 (causal), FiNER-ORD (entity) | Paper, Section 4.1.1 |
| Device | 1x A800-80GB |  |
| Repeated runs | 5 rounds | report mean +/- std |

