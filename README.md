# Fine-Tuning-Gemma2
Fine Tuning Gemma 2 on SajjadAyoubi/persian_qa
# Parameter-Efficient Fine-Tuning of Gemma-2 (9B) for Persian Question Answering

## Overview

This repository implements a parameter-efficient fine-tuning (PEFT) pipeline for adapting the **Gemma-2 9B** decoder-only language model to extractive/generative question answering in **Persian (Farsi)**, a morphologically rich, low-resource language for which few instruction-tuned open-weight models exist. Fine-tuning is performed via **Low-Rank Adaptation (LoRA)** on top of a 4-bit quantized base model, using the [Unsloth](https://github.com/unslothai/unsloth) training framework for memory- and compute-efficient supervised fine-tuning (SFT) on a single consumer/Colab-class GPU.

The pipeline is designed to be model-agnostic within the Unsloth 4-bit model family (`unsloth/gemma-2-9b-bnb-4bit`, `unsloth/gemma-2-27b-bnb-4bit`, and comparable Llama/Mistral/Phi/DeepSeek checkpoints), and the code in this repository was validated end-to-end using `unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit` as a development/debugging checkpoint prior to the full Gemma-2 9B run; the LoRA and training configuration are identical across base models.

## Motivation

Instruction-following LLMs are predominantly trained and evaluated on English-centric corpora. Persian NLP benchmarks — including extractive QA — remain comparatively underserved, and off-the-shelf multilingual LLMs frequently underperform on Persian relative to high-resource languages due to tokenizer fragmentation, limited pretraining exposure, and scarce instruction data. This project investigates whether efficient, low-rank adaptation of a mid-scale open-weight model (Gemma-2 9B) can close part of this gap on a Persian QA task, under realistic hardware constraints (single-GPU, 4-bit quantization).

## Methodology

### 1. Base Model and Quantization
- **Backbone:** Gemma-2 9B, loaded in 4-bit (NF4) quantized form via `bitsandbytes` through Unsloth's `FastLanguageModel` API.
- **Context length:** 2,048 tokens (`max_seq_length`), with Unsloth's internal RoPE scaling support for extension if required.
- **Precision:** Automatic `bf16`/`fp16` selection based on GPU capability (`is_bfloat16_supported()`).

### 2. Parameter-Efficient Fine-Tuning (LoRA)
LoRA adapters ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) are injected into all attention and MLP projection matrices:

| Hyperparameter | Value |
|---|---|
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Rank (`r`) | 128 |
| `lora_alpha` | 128 |
| `lora_dropout` | 0 |
| Bias | none |
| Gradient checkpointing | Unsloth long-context mode |
| Random seed | 3407 |

This configuration updates a small fraction of total parameters while leaving the pretrained backbone frozen, substantially reducing the GPU memory footprint relative to full fine-tuning.

### 3. Data
- **Training format:** ShareGPT-style multi-turn template (`<|im_start|>user ... <|im_end|><|im_start|>assistant ... <|im_end|>`), with the EOS token explicitly appended to each formatted example to prevent unterminated generation at inference time.
- **Evaluation set:** [`SajjadAyoubi/persian_qa`](https://huggingface.co/datasets/SajjadAyoubi/persian_qa) (validation split), a SQuAD-style extractive Persian QA dataset with `(question, context, answer)` triples.

> **Note on training data provenance:** the formatting function is written generically over `question` / `context` / `answers` fields and is compatible with SQuAD-style Persian QA corpora (e.g. PersianQA/ParsiNLU-style datasets). Practitioners should substitute the specific training corpus used and report its size, license, and split statistics here for full reproducibility.

### 4. Training Configuration
Supervised fine-tuning is performed with TRL's `SFTTrainer`:

| Hyperparameter | Value |
|---|---|
| Per-device batch size | 2 |
| Gradient accumulation steps | 16 (effective batch size 32) |
| Max steps | 40 |
| Learning rate | 2 × 10⁻⁴ |
| LR schedule | Cosine, 5 warmup steps |
| Optimizer | 8-bit AdamW (`adamw_8bit`) |
| Weight decay | 0.05 |
| Precision | bf16 (Ampere+) / fp16 (fallback) |
| Seed | 42 |

Experiment tracking and checkpointing are handled via **Weights & Biases**, with support for resuming training from a logged W&B model artifact.

### 5. Evaluation Protocol
Model outputs are scored against gold references using three complementary metrics, reflecting standard practice in extractive/generative QA evaluation:

- **Exact Match (EM):** strict string-level match after normalization.
- **Token-level F1:** unigram overlap between predicted and reference answer spans (SQuAD-style).
- **BERTScore F1** ([Zhang et al., 2020](https://arxiv.org/abs/1904.09675)): contextual embedding similarity, providing a semantics-aware complement to surface-level EM/F1 that is more robust to valid paraphrase or partial answers.

Decoding is performed greedily (`do_sample=False`) with `top_p=0.9`, `temperature=0.7` (inactive under greedy decoding but retained in configuration for sampling-based ablations).

## Repository Structure

```
.
├── Fine-Tuning-Gemma2/
│   └── Gemma2_(9B)_Alpaca.ipynb   # End-to-end training + evaluation notebook
├── README.md
└── LICENSE
```

## Reproducing the Experiment

### Environment
```bash
pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
pip install --no-deps cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
pip install --no-deps unsloth
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

### Credentials
The notebook expects a Hugging Face access token (for gated checkpoints and/or pushing trained adapters) and a Weights & Biases API key. **Do not hardcode credentials in the notebook.** Supply them at runtime, e.g.:

```python
from huggingface_hub import login
from getpass import getpass
login(token=getpass("HF token: "))
```

```bash
wandb login   # prompts interactively; do not pass keys as CLI arguments in shared code
```

### Execution
Run the notebook top-to-bottom on a CUDA-enabled GPU (validated on Google Colab; a T4/L4/A100 class GPU is sufficient given 4-bit quantization + LoRA). The notebook performs, in order: (1) environment setup, (2) 4-bit model and tokenizer loading, (3) LoRA adapter injection, (4) data formatting, (5) SFT training with W&B logging, (6) LoRA/merged/GGUF model export, and (7) held-out evaluation with EM, token-F1, and BERTScore.

## Results

Multiple base models and hyperparameter configurations were evaluated on the Persian QA validation set to identify a stable and effective fine-tuning regime. Token-level F1 is used as the primary comparison metric across runs; Exact Match and BERTScore are reported for the best-performing configuration.

### Learning-rate sensitivity (Gemma-2 9B, 4-bit, LoRA r=128)

| Steps | Learning rate | F1 Score | Notes |
|---|---|---|---|
| 200 | default (2 × 10⁻⁴) | 0.146 | Unstable; learning rate too high for this LoRA configuration |
| 200 | 1 × 10⁻⁴ | 0.48 | |
| 200 | 1 × 10⁻⁵ | 0.49 | |
| 300 | 1 × 10⁻⁵ | 0.49 | |
| **1126** | **1 × 10⁻⁵** | **0.49** | **Best run — EM 0.25, BERTScore (F1) 0.664** |

The default learning rate (2 × 10⁻⁴) produced a substantially degraded model (F1 = 0.146), while lowering it by an order of magnitude to 1 × 10⁻⁵ recovered stable training and more than tripled F1. Extending training from 200 to 1,126 steps at this learning rate did not yield further F1 gains on this evaluation set, suggesting the model had converged well before the full training budget was used.

### Comparison across base models

| Model | Steps | Eval Samples | F1 Score |
|---|---|---|---|
| **Gemma-2-9B-bnb-4bit** (1 × 10⁻⁵, best run) | 1126 | — | **0.49** |
| Qwen2-7B | — | 20 | 0.48 |
| Qwen2-7B | — | 50 | 0.47 |
| Qwen2-7B | — | 250 | 0.46 |
| Gemma-3-4B-pt-unsloth-bnb-4bit | 70 | 20 | 0.47 |
| Qwen2.5-14B-Instruct-bnb-4bit | 70 | 20 | 0.45 |
| Qwen2.5-14B-bnb-4bit | 140 | 250 | 0.45 |

Under a matched low-resource fine-tuning budget, Gemma-2 9B achieved the highest F1 among the models evaluated, marginally ahead of Qwen2-7B and Gemma-3-4B-pt, and ahead of the larger Qwen2.5-14B variants. Qwen2-7B's F1 declined modestly as the evaluation sample size increased (0.48 → 0.46 from 20 to 250 samples), consistent with the smaller samples giving a noisier, upward-biased estimate; the 250-sample result is the more reliable figure for that model. The two Qwen2.5-14B configurations, despite the larger parameter count, did not outperform the smaller Gemma-2 9B and Gemma-3 4B models under these settings — a result worth investigating further (see Limitations).

> **Caveat:** sample sizes across configurations are small (20–250 examples) and inconsistent between models, so differences of a few F1 points between models should not be over-interpreted. The Gemma-2 9B best-run figures (EM, F1, BERTScore) reflect evaluation on the same 20-example subset described in [Reproducing the Experiment](#reproducing-the-experiment); a full-validation-split re-evaluation is recommended before treating these as final reportable numbers.

## Limitations

- Evaluation sample sizes range from 20 to 250 examples across configurations and are not held constant between models, which limits direct comparability; standard error should be reported once all configurations are re-evaluated on the full validation split.
- Token-level F1 and EM are known to be brittle to legitimate paraphrase and morphological variation in Persian; BERTScore is reported for the best configuration to partially mitigate this, but is itself sensitive to the choice of underlying encoder and layer, and was not computed for every configuration above.
- The learning-rate sweep was conducted only for Gemma-2 9B; a comparable sweep for the other base models (Qwen2-7B, Qwen2.5-14B, Gemma-3 4B) would clarify whether their lower F1 reflects the models themselves or under-tuned hyperparameters, particularly for Qwen2.5-14B, whose larger parameter count did not translate into higher F1 here.
- No comparison against full fine-tuning or alternative PEFT methods (e.g., QLoRA at lower rank, prefix-tuning) is currently included; such ablations would strengthen claims about the efficiency/performance trade-off.
- The training corpus size, licensing, and Persian-specific preprocessing (e.g., normalization of Arabic vs. Persian character variants, diacritics) are not yet documented and should be specified for reproducibility.

## License

See `LICENSE`. Gemma-2 usage is additionally governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## Acknowledgments

Built on the Unsloth fine-tuning framework and Hugging Face's `transformers`, `trl`, and `datasets` ecosystem.
