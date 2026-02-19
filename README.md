# GGUF Forge: Adapter Training & Inference Toolkit

A comprehensive toolkit for training and running lightweight adapters for GGUF-based language models (ERNIE, Llama, Mistral, Phi-3, etc.) without modifying the base model.  
Now with **two adapter architectures** (external and universal) and **automatic type detection**.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Training (`train-adapter`)](#training-train-adapter)
  - [Quick Start](#training-quick-start)
  - [Detailed Options](#training-detailed-options)
  - [Examples](#training-examples)
- [Inference (`run-inference`)](#inference-run-inference)
  - [Quick Start](#inference-quick-start)
  - [Operation Modes](#inference-operation-modes)
  - [Detailed Options](#inference-detailed-options)
  - [Examples](#inference-examples)
- [Methodology](#methodology)
  - [External Adapter](#external-adapter)
  - [Universal Adapter](#universal-adapter)
  - [Cost-Efficient Training](#cost-efficient-training)
  - [Transfer Learning](#transfer-learning)
- [Results & Performance](#results--performance)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This toolkit implements **External Logit Correction**, a novel approach for domain adaptation of quantized LLMs. Instead of fine‑tuning the entire model (which is impossible with GGUF), we train a lightweight external adapter that refines the base model's logits. The adapter is:

- **Lightweight**: Typically 256‑512 dimensions vs. billions in base model
- **Fast to train**: Hours on consumer GPU vs. days for full fine‑tuning
- **Transferable**: Adapters trained on small models work on larger family members
- **Non‑invasive**: Base model remains completely unchanged

The system now supports **two adapter architectures**:

- **External Adapter** (original): full‑vocabulary correction, ideal for models where you can afford larger adapter dimensions.
- **Universal Adapter** (cross‑model): top‑k correction using semantic embeddings, designed to be more transferable across model sizes and quantizations.

Both architectures are **auto‑detected** during loading, so you never need to specify which one you are using – the toolkit reads the configuration or inspects the state dict.

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/ShotokanOSS/ggufForge.git
cd ggufForge
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Package

**Standard Installation:**
```bash
pip install -U pip
pip install .
```

**For CUDA Support (GPU Acceleration):**
```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install .
```

**For Development with Additional Tools:**
```bash
pip install .[dev]
```

### 4. Verify Installation
```bash
train-adapter --help
run-inference --help
```

## Training (`train-adapter`)

Train lightweight adapters for GGUF models using streaming datasets.  
The training script can **automatically continue** from an existing adapter repository – just pass its Hugging Face ID as `--model`.

### Training Quick Start

**Basic training (default: universal adapter) on ERNIE model:**
```bash
train-adapter --model unsloth/ERNIE-4.5-21B-A3B-Thinking-GGUF
```

**Train external adapter on Llama 2 with custom parameters:**
```bash
train-adapter \
  --model TheBloke/Llama-2-7B-GGUF \
  --filename llama-2-7b.Q4_K_M.gguf \
  --adapter-type external \
  --adapter-dim 512 \
  --steps 5000 \
  --learning-rate 3e-5
```

**Continue training an existing adapter from Hugging Face:**
```bash
train-adapter --model my-username/my-adapter-repo --resume
```

### Training Detailed Options

#### Model Settings
| Option | Description | Default |
|--------|-------------|---------|
| `--model` | **Required.** HuggingFace repository (base model OR adapter repo) | - |
| `--base-model` | Explicit base model repository (if `--model` is an adapter) | None |
| `--filename` | Specific GGUF filename (if multiple in repo) | Auto‑detect |
| `--context-size` | Context window size | 1024 |
| `--adapter-type` | Adapter architecture: `universal` or `external` | `universal` |
| `--adapter-dim` | Adapter hidden dimension | 256 |
| `--heads` | Attention heads in adapter | 8 |
| `--top-k` | Top‑k tokens for universal adapter correction | 50 |
| `--semantic-dim` | Semantic embedding dimension (universal only) | 384 |

#### Training Parameters
| Option | Description | Default |
|--------|-------------|---------|
| `--steps` | Number of training steps | 14000 |
| `--learning-rate` | Learning rate | 5e-5 |
| `--weight-decay` | Weight decay for AdamW | 0.01 |
| `--accumulation-steps` | Gradient accumulation steps | 32 |
| `--batch-size` | Batch size | 1 |
| `--seed` | Random seed | 42 |
| `--eval-steps` | Evaluate every N steps (0 = disable) | 0 |
| `--log-file` | CSV file to log training metrics | `training_log.csv` |
| `--save-every` | Save checkpoint every N steps | 1000 |

#### Dataset Settings
| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | HuggingFace dataset ID | `prithivMLmods/Atlas-Think-Cot-12M` |
| `--prompt-col` | Column name for prompts | `problem` |
| `--output-col` | Column name for responses | `solution` |
| `--val-samples` | Validation samples | 50 |
| `--max-length` | Maximum text length | None |

#### Output & Upload
| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir` | Checkpoint directory | `checkpoints` |
| `--hf-repo` | HF repo for upload | None |
| `--hf-private` | Make HF repo private | True |
| `--no-upload` | Skip HF upload | False |
| `--hf-token` | Hugging Face token (optional) | None |

#### Special Modes
| Option | Description |
|--------|-------------|
| `--eval-only` | Only evaluate, no training |
| `--checkpoint` | Load specific checkpoint (overrides auto‑load) |
| `--resume` | Resume training from latest checkpoint in `--output-dir` |
| `--verbose` | Detailed output |
| `--gpu-layers` | Number of GPU layers for base model (-1 = all) |

### Training Examples

**1. Fast universal adapter training on dense model (Phi‑3, 1000 steps):**
```bash
train-adapter \
  --model microsoft/Phi-3.5-mini-instruct-GGUF \
  --steps 1000 \
  --learning-rate 5e-5 \
  --adapter-dim 256 \
  --adapter-type universal
```

**2. Extended external adapter training on ultra‑low‑bit MoE (ERNIE, 14000 steps):**
```bash
train-adapter \
  --model unsloth/ERNIE-4.5-21B-A3B-Thinking-GGUF \
  --filename ERNIE-4.5-21B-A3B-Thinking-UD-Q2_K_XL.gguf \
  --adapter-type external \
  --steps 14000 \
  --adapter-dim 512 \
  --output-dir ernie-external-adapter
```

**3. Custom dataset training with periodic evaluation:**
```bash
train-adapter \
  --model TheBloke/Mistral-7B-Instruct-v0.1-GGUF \
  --dataset my-org/custom-dataset \
  --prompt-col instruction \
  --output-col response \
  --eval-steps 500 \
  --log-file my_log.csv \
  --hf-repo my-org/mistral-universal-adapter
```

**4. Evaluation only:**
```bash
train-adapter \
  --model unsloth/ERNIE-4.5-21B-A3B-Thinking-GGUF \
  --eval-only \
  --checkpoint checkpoints/adapter_final.pt
```

**5. Resume interrupted training:**
```bash
train-adapter \
  --model my-org/llama-adapter \
  --resume \
  --steps 20000  # new total steps
```

## Inference (`run-inference`)

Run inference with or without adapters, compare models, or chat interactively.  
The inference script **automatically detects** the adapter type (external or universal) from the loaded weights – no extra flags needed.

### Inference Quick Start

**Single question with adapter (auto‑detects type):**
```bash
run-inference \
  --mode single \
  --question "What is machine learning?" \
  --adapter true
```

**Interactive chat:**
```bash
run-inference --mode chat
```

**Compare base vs. adapter:**
```bash
run-inference \
  --mode compare \
  --question "Explain quantum computing"
```

### Inference Operation Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `single` | Single question/answer | Quick testing |
| `chat` | Interactive conversation with history & summaries | Dialog tasks |
| `compare` | Compare base vs. adapter side by side | Performance evaluation |
| `interactive` | Full menu system (change prompts, config on the fly) | Exploration |

### Inference Detailed Options

#### Model Configuration
| Option | Description | Default |
|--------|-------------|---------|
| `--adapter-repo` | HF repository for adapter | `ShotokanJ/ERNIE-4.5-21B-A3B-Thinking-GGUF-finetune-Atlas-Think-Cot` |
| `--base-repo` | HF repository for base model | `unsloth/ERNIE-4.5-21B-A3B-Thinking-GGUF` |
| `--gguf-filename` | Specific GGUF file | `ERNIE-4.5-21B-A3B-Thinking-UD-Q2_K_XL.gguf` |
| `--adapter` | Use adapter (true/false) | true |
| `--reasoning` | Use reasoning model (think‑tags) | true |
| `--think-tags` | Enable think tags | true |
| `--summary` | Enable automatic response summaries | true |
| `--max-summary-tokens` | Max tokens for summary | 512 |

#### Generation Parameters
| Option | Description | Default |
|--------|-------------|---------|
| `--temperature` | Sampling temperature | 0.6 |
| `--min-p` | Min‑P sampling threshold | 0.05 |
| `--repetition-penalty` | Repetition penalty | 1.1 |
| `--max-tokens` | Maximum new tokens | 6100 |
| `--context-size` | Context window | 8192 |
| `--adapter-window` | Cache window for external adapter | 2048 |

*Note:* For universal adapters, the top‑k value is fixed at 50 (as used during training). This is not configurable via CLI.

#### Input/Output
| Option | Description |
|--------|-------------|
| `--question` / `-q` | Question text |
| `--file` / `-f` | Read question from file |
| `--system-prompt` / `-s` | Custom system prompt |
| `--output` / `-o` | Save response to file |
| `--verbose` / `-v` | Detailed output |
| `--no-progress` | Disable progress bar |

#### Tag Configuration (Reasoning Model)
| Option | Description | Default |
|--------|-------------|---------|
| `--think-start-tag` | Think start tag | `<think>` |
| `--think-end-tag` | Think end tag | `</think>` |
| `--final-start-tag` | Final answer start tag | `<final_answer>` |
| `--final-end-tag` | Final answer end tag | `</final_answer>` |
| `--summary-start-tag` | Summary start tag | `<Summary>` |
| `--summary-end-tag` | Summary end tag | `</Summary>` |

### Inference Examples

**1. Single question with custom parameters (adapter auto‑detected):**
```bash
run-inference \
  --mode single \
  --question "Explain the theory of relativity" \
  --temperature 0.7 \
  --max-tokens 1000 \
  --min-p 0.1 \
  --output response.txt
```

**2. Chat with custom system prompt (adapter disabled):**
```bash
run-inference \
  --mode chat \
  --system-prompt "You are a helpful physics tutor. Explain concepts simply." \
  --temperature 0.5 \
  --adapter false
```

**3. Compare with file input:**
```bash
run-inference \
  --mode compare \
  --file question.txt \
  --max-tokens 500 \
  --output comparison.json
```

**4. Interactive menu mode (full control):**
```bash
run-inference --mode interactive
```

**5. Custom model configuration (different base + adapter):**
```bash
run-inference \
  --mode single \
  --base-repo TheBloke/Llama-2-7B-GGUF \
  --gguf-filename llama-2-7b.Q4_K_M.gguf \
  --adapter-repo my-org/llama-universal-adapter \
  --context-size 4096 \
  --question "Tell me a story"
```

## Methodology

### External Adapter

The original architecture – a single‑block causal transformer that receives **token IDs** and **full base logits** as input, and outputs **correction logits** for the entire vocabulary.

```
Input: [token_ids, base_logits] → Token Embedding + Logit Compressor → Additive Fusion → LayerNorm → 
8‑Head Causal Attention → Feed‑Forward Network (4× expansion) → Output Head → Corrected Logits
```

- **Vocabulary‑aware**: operates on the full logit vector, allowing fine‑grained corrections.
- **Cache‑compatible**: can reuse KV‑cache for efficient generation.
- **Best for**: models where you can afford the extra memory of full‑vocabulary correction.

### Universal Adapter

A more transferable architecture that only corrects the **top‑k tokens** of the base logits, using **semantic embeddings** of the token candidates. This makes the adapter less dependent on the exact vocabulary and more robust to model size changes.

```
Base logits → top‑k (tokens + logits) → token IDs → semantic embeddings (via Sentence‑Transformer) →
concatenate with logits → projection → Multi‑head Self‑Attention → FFN → correction scores for top‑k tokens
```

The final logits become:

`final_logits = base_logits + scatter_add(top_k_corrections)`

- **Cross‑model transfer**: adapters trained on small models work on larger family members.
- **Memory efficient**: only processes top‑k tokens (e.g., 50) instead of the full vocabulary.
- **Best for**: large vocabularies, model families, and ultra‑low‑bit quantizations.

Both architectures are **auto‑detected** during loading: the toolkit inspects the state dict keys or the saved `config.json` and selects the correct inference path automatically.

### Cost‑Efficient Training

| Model Type | Training Steps | GPU Time | Relative Improvement |
|------------|----------------|----------|---------------------|
| Dense Q4/Q5 | 1,000 | ~1 hour | 2.7‑4.5% |
| Ultra‑low‑bit MoE | 9,000‑14,000 | ~4‑6 hours | 11‑21% |

**Why it's efficient:**
1. **Tiny parameter count**: ~1M vs. billions in base model
2. **Short training**: Hours instead of days
3. **Transfer learning**: Train small → use on large
4. **Streaming data**: No dataset download needed

### Transfer Learning

Universal adapters show remarkable transfer capabilities:

| Train On | Use On | Performance Retention |
|----------|--------|----------------------|
| Phi‑3 3.8B | Phi‑3 14B | 93% of improvement |
| Llama‑3.2 1B | Llama‑3.2 3B | 82% of improvement |
| Gemma‑2 2B | Gemma‑2 9B | 59% of improvement |

**Requirements for transfer:**
1. Same model family
2. Compatible vocabulary
3. Similar quantization scheme

## Results & Performance

### Validation Perplexity Improvements

| Model | Quantization | Base PPL | Adapted PPL | Improvement | Adapter Type | Steps |
|-------|--------------|----------|-------------|-------------|--------------|-------|
| Phi‑3 3.8B | Q4_K_M | 2.89 | 2.76 | +4.5% | Universal | 1,000 |
| Llama‑3.2 1B | Q4_K_M | 4.37 | 4.20 | +3.9% | Universal | 1,000 |
| ERNIE‑4.5‑21B | UD‑Q2_K_XL | 4.39 | 3.46 | **+21.2%** | External | 14,000 |
| Qwen3‑30B‑A3B | UD‑IQ1_S | 3.06 | 2.71 | +11.4% | Universal | 9,000 |

### Key Findings

1. **Greater degradation, greater improvement**: Ultra‑low‑bit models benefit most.
2. **Rapid convergence**: Dense models need only 1,000 steps.
3. **Universal adapter transfers**: Improvements hold across model sizes.
4. **Consistent gains**: Improvements are robust across validation sets.

## Troubleshooting

### Common Issues

**1. Model loading fails:**
```bash
# Check available files
run-inference --base-repo TheBloke/Llama-2-7B-GGUF --verbose
# Specify exact filename
run-inference --base-repo TheBloke/Llama-2-7B-GGUF --gguf-filename llama-2-7b.Q4_K_M.gguf
```

**2. Out of memory:**
```bash
# Reduce context size
run-inference --context-size 2048
# Use CPU layers
run-inference --gpu-layers 20
# For training, lower accumulation steps or adapter dimension
train-adapter --adapter-dim 128 --accumulation-steps 16
```

**3. Slow generation:**
```bash
# Reduce adapter window (external adapter only)
run-inference --adapter-window 1024
# Use base model only
run-inference --adapter false
# Disable summaries
run-inference --summary false
```

**4. Poor quality responses:**
```bash
# Adjust temperature
run-inference --temperature 0.3  # More focused
run-inference --temperature 0.9  # More creative
# Adjust min‑P
run-inference --min-p 0.01  # More diverse
run-inference --min-p 0.2   # More conservative
```

**5. Adapter type not recognized:**
The toolkit auto‑detects the type from the state dict. If it fails, ensure the adapter was saved with the correct `config.json` (including `adapter_type` field). You can also manually inspect:
```bash
python -c "import torch; sd = torch.load('adapter_final.pt', map_location='cpu'); print(sd.keys())"
```

### Debug Tips

**Enable verbose mode:**
```bash
run-inference --verbose --mode single --question "Test"
```

**Check model configuration:**
```bash
train-adapter --model unsloth/ERNIE-4.5-21B-A3B-Thinking-GGUF --eval-only --verbose
```

**Monitor GPU usage:**
```bash
nvidia-smi -l 1  # Linux
# or use --no-progress to reduce overhead during inference
run-inference --no-progress --mode chat
```

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](./LICENSE) for details.

**Key Points:**
- Free for commercial and research use
- Attribution required
- No warranty provided
- Patent rights granted

---

*For research paper, detailed methodology, and extended results, see [Study](./study.md)*
