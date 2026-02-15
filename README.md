# GGUF Forge: Adapter Training & Inference Toolkit

A comprehensive toolkit for training and running lightweight adapters for GGUF-based language models (ERNIE, Llama, Mistral, Phi-3, etc.) without modifying the base model.

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
  - [Architecture](#architecture)
  - [Cost-Efficient Training](#cost-efficient-training)
  - [Transfer Learning](#transfer-learning)
- [Results & Performance](#results--performance)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This toolkit implements **External Logit Correction**, a novel approach for domain adaptation of quantized LLMs. Instead of fine-tuning the entire model (which is impossible with GGUF), we train a lightweight external adapter that refines the base model's logits. The adapter is:

- **Lightweight**: Typically 256-512 dimensions vs. billions in base model
- **Fast to train**: Hours on consumer GPU vs. days for full fine-tuning
- **Transferable**: Adapters trained on small models work on larger family members
- **Non-invasive**: Base model remains completely unchanged

The system supports:
- **Training** adapters on any GGUF model with streaming datasets
- **Inference** with or without adapters in multiple modes
- **Chat** with conversation history and automatic summarization
- **Comparison** between base model and adapter performance

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/ShotokanOSS/ggufForge.git
cd adapter-training
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

### Training Quick Start

**Basic training on ERNIE model:**
```bash
train-adapter --model unsloth/ERNIE-4.5-21B-A3B-Thinking-GGUF
```

**Train on Llama 2 with custom parameters:**
```bash
train-adapter \
  --model TheBloke/Llama-2-7B-GGUF \
  --filename llama-2-7b.Q4_K_M.gguf \
  --adapter-dim 512 \
  --steps 5000 \
  --learning-rate 3e-5
```

### Training Detailed Options

#### Model Settings
| Option | Description | Default |
|--------|-------------|---------|
| `--model` | **Required.** HuggingFace repository for base model | - |
| `--filename` | Specific GGUF filename (if multiple in repo) | Auto-detect |
| `--context-size` | Context window size | 1024 |
| `--adapter-dim` | Adapter hidden dimension | 256 |
| `--heads` | Attention heads in adapter | 8 |

#### Training Parameters
| Option | Description | Default |
|--------|-------------|---------|
| `--steps` | Number of training steps | 14000 |
| `--learning-rate` | Learning rate | 5e-5 |
| `--weight-decay` | Weight decay for AdamW | 0.01 |
| `--accumulation-steps` | Gradient accumulation steps | 32 |
| `--batch-size` | Batch size | 1 |
| `--seed` | Random seed | 42 |

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
| `--save-every` | Save checkpoint every N steps | 1000 |

#### Special Modes
| Option | Description |
|--------|-------------|
| `--eval-only` | Only evaluate, no training |
| `--checkpoint` | Load specific checkpoint |
| `--resume` | Resume training from latest |
| `--verbose` | Detailed output |

### Training Examples

**1. Fast training on dense model (Phi-3, 1000 steps):**
```bash
train-adapter \
  --model microsoft/Phi-3.5-mini-instruct-GGUF \
  --steps 1000 \
  --learning-rate 5e-5 \
  --adapter-dim 256
```

**2. Extended training on ultra-low-bit MoE (ERNIE, 14000 steps):**
```bash
train-adapter \
  --model unsloth/ERNIE-4.5-21B-A3B-Thinking-GGUF \
  --filename ERNIE-4.5-21B-A3B-Thinking-UD-Q2_K_XL.gguf \
  --steps 14000 \
  --adapter-dim 512 \
  --output-dir ernie-adapter
```

**3. Custom dataset training:**
```bash
train-adapter \
  --model TheBloke/Mistral-7B-Instruct-v0.1-GGUF \
  --dataset my-org/custom-dataset \
  --prompt-col instruction \
  --output-col response \
  --hf-repo my-org/mistral-adapter
```

**4. Evaluation only:**
```bash
train-adapter \
  --model unsloth/ERNIE-4.5-21B-A3B-Thinking-GGUF \
  --eval-only \
  --checkpoint checkpoints/adapter_final.pt
```

## Inference (`run-inference`)

Run inference with or without adapters, compare models, or chat interactively.

### Inference Quick Start

**Single question with adapter:**
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
| `chat` | Interactive conversation | Dialog tasks |
| `compare` | Compare base vs. adapter | Performance evaluation |
| `interactive` | Full menu system | Exploration |

### Inference Detailed Options

#### Model Configuration
| Option | Description | Default |
|--------|-------------|---------|
| `--adapter-repo` | HF repository for adapter | `ShotokanJ/ERNIE-4.5-21B-A3B-Thinking-GGUF-finetune-Atlas-Think-Cot` |
| `--base-repo` | HF repository for base model | `unsloth/ERNIE-4.5-21B-A3B-Thinking-GGUF` |
| `--gguf-filename` | Specific GGUF file | `ERNIE-4.5-21B-A3B-Thinking-UD-Q2_K_XL.gguf` |
| `--adapter` | Use adapter (true/false) | true |
| `--reasoning` | Use reasoning model | true |
| `--think-tags` | Enable think tags | true |
| `--summary` | Enable summaries | true |

#### Generation Parameters
| Option | Description | Default |
|--------|-------------|---------|
| `--temperature` | Sampling temperature | 0.6 |
| `--min-p` | Min-P sampling threshold | 0.05 |
| `--repetition-penalty` | Repetition penalty | 1.1 |
| `--max-tokens` | Maximum new tokens | 6100 |
| `--context-size` | Context window | 8192 |

#### Input/Output
| Option | Description |
|--------|-------------|
| `--question` / `-q` | Question text |
| `--file` / `-f` | Read question from file |
| `--system-prompt` / `-s` | Custom system prompt |
| `--output` / `-o` | Save response to file |
| `--verbose` / `-v` | Detailed output |

#### Tag Configuration
| Option | Description | Default |
|--------|-------------|---------|
| `--think-start-tag` | Think start tag | `<think>` |
| `--think-end-tag` | Think end tag | `</think>` |
| `--final-start-tag` | Final start tag | `<final_answer>` |
| `--final-end-tag` | Final end tag | `</final_answer>` |

### Inference Examples

**1. Single question with custom parameters:**
```bash
run-inference \
  --mode single \
  --question "Explain the theory of relativity" \
  --temperature 0.7 \
  --max-tokens 1000 \
  --min-p 0.1 \
  --output response.txt
```

**2. Chat with custom system prompt:**
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

**4. Interactive menu mode:**
```bash
run-inference --mode interactive
```

**5. Custom model configuration:**
```bash
run-inference \
  --mode single \
  --base-repo TheBloke/Llama-2-7B-GGUF \
  --gguf-filename llama-2-7b.Q4_K_M.gguf \
  --adapter-repo my-org/llama-adapter \
  --context-size 4096 \
  --question "Tell me a story"
```

## Methodology

### Architecture

The External Corrector is a single-block causal transformer:

```
Input: [token_ids, base_logits] → Token Embedding + Logit Compressor → Additive Fusion → LayerNorm → 
8-Head Causal Attention → Feed-Forward Network (4× expansion) → Output Head → Corrected Logits
```

**Mathematically:**  
`corrected_logits = base_logits + adapter(token_ids, base_logits)`

**Key Features:**
- Embedding dimension: 256-512 (adjustable)
- Vocabulary size matches base model
- Pure PyTorch, runs alongside llama.cpp
- Cache-compatible for efficient generation

### Cost-Efficient Training

| Model Type | Training Steps | GPU Time | Relative Improvement |
|------------|----------------|----------|---------------------|
| Dense Q4/Q5 | 1,000 | ~1 hour | 2.7-4.5% |
| Ultra-low-bit MoE | 9,000-14,000 | ~4-6 hours | 11-21% |

**Why it's efficient:**
1. **Tiny parameter count**: ~1M vs. billions in base model
2. **Short training**: Hours instead of days
3. **Transfer learning**: Train small → use on large
4. **Streaming data**: No dataset download needed

### Transfer Learning

Adapters show remarkable transfer capabilities:

| Train On | Use On | Performance Retention |
|----------|--------|----------------------|
| Phi-3 3.8B | Phi-3 14B | 93% of improvement |
| Llama-3.2 1B | Llama-3.2 3B | 82% of improvement |
| Gemma-2 2B | Gemma-2 9B | 59% of improvement |

**Requirements for transfer:**
1. Same model family
2. Compatible vocabulary
3. Similar quantization scheme

## Results & Performance

### Validation Perplexity Improvements

| Model | Quantization | Base PPL | Adapted PPL | Improvement | Training Steps |
|-------|--------------|----------|-------------|-------------|----------------|
| Phi-3 3.8B | Q4_K_M | 2.89 | 2.76 | +4.5% | 1,000 |
| Llama-3.2 1B | Q4_K_M | 4.37 | 4.20 | +3.9% | 1,000 |
| ERNIE-4.5-21B | UD-Q2_K_XL | 4.39 | 3.46 | **+21.2%** | 14,000 |
| Qwen3-30B-A3B | UD-IQ1_S | 3.06 | 2.71 | +11.4% | 9,000 |

### Key Findings

1. **Greater degradation, greater improvement**: Ultra-low-bit models benefit most
2. **Rapid convergence**: Dense models need only 1,000 steps
3. **Consistent gains**: Improvements hold across validation sets
4. **Zero-shot transfer**: Works without retraining on target model

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
```

**3. Slow generation:**
```bash
# Reduce adapter window
run-inference --adapter-window 1024
# Use base model only
run-inference --adapter false
```

**4. Poor quality responses:**
```bash
# Adjust temperature
run-inference --temperature 0.3  # More focused
run-inference --temperature 0.9  # More creative
# Adjust min-P
run-inference --min-p 0.01  # More diverse
run-inference --min-p 0.2   # More conservative
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
# or use --no-progress to reduce overhead
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
