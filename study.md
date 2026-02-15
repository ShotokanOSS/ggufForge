# External Logit Correction: Cost-Efficient Domain Adaptation of Quantized LLMs and MoE Models

**Authors**: ShotokanOSS (Experimental Study, February 2026)  
**Date**: February 09, 2026

## Abstract

We introduce *External Corrector*, a lightweight external adapter that refines the logits of frozen quantized large language models (LLMs) and Mixture-of-Experts (MoE) models executed via llama.cpp. The adapter processes token IDs and compressed base-model logits through a compact causal transformer, outputting a residual correction added to the original logits.

Trained for 1,000–14,000 optimizer steps on streamed domain-specific data (with gradient accumulation of 32), the adapter consistently reduces validation perplexity across diverse model families and quantization levels—from standard Q4/Q5_K_M dense models to ultra-low-bit MoE models in aggressive compression regimes. Adapters trained on smaller models transfer effectively to larger models within the same family without retraining.

The method is particularly effective in highly degraded quantization settings, with relative perplexity reductions up to ~21%. By requiring short training on small models and enabling zero-shot transfer to larger targets, it provides a highly cost-efficient mechanism for domain adaptation across a broad range of tasks and model sizes.

## 1. Introduction

Aggressive quantization enables local deployment of large models but often degrades task-specific performance. Ultra-low-bit formats push memory efficiency to extremes, allowing 20–30B-class MoE models on consumer hardware, yet at significant capability cost.

Conventional fine-tuning scales poorly with model size and is impractical for GGUF models in llama.cpp. We propose a fully external logit-correction adapter that requires no base-model modifications.

Key advantages for cost efficiency:
- Tiny adapter size.
- Short training (hours on a single GPU).
- Training on small, fast models yields transferable benefits to much larger models—dramatically reducing adaptation cost.

## 2. Related Work

Logit-space interventions and external verifiers have shown promise for frozen-model steering. Unlike internal adapters, our method is purely external and llama.cpp-native.

Recent ultra-low-bit quantization advances maximize efficiency but increase domain degradation. Our technique complements these by recovering structured errors at low cost.

## 3. Method

### 3.1 Architecture

Single-block causal transformer (embed_dim = 256 in reported experiments):

- Token embedding + logit compressor (Linear(vocab_size → embed_dim) + GELU)
- Additive fusion + LayerNorm
- 8-head causal attention
- Feed-forward (4× expansion, GELU)
- Output head to vocab_size (no bias)

Corrected logits = base_logits + corrector(token_ids, base_logits)

### 3.2 Training and Transfer

- Dataset: Streamed examples (here: reasoning domain from `prithivMLmods/Atlas-Think-Cot-12M`)
- Gradient accumulation: 32 steps
- Optimizer: AdamW (lr=5e-5)
- Optimizer steps: 1,000–14,000 (varied per experiment for convergence)
- Base model frozen

**Transfer**: Zero-shot loading of adapter weights onto larger family member with matching vocabulary.

The method is domain-agnostic and extensible to any streamed dataset.

## 4. Experiments

Hold-out validation perplexity (50 examples). Training steps (optimizer updates) were adjusted per run to reach stable convergence—lower for well-preserved Q4/Q5 models, higher for more degraded ultra-low-bit models.

## 5. Results

### 5.1 Main Results and Training Effort Comparison

| Model Family / Variant                 | Train Model (Params) | Quantization       | Training Steps (Optimizer) | Base PPL | Adapted PPL | Δ PPL | Rel. Improvement | Transfer Targets (Δ PPL / Rel.)                          |
|---------------------------------------|----------------------|--------------------|----------------------------|----------|-------------|-------|------------------|----------------------------------------------------------|
| Phi-3                                 | 3.8B                 | Q4_K_M             | 1,000                      | 2.89     | 2.76        | +0.13 | ~4.5%            | 14B 4k: +0.11 / ~4.2%<br>14B 128k: +0.10 / ~3.8%         |
| Llama-3.2                             | 1B                   | Q4_K_M             | 1,000                      | 4.37     | 4.20        | +0.17 | ~3.9%            | 3B: +0.12 / ~3.2%                                        |
| Gemma-2                               | 2B                   | Q4_K_M             | 1,000                      | 4.84     | 4.70        | +0.13 | ~2.7%            | 9B: +0.07 / ~1.6%                                        |
| Qwen3-30B-A3B-Instruct                | ~30.5B / ~3.3B active| UD-IQ1_S           | 9,000                      | 3.06     | 2.71        | +0.35 | ~11.4%           | –                                                        |
| ERNIE-4.5-21B-A3B-Thinking             | ~21B / ~3B active    | UD-Q2_K_XL         | 14,000                     | 4.39     | 3.46        | +0.93 | **~21.2%**       | –                                                        |

### 5.2 Analysis of Training Effort vs. Gains

- Dense Q4/Q5 models converged rapidly (1,000 optimizer steps), yielding consistent moderate gains (2.7–4.5% relative).
- Ultra-low-bit MoE models required more steps (9,000–14,000) due to greater initial degradation but delivered substantially larger relative improvements (11–21%).
- All training used identical hyper-parameters; step counts reflect empirical convergence needs, demonstrating the method’s adaptability across quantization regimes.
- Total runtime remained modest (hours on a single consumer GPU) even for the longest runs.

### 5.3 Transfer Performance

Adapters trained on small dense models (1B–3.8B) reliably improve larger family members with only minor degradation in relative gain:
- Phi-3: ~4.5% on 3.8B → ~4% on 14B
- Llama-3.2: ~3.9% on 1B → ~3.2% on 3B
- Gemma-2: ~2.7% on 2B → ~1.6% on 9B

Zero-shot transfer requires only vocabulary compatibility and captures scale-invariant family-specific logit biases.

## 6. Cost-Efficient Domain Adaptation

The results highlight the method’s efficiency:

1. **Low training effort on small models**  
   Strong gains in dense families with only 1,000 optimizer steps.

2. **Scalable effort for high-degradation cases**  
   More degraded models need additional steps but yield proportionally larger returns.

3. **Free transfer to large models**  
   Training cost is incurred once on the small model; benefits apply instantly to larger targets.

4. **Overall low compute footprint**  
   Entire experiments complete in hours on consumer hardware, versus days/weeks for conventional fine-tuning of large models.

This train-small-transfer-large paradigm makes targeted adaptation feasible even for resource-constrained users.

## 7. Limitations and Future Work

- Reasoning-domain proof-of-concept; broader domains pending.
- Perplexity metric; task-specific accuracy needed.
- Manual step selection; automated convergence detection possible.

Future directions:
- Multi-domain training.
- Cross-family transfer.
- Deeper or tied-weight adapters.
- Further quantization regimes.

## 8. Conclusion

External logit correction delivers robust, transferable perplexity reductions across quantization levels, with training effort appropriately scaled to initial model degradation. The ability to train cheaply on small models and transfer freely to large ones offers a highly cost-efficient path to domain adaptation of quantized LLMs—applicable to diverse tasks on consumer hardware.
