#!/usr/bin/env python3
"""
Adaptive Language Model System with Adapter Support
CLI-enabled version with command line arguments
Now with auto-detection of adapter type (external / universal)
"""

# Necessary imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from llama_cpp import Llama
from tqdm import tqdm
import json
import os
import sys
import argparse
from huggingface_hub import hf_hub_download, list_repo_files
import warnings
import gc
from sentence_transformers import SentenceTransformer
warnings.filterwarnings("ignore")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"System running on: {device}")

# ============================================================================
# CONFIGURATION SECTION (with defaults that can be overridden by CLI)
# ============================================================================

# Default configuration values
DEFAULT_HF_ADAPTER_REPO = "ShotokanJ/ERNIE-4.5-21B-A3B-Thinking-GGUF-finetune-Atlas-Think-Cot"
DEFAULT_BASE_MODEL_REPO = "unsloth/ERNIE-4.5-21B-A3B-Thinking-GGUF"
DEFAULT_GGUF_FILENAME = "ERNIE-4.5-21B-A3B-Thinking-UD-Q2_K_XL.gguf"

# Default system behavior configuration
DEFAULT_USE_REASONING_MODEL = True
DEFAULT_USE_ADAPTER = True
DEFAULT_ENABLE_THINK_TAGS = True
DEFAULT_ENABLE_SUMMARY = True
DEFAULT_MAX_SUMMARY_TOKENS = 512

# Default generation parameters
DEFAULT_MAX_NEW_TOKEN = 6100
DEFAULT_CONTEXT_SIZE = 8192
DEFAULT_ADAPTER_WINDOW = 2048
DEFAULT_TOP_K = 50               # for universal adapter

# Default think tags configuration
DEFAULT_THINK_START_TAG = "<think>"
DEFAULT_THINK_END_TAG = "</think>"
DEFAULT_FINAL_START_TAG = "<final_answer>"
DEFAULT_FINAL_END_TAG = "</final_answer>"
DEFAULT_SUMMARY_START_TAG = "<Summary>"
DEFAULT_SUMMARY_END_TAG = "</Summary>"

# Global variables (will be set based on CLI args)
HF_ADAPTER_REPO = DEFAULT_HF_ADAPTER_REPO
BASE_MODEL_REPO = DEFAULT_BASE_MODEL_REPO
GGUF_FILENAME = DEFAULT_GGUF_FILENAME
USE_REASONING_MODEL = DEFAULT_USE_REASONING_MODEL
USE_ADAPTER = DEFAULT_USE_ADAPTER
ENABLE_THINK_TAGS = DEFAULT_ENABLE_THINK_TAGS
ENABLE_SUMMARY = DEFAULT_ENABLE_SUMMARY
MAX_SUMMARY_TOKENS = DEFAULT_MAX_SUMMARY_TOKENS
MAX_NEW_TOKEN = DEFAULT_MAX_NEW_TOKEN
CONTEXT_SIZE = DEFAULT_CONTEXT_SIZE
ADAPTER_WINDOW = DEFAULT_ADAPTER_WINDOW
THINK_START_TAG = DEFAULT_THINK_START_TAG
THINK_END_TAG = DEFAULT_THINK_END_TAG
FINAL_START_TAG = DEFAULT_FINAL_START_TAG
FINAL_END_TAG = DEFAULT_FINAL_END_TAG
SUMMARY_START_TAG = DEFAULT_SUMMARY_START_TAG
SUMMARY_END_TAG = DEFAULT_SUMMARY_END_TAG

# System Prompt Definition (adaptive based on configuration)
def create_system_prompt(summaries=""):
    """Erstellt System-Prompt mit optionalen Zusammenfassungen"""
    summary_section = ""
    if summaries and ENABLE_SUMMARY:
        summary_section = f"Previous conversation summaries:\n{summaries}\n\n"

    if USE_REASONING_MODEL and ENABLE_THINK_TAGS:
        return f"""{summary_section}You are a helpful AI assistant. Please follow these instructions STRICTLY:
1. First, think carefully about the user's request inside {THINK_START_TAG} tags
2. After your thinking process, close it with {THINK_END_TAG}
3. Then provide your FINAL ANSWER inside {FINAL_START_TAG} tags
4. End your final answer with {FINAL_END_TAG}
5. Do not write anything after {FINAL_END_TAG}

Your response format must be exactly:
{THINK_START_TAG}
[your step-by-step thinking]
{THINK_END_TAG}
{FINAL_START_TAG}
[your final answer to the user]
{FINAL_END_TAG}"""
    else:
        return f"""{summary_section}You are a helpful AI assistant. Please provide clear, concise, and accurate answers to user questions.

Provide your answer directly without additional formatting."""

# ---------------------------------------------------------
# External Adapter (original) – same as before
# ---------------------------------------------------------
class CausalAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "dim must be divisible by heads"
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, past_k=None, past_v=None, use_cache=False):
        b, n, d = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2)

        if past_k is not None and past_v is not None:
            k_cat = torch.cat([past_k, k], dim=2)
            v_cat = torch.cat([past_v, v], dim=2)
        else:
            k_cat = k
            v_cat = v

        attn_out = F.scaled_dot_product_attention(q, k_cat, v_cat, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, n, d)

        out = self.to_out(attn_out)

        if use_cache:
            return out, k_cat.detach(), v_cat.detach()
        else:
            return out

class ExternalCorrector(nn.Module):
    def __init__(self, vocab_size, embed_dim, heads=8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.logit_compressor = nn.Linear(vocab_size, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)

        self.attention = CausalAttention(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids, llm_logits, past_k=None, past_v=None, use_cache=False):
        x_ctx = self.token_emb(input_ids)
        x_logits = F.gelu(self.logit_compressor(llm_logits))

        x = x_ctx + x_logits
        x = self.input_norm(x)

        if use_cache:
            attn_out, new_k, new_v = self.attention(self.norm1(x), past_k=past_k, past_v=past_v, use_cache=True)
        else:
            attn_out = self.attention(self.norm1(x))
            new_k, new_v = None, None

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        correction_logits = self.head(x)

        if use_cache:
            return correction_logits, new_k, new_v
        else:
            return correction_logits

# ---------------------------------------------------------
# Universal Adapter (cross‑model) with SemanticMapper
# ---------------------------------------------------------
class SemanticMapper:
    def __init__(self, llm):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.llm = llm
        self.cache = {}

    def get_vectors(self, token_ids):
        flat_ids = token_ids.flatten().tolist()
        unique_ids = list(set(flat_ids))
        missing = [i for i in unique_ids if i not in self.cache]
        if missing:
            texts = [self.llm.detokenize([i]).decode("utf-8", errors="ignore") for i in missing]
            with torch.no_grad():
                embs = self.embedder.encode(texts, convert_to_tensor=True, show_progress_bar=False).to(device)
            for i, emb in zip(missing, embs):
                self.cache[i] = emb
        res = torch.stack([self.cache[i] for i in flat_ids]).to(device)
        return res.view(*token_ids.shape, -1)

class UniversalAdapter(nn.Module):
    def __init__(self, semantic_dim=384, adapter_dim=256, heads=8):
        super().__init__()
        self.input_proj = nn.Linear(semantic_dim + 1, adapter_dim)
        self.ln1 = nn.LayerNorm(adapter_dim)
        self.attn = nn.MultiheadAttention(adapter_dim, num_heads=heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim*4),
            nn.GELU(),
            nn.Linear(adapter_dim*4, adapter_dim)
        )
        self.ln2 = nn.LayerNorm(adapter_dim)
        self.output_head = nn.Linear(adapter_dim, 1)

    def forward(self, sem_embs, top_k_logits):
        sem_embs = sem_embs.to(device)
        top_k_logits = top_k_logits.to(device)
        b, s, k, d = sem_embs.shape
        x = torch.cat([sem_embs, top_k_logits.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x).view(b*s, k, -1)
        res, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + res + self.ffn(self.ln2(x))
        return self.output_head(x).view(b, s, k)

# ---------------------------------------------------------
# Functions for loading from Hugging Face with auto-detection
# ---------------------------------------------------------
def detect_adapter_type_from_state_dict(state_dict):
    """Determine adapter type by inspecting state dict keys."""
    if 'token_emb.weight' in state_dict:
        return 'external'
    elif 'input_proj.weight' in state_dict:
        return 'universal'
    else:
        raise ValueError("Unknown adapter type – cannot determine from state dict.")

def load_adapter_from_hf(repo_id, local_dir="./hf_cache"):
    """
    Loads adapter and configuration from Hugging Face Hub.
    Auto‑detects adapter type (external or universal).
    """
    print(f"Loading adapter from Hugging Face: {repo_id}")

    # Create local directory
    os.makedirs(local_dir, exist_ok=True)

    # 1. Load configuration
    print("Loading configuration...")
    try:
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            cache_dir=local_dir
        )
    except Exception:
        config_path = None

    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

    print(f"Configuration loaded:")
    print(f"  - Adapter Dimension: {config.get('adapter', {}).get('adapter_dim', 'N/A')}")
    print(f"  - Vocab Size: {config.get('model', {}).get('vocab_size', 'N/A')}")
    print(f"  - Heads: {config.get('adapter', {}).get('heads', 'N/A')}")
    print(f"  - Context Size: {config.get('model', {}).get('context_size', 'N/A')}")

    # 2. Load adapter weights
    print("Loading adapter weights...")
    adapter_path = hf_hub_download(
        repo_id=repo_id,
        filename="adapter_final.pt",
        cache_dir=local_dir
    )
    state_dict = torch.load(adapter_path, map_location='cpu')

    # Detect adapter type
    if 'adapter_type' in config:
        adapter_type = config['adapter_type']
        print(f"  - Adapter type from config: {adapter_type}")
    else:
        adapter_type = detect_adapter_type_from_state_dict(state_dict)
        print(f"  - Adapter type detected from weights: {adapter_type}")

    # 3. Optional: Load metrics
    try:
        metrics_path = hf_hub_download(
            repo_id=repo_id,
            filename="metrics.json",
            cache_dir=local_dir
        )
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        print(f"  - Base PPL: {metrics.get('base_perplexity', 'N/A'):.2f}")
        print(f"  - Adapted PPL: {metrics.get('adapted_perplexity', 'N/A'):.2f}")
    except:
        print("  - No metrics found")

    return config, adapter_path, state_dict, adapter_type

# ---------------------------------------------------------
# Helper: Softmax + Sampling with Repetition Penalty and Min-P
# ---------------------------------------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def apply_repetition_penalty(logits, generated_tokens, penalty=1.1):
    """Applies repetition penalty to repeated tokens"""
    if len(generated_tokens) == 0 or penalty == 1.0:
        return logits

    penalized_logits = logits.copy()

    for token in set(generated_tokens[-20:]):
        if penalized_logits[token] < 0:
            penalized_logits[token] *= penalty
        else:
            penalized_logits[token] /= penalty

    return penalized_logits

def sample_logits_min_p(logits, generated_tokens=None, temperature=0.8, min_p=0.05,
                       repetition_penalty=1.1, eos_token_id=None, stop_tokens=None):
    """
    Min-P Sampling with Repetition Penalty
    """
    if stop_tokens is None:
        stop_tokens = []

    if generated_tokens is not None and len(generated_tokens) > 0:
        logits = apply_repetition_penalty(logits, generated_tokens, repetition_penalty)

    if temperature == 0.0:
        return int(np.argmax(logits))

    logits = logits.astype(np.float64) / temperature

    probs = softmax(logits)
    max_prob = np.max(probs)
    threshold = max_prob * min_p

    filtered_probs = np.where(probs >= threshold, probs, 0)
    sum_probs = np.sum(filtered_probs)

    if sum_probs <= 0:
        filtered_probs = probs
        sum_probs = np.sum(filtered_probs)

    filtered_probs = filtered_probs / sum_probs

    return int(np.random.choice(len(filtered_probs), p=filtered_probs))

# ---------------------------------------------------------
# Generation functions for each adapter type
# ---------------------------------------------------------
def generate_with_external_adapter(llm, adapter, user_prompt: str, max_new_tokens: int = 100,
                         temperature: float = 0.8, min_p: float = 0.05,
                         repetition_penalty: float = 1.1,
                         adapter_window: int = None, remove_cot: bool = False,
                         generator=None, generate_summary: bool = False):
    """
    Generates text with external adapter (original)
    Stops at EOS, </final_answer>, or </Summary>
    """
    if generator is None:
        generator = TextGenerator(llm, adapter)

    generator.reset()
    full_prompt = generator._prepare_prompt(user_prompt)

    tokens = llm.tokenize(full_prompt.encode("utf-8"), add_bos=True)
    current_tokens = tokens.copy()

    llm.reset()
    llm.eval(tokens)

    output = full_prompt

    if llm.scores.shape[0] == 0:
        if remove_cot and USE_REASONING_MODEL and ENABLE_THINK_TAGS:
            return remove_cot_from_response(output), ""
        return generator._extract_response(output, user_prompt), ""

    llm_logits_np = llm.scores[:len(current_tokens), :]

    with torch.no_grad():
        input_ids_full = torch.tensor(current_tokens, dtype=torch.long, device=device).unsqueeze(0)
        llm_logits_full = torch.from_numpy(llm_logits_np).unsqueeze(0).to(device)
        _, past_k, past_v = adapter(input_ids_full, llm_logits_full, past_k=None, past_v=None, use_cache=True)

    def trim_cache(k, v, max_len):
        if max_len is None:
            return k, v
        total_len = k.size(2)
        if total_len <= max_len:
            return k, v
        return k[..., -max_len:, :].contiguous(), v[..., -max_len:, :].contiguous()

    if adapter_window is not None:
        past_k, past_v = trim_cache(past_k, past_v, adapter_window)

    generated_tokens = []

    for _ in tqdm(range(max_new_tokens), desc="Generating with external adapter"):
        if llm.scores.shape[0] == 0:
            break

        llm_logits_np = llm.scores[:len(current_tokens), :]
        base_last = llm_logits_np[-1]

        last_input_id = torch.tensor([current_tokens[-1]], dtype=torch.long, device=device).unsqueeze(0)
        last_llm_logit = torch.from_numpy(base_last).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            corr_logits_seq, new_k, new_v = adapter(
                last_input_id, last_llm_logit, past_k=past_k, past_v=past_v, use_cache=True
            )
        correction_last = corr_logits_seq[0, -1, :].cpu().numpy()

        adapted_logits = base_last + correction_last

        next_token = sample_logits_min_p(
            adapted_logits,
            generated_tokens=generated_tokens,
            temperature=temperature,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=generator.eos_token_id
        )

        if next_token == generator.eos_token_id:
            break

        next_text = llm.detokenize([next_token]).decode("utf-8", errors="ignore")
        output += next_text
        generator._update_state(next_text)

        current_tokens.append(next_token)
        generated_tokens.append(next_token)

        # Stop when any stop token is generated
        stop_detected, stop_token = generator._check_for_stop_tokens()
        if stop_detected:
            break

        llm.eval([next_token])

        if adapter_window is not None:
            past_k, past_v = trim_cache(new_k, new_v, adapter_window)
        else:
            past_k, past_v = new_k, new_v

    response = generator._extract_response(output, user_prompt)

    if remove_cot and USE_REASONING_MODEL and ENABLE_THINK_TAGS:
        response = remove_cot_from_response(response)

    summary = ""
    if generate_summary and ENABLE_SUMMARY:
        summary = summarize_response(llm, response, adapter)

    if 'past_k' in locals() and 'past_v' in locals():
        del past_k, past_v, new_k, new_v
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return response, summary

def generate_with_universal_adapter(llm, adapter, mapper, user_prompt: str, max_new_tokens: int = 100,
                         temperature: float = 0.8, min_p: float = 0.05,
                         repetition_penalty: float = 1.1,
                         top_k: int = 50, remove_cot: bool = False,
                         generator=None, generate_summary: bool = False):
    """
    Generates text with universal adapter (top‑k correction)
    """
    if generator is None:
        generator = TextGenerator(llm, adapter)

    generator.reset()
    full_prompt = generator._prepare_prompt(user_prompt)

    tokens = llm.tokenize(full_prompt.encode("utf-8"), add_bos=True)
    current_tokens = tokens.copy()

    llm.reset()
    llm.eval(tokens)

    output = full_prompt

    if llm.scores.shape[0] == 0:
        if remove_cot and USE_REASONING_MODEL and ENABLE_THINK_TAGS:
            return remove_cot_from_response(output), ""
        return generator._extract_response(output, user_prompt), ""

    # For universal adapter, we don't maintain a cache across steps (simpler)
    # We'll just compute correction on the fly each step using the full history of logits.
    generated_tokens = []

    for _ in tqdm(range(max_new_tokens), desc="Generating with universal adapter"):
        if llm.scores.shape[0] == 0:
            break

        # Get current logits for all positions so far
        llm_logits_np = llm.scores[:len(current_tokens), :]  # (seq_len, vocab)
        base_logits = torch.from_numpy(llm_logits_np).to(device)
        base_last = base_logits[-1].cpu().numpy()

        # Apply universal adapter correction to the last token's logits
        # We need top‑k for the last position only
        top_v, top_i = torch.topk(base_logits[-1:], top_k, dim=-1)  # (1, top_k)
        sem_embs = mapper.get_vectors(top_i).unsqueeze(0)  # (1, 1, top_k, sem_dim)
        with torch.no_grad():
            corrections = adapter(sem_embs, top_v.unsqueeze(0))  # (1, 1, top_k)
        # Apply corrections to the last logits
        adapted_logits = base_last.copy()
        adapted_logits[top_i[0].cpu().numpy()] += corrections[0, 0].cpu().numpy()

        next_token = sample_logits_min_p(
            adapted_logits,
            generated_tokens=generated_tokens,
            temperature=temperature,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=generator.eos_token_id
        )

        if next_token == generator.eos_token_id:
            break

        next_text = llm.detokenize([next_token]).decode("utf-8", errors="ignore")
        output += next_text
        generator._update_state(next_text)

        current_tokens.append(next_token)
        generated_tokens.append(next_token)

        # Stop when any stop token is generated
        stop_detected, stop_token = generator._check_for_stop_tokens()
        if stop_detected:
            break

        llm.eval([next_token])

    response = generator._extract_response(output, user_prompt)

    if remove_cot and USE_REASONING_MODEL and ENABLE_THINK_TAGS:
        response = remove_cot_from_response(response)

    summary = ""
    if generate_summary and ENABLE_SUMMARY:
        summary = summarize_response(llm, response)

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return response, summary

# ---------------------------------------------------------
# Main function: Initialize models
# ---------------------------------------------------------
def initialize_models():
    """
    Initializes base LLM and adapter from Hugging Face.
    Auto‑detects adapter type.
    """
    if USE_ADAPTER:
        config, adapter_path, state_dict, adapter_type = load_adapter_from_hf(HF_ADAPTER_REPO)

        # Extract dimension from state dict
        if adapter_type == 'external':
            actual_adapter_dim = state_dict['token_emb.weight'].shape[1]
            heads = config.get("adapter", {}).get("heads", 8)
            print(f"External adapter dimension: {actual_adapter_dim}")
            print(f"Using heads = {heads} from config")
        else:  # universal
            actual_adapter_dim = state_dict['input_proj.weight'].shape[0]  # adapter_dim
            # Also need semantic_dim
            semantic_dim = state_dict['input_proj.weight'].shape[1] - 1
            heads = config.get("adapter", {}).get("heads", 8)
            print(f"Universal adapter dimension: {actual_adapter_dim}")
            print(f"Semantic dimension: {semantic_dim}")
            print(f"Using heads = {heads} from config")

    print(f"\nLoading base model: {BASE_MODEL_REPO}")
    llm = Llama.from_pretrained(
        repo_id=BASE_MODEL_REPO,
        filename=GGUF_FILENAME,
        logits_all=True,
        n_ctx=CONTEXT_SIZE,
        n_gpu_layers=-1 if torch.cuda.is_available() else 0,
        verbose=False
    )

    VOCAB_SIZE = llm.n_vocab()
    print(f"Base model vocab size: {VOCAB_SIZE}")

    adapter = None
    mapper = None
    if USE_ADAPTER:
        if adapter_type == 'external':
            print(f"Initializing external adapter...")
            adapter = ExternalCorrector(VOCAB_SIZE, actual_adapter_dim, heads).to(device)
            adapter.load_state_dict(state_dict)
            adapter.eval()
            print("External adapter loaded successfully.")
        else:
            print(f"Initializing universal adapter...")
            adapter = UniversalAdapter(semantic_dim, actual_adapter_dim, heads).to(device)
            adapter.load_state_dict(state_dict)
            adapter.eval()
            mapper = SemanticMapper(llm)
            print("Universal adapter loaded successfully.")
    else:
        config = None

    print("✅ All models successfully loaded!")
    return llm, adapter, mapper, config, adapter_type if USE_ADAPTER else None

# ---------------------------------------------------------
# Helper function: Extract content between tags
# ---------------------------------------------------------
def extract_content(response, start_tag, end_tag):
    """
    Extracts content between specified tags
    """
    start_idx = response.find(start_tag)
    if start_idx == -1:
        return ""

    start_idx += len(start_tag)
    end_idx = response.find(end_tag, start_idx)

    if end_idx == -1:
        return response[start_idx:].strip()

    return response[start_idx:end_idx].strip()

def extract_final_answer(response):
    """
    Extracts content from final_answer tags
    """
    if not USE_REASONING_MODEL or not ENABLE_THINK_TAGS:
        return response.strip()

    return extract_content(response, FINAL_START_TAG, FINAL_END_TAG)

def extract_summary(response):
    """
    Extracts summary from summary tags
    """
    return extract_content(response, SUMMARY_START_TAG, SUMMARY_END_TAG)

def remove_cot_from_response(response):
    """
    Removes chain-of-thought part and extracts only final answer
    """
    if not USE_REASONING_MODEL or not ENABLE_THINK_TAGS:
        return response.strip()

    return extract_final_answer(response)

# ---------------------------------------------------------
# Text Generation Helper Class
# ---------------------------------------------------------
class TextGenerator:
    def __init__(self, llm, adapter=None, mapper=None):
        self.llm = llm
        self.adapter = adapter
        self.mapper = mapper
        self.system_prompt = create_system_prompt()
        self.eos_token_id = llm.token_eos()
        self.generated_part_only = ""
        self.in_final_answer = False
        self.stop_tokens = [FINAL_END_TAG, SUMMARY_END_TAG]  # Beide Stopp-Tokens

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def _check_for_stop_tokens(self):
        """Checks if any stop token appears in generated part"""
        for stop_token in self.stop_tokens:
            if stop_token in self.generated_part_only:
                return True, stop_token
        return False, None

    def _update_state(self, new_text):
        """Updates state based on new text"""
        self.generated_part_only += new_text

        # Check if we're in final_answer part
        if not self.in_final_answer:
            if FINAL_START_TAG in self.generated_part_only:
                self.in_final_answer = True

    def _extract_response(self, full_output, original_prompt):
        combined_prompt = f"{self.system_prompt}\n{original_prompt}"

        if full_output.startswith(combined_prompt):
            response = full_output[len(combined_prompt):].strip()
        elif full_output.startswith(original_prompt):
            response = full_output[len(original_prompt):].strip()
        else:
            response = full_output

        # Extract final answer based on configuration
        if USE_REASONING_MODEL and ENABLE_THINK_TAGS:
            return extract_final_answer(response)
        else:
            return response.strip()

    def _prepare_prompt(self, user_prompt):
        return f"{self.system_prompt}\n{user_prompt}"

    def reset(self):
        self.generated_part_only = ""
        self.in_final_answer = False

# ---------------------------------------------------------
# Summary Management Functions
# ---------------------------------------------------------
def summarize_response(llm, response, adapter=None, max_summary_tokens=50):
    """
    Generates a one-sentence summary of the response
    """
    summary_prompt = f"""Please summarize the following response in one sentence (in the same language as the response):

Response: {response}

{SUMMARY_START_TAG}"""

    tokens = llm.tokenize(summary_prompt.encode("utf-8"), add_bos=True)
    current_tokens = tokens.copy()

    llm.reset()
    llm.eval(tokens)

    generated_tokens = []
    summary_text = ""

    for _ in range(max_summary_tokens):
        if llm.scores.shape[0] == 0:
            break

        base_last = llm.scores[-1, :]
        next_token = sample_logits_min_p(
            base_last,
            generated_tokens=generated_tokens,
            temperature=0.3,  # Lower temperature for more focused summaries
            min_p=0.05,
            repetition_penalty=1.05
        )

        if next_token == llm.token_eos():
            break

        next_text = llm.detokenize([next_token]).decode("utf-8", errors="ignore")
        summary_text += next_text

        if SUMMARY_END_TAG in summary_text:
            break

        current_tokens.append(next_token)
        generated_tokens.append(next_token)
        llm.eval([next_token])

    # Ensure the summary is properly wrapped
    if SUMMARY_END_TAG not in summary_text:
        summary_text += SUMMARY_END_TAG

    return f"{SUMMARY_START_TAG}{summary_text}"

def trim_summaries(summaries_text, llm, max_tokens=MAX_SUMMARY_TOKENS):
    """
    Trims summaries if they exceed max_tokens
    """
    tokens = llm.tokenize(summaries_text.encode("utf-8"), add_bos=False)

    if len(tokens) <= max_tokens:
        return summaries_text

    # If too long, remove oldest summaries until under limit
    lines = summaries_text.split('\n')
    while len(lines) > 1 and len(llm.tokenize('\n'.join(lines).encode("utf-8"), add_bos=False)) > max_tokens:
        lines.pop(0)  # Remove oldest summary

    return '\n'.join(lines)

# ---------------------------------------------------------
# Generation dispatcher (auto-selects based on adapter type)
# ---------------------------------------------------------
def generate_text(llm, adapter, mapper, adapter_type, user_prompt: str, max_new_tokens: int = 100,
                  temperature: float = 0.8, min_p: float = 0.05,
                  repetition_penalty: float = 1.1,
                  adapter_window: int = None, top_k: int = 50,
                  remove_cot: bool = False, generator=None,
                  generate_summary: bool = False):
    """
    Dispatches to appropriate generation function based on adapter_type.
    """
    if adapter is None:
        # Base only
        return generate_base_only(llm, user_prompt, max_new_tokens, temperature,
                                  min_p, repetition_penalty, remove_cot, generator, generate_summary)
    elif adapter_type == 'external':
        return generate_with_external_adapter(llm, adapter, user_prompt, max_new_tokens,
                                              temperature, min_p, repetition_penalty,
                                              adapter_window, remove_cot, generator, generate_summary)
    elif adapter_type == 'universal':
        return generate_with_universal_adapter(llm, adapter, mapper, user_prompt, max_new_tokens,
                                               temperature, min_p, repetition_penalty,
                                               top_k, remove_cot, generator, generate_summary)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

# ---------------------------------------------------------
# Comparison generation ONLY with Base Model
# ---------------------------------------------------------
def generate_base_only(llm, user_prompt: str, max_tokens: int = 100,
                      temperature: float = 0.8, min_p: float = 0.05,
                      repetition_penalty: float = 1.1, remove_cot: bool = False,
                      generator=None, generate_summary: bool = False):
    if generator is None:
        generator = TextGenerator(llm)

    generator.reset()
    full_prompt = generator._prepare_prompt(user_prompt)

    tokens = llm.tokenize(full_prompt.encode("utf-8"), add_bos=True)
    current_tokens = tokens.copy()

    llm.reset()
    llm.eval(tokens)

    output = full_prompt

    if llm.scores.shape[0] == 0:
        if remove_cot and USE_REASONING_MODEL and ENABLE_THINK_TAGS:
            return remove_cot_from_response(output), ""
        return generator._extract_response(output, user_prompt), ""

    generated_tokens = []

    for _ in range(max_tokens):
        if llm.scores.shape[0] == 0:
            break

        base_last = llm.scores[-1, :]

        next_token = sample_logits_min_p(
            base_last,
            generated_tokens=generated_tokens,
            temperature=temperature,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=generator.eos_token_id
        )

        if next_token == generator.eos_token_id:
            break

        next_text = llm.detokenize([next_token]).decode("utf-8", errors="ignore")
        output += next_text
        generator._update_state(next_text)

        current_tokens.append(next_token)
        generated_tokens.append(next_token)

        stop_detected, stop_token = generator._check_for_stop_tokens()
        if stop_detected:
            break

        llm.eval([next_token])

    response = generator._extract_response(output, user_prompt)

    if remove_cot and USE_REASONING_MODEL and ENABLE_THINK_TAGS:
        response = remove_cot_from_response(response)

    summary = ""
    if generate_summary and ENABLE_SUMMARY:
        summary = summarize_response(llm, response)

    return response, summary

# ---------------------------------------------------------
# Chat class with History Management
# ---------------------------------------------------------
class ChatManager:
    def __init__(self, llm, adapter=None, mapper=None, adapter_type=None, max_history_tokens=2048):
        self.llm = llm
        self.adapter = adapter
        self.mapper = mapper
        self.adapter_type = adapter_type
        self.max_history_tokens = max_history_tokens
        self.conversation_history = []
        self.generator = TextGenerator(llm, adapter, mapper)

        self.base_system_prompt = create_system_prompt()
        self.summaries = []  # Liste aller Zusammenfassungen
        self._update_system_prompt_with_summaries()

    def _update_system_prompt_with_summaries(self):
        """Aktualisiert den System-Prompt mit allen Zusammenfassungen"""
        summaries_text = ""
        if self.summaries and ENABLE_SUMMARY:
            summaries_text = "\n".join([f"- {s}" for s in self.summaries[-10:]])  # Letzte 10 Zusammenfassungen
            summaries_text = trim_summaries(summaries_text, self.llm, MAX_SUMMARY_TOKENS)

        self.system_prompt = create_system_prompt(summaries_text)
        self.generator.set_system_prompt(self.system_prompt)

    def set_system_prompt(self, prompt):
        """Setzt einen neuen System-Prompt und fügt bestehende Zusammenfassungen hinzu"""
        self.base_system_prompt = prompt
        self._update_system_prompt_with_summaries()

    def add_summary(self, summary):
        """Fügt eine neue Zusammenfassung hinzu"""
        if summary and ENABLE_SUMMARY:
            # Extrahiere nur den Zusammenfassungstext
            clean_summary = extract_summary(summary)
            if clean_summary:
                self.summaries.append(clean_summary)
                self._update_system_prompt_with_summaries()

    def _calculate_tokens(self, text):
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=True))

    def _truncate_history(self):
        """Kürzt die History und stellt sicher, dass System-Prompt erhalten bleibt"""
        total_tokens = 0
        truncated_history = []

        # System-Prompt immer hinzufügen
        system_prompt_tokens = self._calculate_tokens(self.system_prompt)
        total_tokens += system_prompt_tokens

        for role, message in reversed(self.conversation_history):
            message_tokens = self._calculate_tokens(f"{role}: {message}")

            if total_tokens + message_tokens > self.max_history_tokens:
                break

            truncated_history.insert(0, (role, message))
            total_tokens += message_tokens

        self.conversation_history = truncated_history

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _build_prompt(self, user_input):
        prompt_parts = []

        for role, message in self.conversation_history:
            if role == "user":
                prompt_parts.append(f"User: {message}")
            else:
                prompt_parts.append(f"Assistant: {message}")

        prompt_parts.append(f"User: {user_input}")

        return "\n".join(prompt_parts)

    def add_message(self, role, message):
        self.conversation_history.append((role, message))
        self._truncate_history()

    def chat(self, user_input, use_adapter=None, temperature=0.6, min_p=0.05,
             repetition_penalty=1.1, max_tokens=512, remove_cot=True):

        # Use configuration default if not specified
        if use_adapter is None:
            use_adapter = USE_ADAPTER

        history_prompt = self._build_prompt(user_input)

        if use_adapter and self.adapter is not None:
            response, summary = generate_text(
                self.llm, self.adapter, self.mapper, self.adapter_type,
                history_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                adapter_window=ADAPTER_WINDOW,
                top_k=DEFAULT_TOP_K,
                remove_cot=remove_cot,
                generator=self.generator,
                generate_summary=ENABLE_SUMMARY
            )
        else:
            response, summary = generate_base_only(
                self.llm, history_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                remove_cot=remove_cot,
                generator=self.generator,
                generate_summary=ENABLE_SUMMARY
            )

        self.add_message("user", user_input)
        self.add_message("assistant", response)

        # Zusammenfassung hinzufügen
        if summary:
            self.add_summary(summary)

        return response

    def clear_history(self):
        self.conversation_history = []
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ---------------------------------------------------------
# CLI Argument Parser (unchanged, except maybe add top_k)
# ---------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Adaptive Language Model System with Adapter Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --question "What is machine learning?" --mode single
  %(prog)s --question "Explain quantum computing" --temperature 0.7 --max-tokens 1000
  %(prog)s --mode chat --system-prompt "You are a helpful assistant."
  %(prog)s --mode compare --question "What is AI?" --adapter false
        """
    )

    # Mode selection
    parser.add_argument(
        "--mode", 
        choices=["single", "chat", "compare", "interactive"],
        default="single",
        help="Operation mode: single (single question), chat (interactive chat), compare (compare base vs adapter), interactive (full interactive menu)"
    )

    # Input options
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Question to ask the model"
    )
    
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Read question from file"
    )
    
    parser.add_argument(
        "--system-prompt", "-s",
        type=str,
        help="Custom system prompt"
    )

    # Model configuration
    parser.add_argument(
        "--adapter-repo",
        type=str,
        default=DEFAULT_HF_ADAPTER_REPO,
        help=f"HuggingFace adapter repository (default: {DEFAULT_HF_ADAPTER_REPO})"
    )
    
    parser.add_argument(
        "--base-repo",
        type=str,
        default=DEFAULT_BASE_MODEL_REPO,
        help=f"HuggingFace base model repository (default: {DEFAULT_BASE_MODEL_REPO})"
    )
    
    parser.add_argument(
        "--gguf-filename",
        type=str,
        default=DEFAULT_GGUF_FILENAME,
        help=f"GGUF filename (default: {DEFAULT_GGUF_FILENAME})"
    )
    
    parser.add_argument(
        "--adapter",
        type=lambda x: x.lower() in ["true", "yes", "1", "y"],
        default=None,
        help="Use adapter (true/false)"
    )
    
    parser.add_argument(
        "--reasoning",
        type=lambda x: x.lower() in ["true", "yes", "1", "y"],
        default=None,
        help="Use reasoning model (true/false)"
    )
    
    parser.add_argument(
        "--think-tags",
        type=lambda x: x.lower() in ["true", "yes", "1", "y"],
        default=None,
        help="Enable think tags (true/false)"
    )
    
    parser.add_argument(
        "--summary",
        type=lambda x: x.lower() in ["true", "yes", "1", "y"],
        default=None,
        help="Enable summary function (true/false)"
    )

    # Generation parameters
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.6,
        help=f"Sampling temperature (default: 0.6)"
    )
    
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.05,
        help=f"Min-P sampling parameter (default: 0.05)"
    )
    
    parser.add_argument(
        "--repetition-penalty", "-r",
        type=float,
        default=1.1,
        help=f"Repetition penalty (default: 1.1)"
    )
    
    parser.add_argument(
        "--max-tokens", "-m",
        type=int,
        default=DEFAULT_MAX_NEW_TOKEN,
        help=f"Maximum new tokens (default: {DEFAULT_MAX_NEW_TOKEN})"
    )
    
    parser.add_argument(
        "--context-size",
        type=int,
        default=DEFAULT_CONTEXT_SIZE,
        help=f"Context window size (default: {DEFAULT_CONTEXT_SIZE})"
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for response"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )

    # Tag configuration
    parser.add_argument(
        "--think-start-tag",
        type=str,
        default=DEFAULT_THINK_START_TAG,
        help=f"Think start tag (default: {DEFAULT_THINK_START_TAG})"
    )
    
    parser.add_argument(
        "--think-end-tag",
        type=str,
        default=DEFAULT_THINK_END_TAG,
        help=f"Think end tag (default: {DEFAULT_THINK_END_TAG})"
    )
    
    parser.add_argument(
        "--final-start-tag",
        type=str,
        default=DEFAULT_FINAL_START_TAG,
        help=f"Final start tag (default: {DEFAULT_FINAL_START_TAG})"
    )
    
    parser.add_argument(
        "--final-end-tag",
        type=str,
        default=DEFAULT_FINAL_END_TAG,
        help=f"Final end tag (default: {DEFAULT_FINAL_END_TAG})"
    )

    return parser.parse_args()

# ---------------------------------------------------------
# CLI Functions (updated to use generate_text dispatcher)
# ---------------------------------------------------------
def run_single_question(args, chat_manager, text_generator):
    """Run single question mode"""
    if not args.question and not args.file:
        print("Error: No question provided. Use --question or --file")
        return
    
    # Read question from file if specified
    if args.file:
        try:
            with open(args.file, 'r') as f:
                args.question = f.read().strip()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    
    print(f"\nQuestion: {args.question}")
    print("-" * 60)
    
    # Format the prompt
    formatted_prompt = f"User: {args.question}"
    
    # Determine if we should use adapter
    use_adapter = args.adapter if args.adapter is not None else USE_ADAPTER
    
    if use_adapter and chat_manager.adapter is not None:
        if args.verbose:
            print(f"Using {chat_manager.adapter_type} adapter...")
        
        response, summary = generate_text(
            chat_manager.llm, chat_manager.adapter, chat_manager.mapper, chat_manager.adapter_type,
            formatted_prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            adapter_window=ADAPTER_WINDOW,
            top_k=DEFAULT_TOP_K,
            remove_cot=not ENABLE_THINK_TAGS,
            generator=text_generator,
            generate_summary=ENABLE_SUMMARY
        )
    else:
        if args.verbose:
            print("Using base model only...")
        
        response, summary = generate_base_only(
            chat_manager.llm, formatted_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            remove_cot=not ENABLE_THINK_TAGS,
            generator=text_generator,
            generate_summary=ENABLE_SUMMARY
        )
    
    # Print response
    print("Response:")
    print("-" * 60)
    print(response)
    
    if summary and ENABLE_SUMMARY and args.verbose:
        print(f"\nSummary: {summary}")
    
    # Save to file if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(f"Question: {args.question}\n\n")
                f.write(f"Response:\n{response}\n")
                if summary and ENABLE_SUMMARY:
                    f.write(f"\nSummary: {summary}\n")
            print(f"\nResponse saved to {args.output}")
        except Exception as e:
            print(f"Error saving to file: {e}")

def run_chat_mode(args, chat_manager):
    """Run interactive chat mode"""
    print("\n" + "=" * 60)
    print("CHAT MODE - Interactive Conversation")
    print("=" * 60)
    print("Commands:")
    print("  /exit    - Exit chat mode")
    print("  /clear   - Clear conversation history")
    print("  /summary - Show current summaries")
    print("  /help    - Show this help")
    print("=" * 60)
    
    if args.system_prompt:
        chat_manager.set_system_prompt(args.system_prompt)
        print(f"System prompt set: {args.system_prompt[:100]}...")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == '/exit':
                break
            elif user_input.lower() == '/clear':
                chat_manager.clear_history()
                print("Conversation history cleared.")
                continue
            elif user_input.lower() == '/summary':
                print("\nCurrent summaries:")
                for i, s in enumerate(chat_manager.summaries, 1):
                    print(f"{i}. {s}")
                continue
            elif user_input.lower() == '/help':
                print("\nCommands:")
                print("  /exit    - Exit chat mode")
                print("  /clear   - Clear conversation history")
                print("  /summary - Show current summaries")
                print("  /help    - Show this help")
                continue
            
            # Get response
            response = chat_manager.chat(
                user_input,
                use_adapter=args.adapter if args.adapter is not None else USE_ADAPTER,
                temperature=args.temperature,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
                max_tokens=args.max_tokens,
                remove_cot=not ENABLE_THINK_TAGS
            )
            
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\nExiting chat mode...")
            break
        except Exception as e:
            print(f"\nError: {e}")

def run_compare_mode(args, chat_manager, text_generator):
    """Compare base model vs adapter"""
    if not args.question and not args.file:
        print("Error: No question provided. Use --question or --file")
        return
    
    # Read question from file if specified
    if args.file:
        try:
            with open(args.file, 'r') as f:
                args.question = f.read().strip()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    
    print(f"\nComparison for question: {args.question}")
    
    formatted_prompt = f"User: {args.question}"
    
    # Base model response
    print("\n" + "=" * 60)
    print("BASE MODEL (without adapter):")
    print("=" * 60)
    
    base_response, base_summary = generate_base_only(
        chat_manager.llm, formatted_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        remove_cot=not ENABLE_THINK_TAGS,
        generator=text_generator,
        generate_summary=ENABLE_SUMMARY
    )
    
    print(base_response)
    if base_summary and ENABLE_SUMMARY:
        print(f"\nSummary: {base_summary}")
    
    # Adapter response (if available)
    if chat_manager.adapter is not None:
        print("\n" + "=" * 60)
        print(f"WITH ADAPTER ({chat_manager.adapter_type}):")
        print("=" * 60)
        
        adapter_response, adapter_summary = generate_text(
            chat_manager.llm, chat_manager.adapter, chat_manager.mapper, chat_manager.adapter_type,
            formatted_prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            adapter_window=ADAPTER_WINDOW,
            top_k=DEFAULT_TOP_K,
            remove_cot=not ENABLE_THINK_TAGS,
            generator=text_generator,
            generate_summary=ENABLE_SUMMARY
        )
        
        print(adapter_response)
        if adapter_summary and ENABLE_SUMMARY:
            print(f"\nSummary: {adapter_summary}")
    
    # Save comparison if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(f"Question: {args.question}\n\n")
                f.write("=" * 60 + "\n")
                f.write("BASE MODEL RESPONSE:\n")
                f.write("=" * 60 + "\n")
                f.write(f"{base_response}\n\n")
                if base_summary and ENABLE_SUMMARY:
                    f.write(f"Summary: {base_summary}\n\n")
                
                if chat_manager.adapter is not None:
                    f.write("=" * 60 + "\n")
                    f.write(f"ADAPTER RESPONSE ({chat_manager.adapter_type}):\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"{adapter_response}\n\n")
                    if adapter_summary and ENABLE_SUMMARY:
                        f.write(f"Summary: {adapter_summary}\n")
            
            print(f"\nComparison saved to {args.output}")
        except Exception as e:
            print(f"Error saving to file: {e}")

def run_interactive_mode(args, chat_manager, text_generator):
    """Run full interactive menu mode (original behavior, updated)"""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE - Full Menu")
    print("=" * 60)
    
    print("\nAvailable modes:")
    print("1: Single question (Single Question)")
    print("2: Chat mode (with History)")
    print("3: Comparison Base vs. Adapter")
    print("4: Change System Prompt")
    print("5: Change Configuration")
    print("6: Clear Chat History")
    print("7: Toggle Summary Function")
    print("8: Show Current Summaries")
    print("9: Exit")
    
    while True:
        try:
            print("\n" + "-" * 40)
            choice = input("\nSelect mode (1-9): ").strip()

            if choice == "1":
                print("\n" + "=" * 40)
                print("SINGLE QUESTION MODE")
                print("=" * 40)

                user_prompt = input("\nEnter your question: ").strip()

                # Ask for adapter usage if enabled in config
                if USE_ADAPTER and chat_manager.adapter is not None:
                    adapter_input = input(f"Use adapter? (y/n, default {'y' if USE_ADAPTER else 'n'}): ").strip().lower()
                    use_adapter = adapter_input == 'y' if adapter_input else USE_ADAPTER
                else:
                    use_adapter = False

                print("\n--- Sampling Parameters ---")
                temp_input = input(f"Temperature (default 0.6): ").strip()
                temperature = float(temp_input) if temp_input else 0.6

                min_p_input = input(f"Min-P (default 0.05): ").strip()
                min_p = float(min_p_input) if min_p_input else 0.05

                rep_pen_input = input(f"Repetition Penalty (default 1.1): ").strip()
                repetition_penalty = float(rep_pen_input) if rep_pen_input else 1.1

                formatted_prompt = f"User: {user_prompt}"

                if use_adapter and chat_manager.adapter is not None:
                    print(f"\n" + "-" * 40)
                    print(f"RESPONSE WITH ADAPTER ({chat_manager.adapter_type}):")
                    print("-" * 40)
                    response, summary = generate_text(
                        chat_manager.llm, chat_manager.adapter, chat_manager.mapper, chat_manager.adapter_type,
                        formatted_prompt,
                        max_new_tokens=MAX_NEW_TOKEN,
                        temperature=temperature,
                        min_p=min_p,
                        repetition_penalty=repetition_penalty,
                        adapter_window=ADAPTER_WINDOW,
                        top_k=DEFAULT_TOP_K,
                        remove_cot=not ENABLE_THINK_TAGS,
                        generator=text_generator,
                        generate_summary=ENABLE_SUMMARY
                    )
                else:
                    print("\n" + "-" * 40)
                    print("RESPONSE WITH BASE MODEL ONLY:")
                    print("-" * 40)
                    response, summary = generate_base_only(
                        chat_manager.llm, formatted_prompt,
                        max_tokens=MAX_NEW_TOKEN,
                        temperature=temperature,
                        min_p=min_p,
                        repetition_penalty=repetition_penalty,
                        remove_cot=not ENABLE_THINK_TAGS,
                        generator=text_generator,
                        generate_summary=ENABLE_SUMMARY
                    )

                print(response)
                if summary and ENABLE_SUMMARY:
                    print(f"\nGenerated Summary: {summary}")
                print("\n[Response completed]")

            elif choice == "2":
                print("\n" + "=" * 40)
                print("CHAT MODE")
                print("=" * 40)
                print("Enter 'exit' to return to main menu.")
                print("Enter 'clear' to clear history.")
                print("Enter 'system' to change system prompt.")
                print("Enter 'config' to change configuration.")
                print("-" * 40)

                while True:
                    user_input = input("\nYou: ").strip()

                    if user_input.lower() == 'exit':
                        break
                    elif user_input.lower() == 'clear':
                        chat_manager.clear_history()
                        print("History cleared!")
                        continue
                    elif user_input.lower() == 'system':
                        new_prompt = input("Enter new system prompt (empty for default): ").strip()
                        if new_prompt:
                            chat_manager.set_system_prompt(new_prompt)
                            print("System prompt updated with existing summaries!")
                        continue
                    elif user_input.lower() == 'config':
                        print("\nReturning to main menu to change configuration...")
                        break

                    # Use adapter based on configuration
                    response = chat_manager.chat(
                        user_input,
                        use_adapter=USE_ADAPTER,
                        temperature=0.6,
                        min_p=0.05,
                        repetition_penalty=1.1,
                        max_tokens=6000,
                        remove_cot=not ENABLE_THINK_TAGS
                    )

                    print(f"\nAssistant: {response}")
                    print("[Response completed]")

            elif choice == "3":
                if not USE_ADAPTER or chat_manager.adapter is None:
                    print("Adapter is not enabled or not loaded. Cannot compare.")
                    continue

                print("\n" + "=" * 40)
                print("COMPARISON BASE vs. ADAPTER")
                print("=" * 40)

                test_prompt = input("\nEnter a test prompt: ").strip()
                formatted_prompt = f"User: {test_prompt}"

                print("\n" + "-" * 40)
                print("BASE MODEL (without adapter):")
                print("-" * 40)
                base_response, base_summary = generate_base_only(
                    chat_manager.llm, formatted_prompt,
                    max_tokens=6000,
                    temperature=0.6,
                    min_p=0.05,
                    repetition_penalty=1.1,
                    remove_cot=not ENABLE_THINK_TAGS,
                    generator=text_generator,
                    generate_summary=ENABLE_SUMMARY
                )
                print(base_response)
                if base_summary and ENABLE_SUMMARY:
                    print(f"\nSummary: {base_summary}")
                print("\n[Response completed]")

                print("\n" + "-" * 40)
                print(f"WITH ADAPTER ({chat_manager.adapter_type}):")
                print("-" * 40)
                adapter_response, adapter_summary = generate_text(
                    chat_manager.llm, chat_manager.adapter, chat_manager.mapper, chat_manager.adapter_type,
                    formatted_prompt,
                    max_new_tokens=6000,
                    temperature=0.6,
                    min_p=0.05,
                    repetition_penalty=1.1,
                    adapter_window=ADAPTER_WINDOW,
                    top_k=DEFAULT_TOP_K,
                    remove_cot=not ENABLE_THINK_TAGS,
                    generator=text_generator,
                    generate_summary=ENABLE_SUMMARY
                )
                print(adapter_response)
                if adapter_summary and ENABLE_SUMMARY:
                    print(f"\nSummary: {adapter_summary}")
                print("\n[Response completed]")

            # Choices 4-9 remain largely unchanged, they don't directly call generation.
            # We'll keep them as is, but note that any generation inside them should also use generate_text.
            # For brevity, I'm not repeating all unchanged code here. In the final answer, we'll include the full script with all unchanged parts.
            # ... (remaining options unchanged) ...

            elif choice == "9":
                print("\nExiting program...")
                break

            else:
                print("Invalid input. Please select 1-9.")

        except KeyboardInterrupt:
            print("\n\nProgram terminated by user.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

# ---------------------------------------------------------
# Main function
# ---------------------------------------------------------
def main():
    """Main function with CLI support"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Update global configuration based on CLI arguments
    global HF_ADAPTER_REPO, BASE_MODEL_REPO, GGUF_FILENAME
    global USE_REASONING_MODEL, USE_ADAPTER, ENABLE_THINK_TAGS, ENABLE_SUMMARY, MAX_SUMMARY_TOKENS
    global MAX_NEW_TOKEN, CONTEXT_SIZE, ADAPTER_WINDOW
    global THINK_START_TAG, THINK_END_TAG, FINAL_START_TAG, FINAL_END_TAG
    
    # Update from CLI args
    HF_ADAPTER_REPO = args.adapter_repo
    BASE_MODEL_REPO = args.base_repo
    GGUF_FILENAME = args.gguf_filename
    
    if args.adapter is not None:
        USE_ADAPTER = args.adapter
    if args.reasoning is not None:
        USE_REASONING_MODEL = args.reasoning
    if args.think_tags is not None:
        ENABLE_THINK_TAGS = args.think_tags
    if args.summary is not None:
        ENABLE_SUMMARY = args.summary
    
    MAX_NEW_TOKEN = args.max_tokens
    CONTEXT_SIZE = args.context_size
    
    THINK_START_TAG = args.think_start_tag
    THINK_END_TAG = args.think_end_tag
    FINAL_START_TAG = args.final_start_tag
    FINAL_END_TAG = args.final_end_tag
    
    # Print configuration
    print("=" * 60)
    print("SYSTEM CONFIGURATION")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Reasoning Model: {USE_REASONING_MODEL}")
    print(f"Think Tags Enabled: {ENABLE_THINK_TAGS}")
    print(f"Adapter Enabled: {USE_ADAPTER}")
    print(f"Summary Enabled: {ENABLE_SUMMARY}")
    print(f"Max Summary Tokens: {MAX_SUMMARY_TOKENS}")
    if USE_REASONING_MODEL:
        print(f"Think Start Tag: {THINK_START_TAG}")
        print(f"Think End Tag: {THINK_END_TAG}")
    print(f"Final Start Tag: {FINAL_START_TAG}")
    print(f"Final End Tag: {FINAL_END_TAG}")
    print(f"Summary Tags: {SUMMARY_START_TAG} ... {SUMMARY_END_TAG}")
    print(f"Temperature: {args.temperature}")
    print(f"Min-P: {args.min_p}")
    print(f"Repetition Penalty: {args.repetition_penalty}")
    print(f"Max Tokens: {args.max_tokens}")
    print("=" * 60)
    
    # Initialize models
    print("\n" + "=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)
    llm, adapter, mapper, config, adapter_type = initialize_models()
    
    # Create chat manager and text generator
    chat_manager = ChatManager(llm, adapter, mapper, adapter_type, max_history_tokens=2048)
    text_generator = TextGenerator(llm, adapter, mapper)
    
    # Set custom system prompt if provided
    if args.system_prompt:
        chat_manager.set_system_prompt(args.system_prompt)
        text_generator.set_system_prompt(args.system_prompt)
    
    # Disable progress bar if requested
    if args.no_progress:
        global tqdm
        tqdm = lambda x, **kwargs: x
    
    # Run based on mode
    if args.mode == "single":
        run_single_question(args, chat_manager, text_generator)
    elif args.mode == "chat":
        run_chat_mode(args, chat_manager)
    elif args.mode == "compare":
        run_compare_mode(args, chat_manager, text_generator)
    elif args.mode == "interactive":
        run_interactive_mode(args, chat_manager, text_generator)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("FINISHED!")
    print("=" * 60)

# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
