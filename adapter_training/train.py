#!/usr/bin/env python3
"""
Adapter training for various LLMs with CLI control
Supports ERNIE, Llama, Mistral and other GGUF models

Enhanced version:
- Real‑time progress bar (loss, LR, grad norm)
- Optional periodic evaluation and CSV logging
- Automatic continuation from an existing adapter repository
- Two adapter types: 'external' (original) and 'universal' (cross‑model)
"""

import os
import sys
import json
import math
import csv
import tempfile
import argparse
import warnings
warnings.filterwarnings("ignore")

# GPU optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from llama_cpp import Llama
from torch.nn.utils import clip_grad_norm_
from huggingface_hub import HfApi, create_repo, upload_folder, get_token, whoami
from huggingface_hub import hf_hub_download, list_repo_files
from sentence_transformers import SentenceTransformer

# =========================================================
# CLI Argument Parser
# =========================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train an adapter for various LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch on ERNIE (universal adapter)
  python train_adapter.py --model unsloth/ERNIE-4.5-21B-A3B-Thinking-GGUF

  # Train external adapter (original style)
  python train_adapter.py --model unsloth/ERNIE-4.5-21B-A3B-Thinking-GGUF --adapter-type external

  # Continue training an existing adapter from Hugging Face
  python train_adapter.py --model my-username/my-adapter-repo

  # Specify base model explicitly (if auto‑detection fails)
  python train_adapter.py --model my-adapter --base-model unsloth/ERNIE-4.5-21B-A3B-Thinking-GGUF

  # Enable periodic evaluation and logging
  python train_adapter.py --model ... --eval-steps 500 --log-file train_log.csv
        """
    )

    # Model parameters
    model_group = parser.add_argument_group('Model Settings')
    model_group.add_argument('--model', type=str, required=True,
                          help='Hugging Face repository (base model OR adapter)')
    model_group.add_argument('--base-model', type=str, default=None,
                          help='Explicit base model repository (if --model is an adapter)')
    model_group.add_argument('--filename', type=str, default=None,
                          help='Specific GGUF filename (if multiple available)')
    model_group.add_argument('--context-size', type=int, default=1024,
                          help='Context size (default: 1024)')
    model_group.add_argument('--adapter-dim', type=int, default=256,
                          help='Adapter dimension (default: 256)')
    model_group.add_argument('--heads', type=int, default=8,
                          help='Number of attention heads (default: 8)')
    model_group.add_argument('--adapter-type', type=str, default='universal',
                          choices=['universal', 'external'],
                          help='Adapter architecture (default: universal)')
    model_group.add_argument('--top-k', type=int, default=50,
                          help='Top‑k for universal adapter (default: 50)')
    model_group.add_argument('--semantic-dim', type=int, default=384,
                          help='Semantic embedding dimension (default: 384)')

    # Training parameters
    train_group = parser.add_argument_group('Training Settings')
    train_group.add_argument('--steps', type=int, default=14000,
                          help='Number of training steps (default: 14000)')
    train_group.add_argument('--learning-rate', type=float, default=5e-5,
                          help='Learning rate (default: 5e-5)')
    train_group.add_argument('--weight-decay', type=float, default=0.01,
                          help='Weight decay (default: 0.01)')
    train_group.add_argument('--accumulation-steps', type=int, default=32,
                          help='Gradient accumulation steps (default: 32)')
    train_group.add_argument('--batch-size', type=int, default=1,
                          help='Batch size (default: 1)')
    train_group.add_argument('--seed', type=int, default=42,
                          help='Random seed (default: 42)')
    train_group.add_argument('--eval-steps', type=int, default=0,
                          help='Evaluate every N steps (0 = disable) (default: 0)')
    train_group.add_argument('--log-file', type=str, default='training_log.csv',
                          help='CSV file to log training metrics (default: training_log.csv)')

    # Dataset parameters
    data_group = parser.add_argument_group('Dataset Settings')
    data_group.add_argument('--dataset', type=str,
                          default='prithivMLmods/Atlas-Think-Cot-12M',
                          help='Hugging Face dataset ID (default: prithivMLmods/Atlas-Think-Cot-12M)')
    data_group.add_argument('--prompt-col', type=str, default='problem',
                          help='Column name for prompts (default: problem)')
    data_group.add_argument('--output-col', type=str, default='solution',
                          help='Column name for outputs (default: solution)')
    data_group.add_argument('--val-samples', type=int, default=50,
                          help='Number of validation samples (default: 50)')
    data_group.add_argument('--max-length', type=int, default=None,
                          help='Maximum text length (default: None = auto)')

    # Hugging Face upload
    hf_group = parser.add_argument_group('Hugging Face Upload')
    hf_group.add_argument('--hf-repo', type=str, default=None,
                         help='Hugging Face repository for upload')
    hf_group.add_argument('--hf-private', action='store_true', default=True,
                         help='Mark repository as private (default: True)')
    hf_group.add_argument('--no-upload', action='store_true',
                         help='No upload to Hugging Face')
    hf_group.add_argument('--hf-token', type=str, default=None,
                         help='Hugging Face token (optional)')

    # Other options
    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument('--output-dir', type=str, default='checkpoints',
                          help='Output directory (default: checkpoints)')
    other_group.add_argument('--checkpoint', type=str, default=None,
                          help='Load adapter checkpoint (overrides auto‑load from --model)')
    other_group.add_argument('--eval-only', action='store_true',
                          help='Perform only evaluation')
    other_group.add_argument('--gpu-layers', type=int, default=-1,
                          help='Number of GPU layers (-1 = all)')
    other_group.add_argument('--verbose', action='store_true',
                          help='Detailed outputs')
    other_group.add_argument('--save-every', type=int, default=1000,
                          help='Save checkpoint every N steps')
    other_group.add_argument('--resume', action='store_true',
                          help='Resume training from latest checkpoint in output dir')

    return parser.parse_args()

# =========================================================
# Model Classes
# =========================================================

# ---------- External Adapter (original) ----------
class CausalAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        return self.to_out(out)

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

    def forward(self, input_ids, llm_logits):
        x_ctx = self.token_emb(input_ids)
        x_logits = F.gelu(self.logit_compressor(llm_logits))
        x = x_ctx + x_logits
        x = self.input_norm(x)

        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        correction_logits = self.head(x)
        return correction_logits


# ---------- Universal Adapter (cross‑model) ----------
class SemanticMapper:
    def __init__(self, llm, device):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.llm = llm
        self.cache = {}
        self.device = device

    def get_vectors(self, token_ids):
        flat_ids = token_ids.flatten().tolist()
        unique_ids = list(set(flat_ids))
        missing = [i for i in unique_ids if i not in self.cache]
        if missing:
            texts = [self.llm.detokenize([i]).decode("utf-8", errors="ignore") for i in missing]
            with torch.no_grad():
                embs = self.embedder.encode(texts, convert_to_tensor=True, show_progress_bar=False).to(self.device)
            for i, emb in zip(missing, embs):
                self.cache[i] = emb
        res = torch.stack([self.cache[i] for i in flat_ids]).to(self.device)
        return res.view(*token_ids.shape, -1)

class UniversalAdapter(nn.Module):
    def __init__(self, semantic_dim=384, adapter_dim=256, heads=8):
        super().__init__()
        self.input_proj = nn.Linear(semantic_dim + 1, adapter_dim)
        self.ln1 = nn.LayerNorm(adapter_dim)
        self.attn = nn.MultiheadAttention(adapter_dim, num_heads=heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim * 4),
            nn.GELU(),
            nn.Linear(adapter_dim * 4, adapter_dim)
        )
        self.ln2 = nn.LayerNorm(adapter_dim)
        self.output_head = nn.Linear(adapter_dim, 1)

    def forward(self, sem_embs, top_k_logits):
        # sem_embs: (batch, seq, top_k, semantic_dim)
        # top_k_logits: (batch, seq, top_k)
        b, s, k, d = sem_embs.shape
        x = torch.cat([sem_embs, top_k_logits.unsqueeze(-1)], dim=-1)  # (b,s,k,d+1)
        x = self.input_proj(x).view(b * s, k, -1)
        res, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + res + self.ffn(self.ln2(x))
        return self.output_head(x).view(b, s, k)


# =========================================================
# Helper Functions
# =========================================================

def setup_device():
    """GPU/CPU Setup"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  CUDA not available - using CPU")
    return device

def load_model(model_repo, filename=None, context_size=1024, gpu_layers=-1, verbose=False):
    """Loads the base model (GGUF)"""
    print(f"\n📥 Loading base model: {model_repo}")
    try:
        llm = Llama.from_pretrained(
            repo_id=model_repo,
            filename=filename,
            logits_all=True,
            n_ctx=context_size,
            n_gpu_layers=gpu_layers if torch.cuda.is_available() else 0,
            verbose=verbose,
            n_threads=4 if torch.cuda.is_available() else os.cpu_count() // 2
        )
        print(f"✅ Model loaded:")
        print(f"   - Vocab Size: {llm.n_vocab()}")
        print(f"   - Context Size: {llm.n_ctx()}")
        print(f"   - Model Size: {llm._model.n_params() / 1e9:.1f}B parameters")
        return llm
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"   Available files in {model_repo}:")
        try:
            files = list(list_repo_files(model_repo))
            for file in files[:10]:
                if file.endswith('.gguf'):
                    print(f"     - {file}")
            if len(files) > 10:
                print(f"     ... and {len(files)-10} more")
        except:
            pass
        sys.exit(1)

def try_load_adapter_repo(repo_id, token=None):
    """
    Check if repo_id contains a saved adapter (config.json + adapter_final.pt).
    If yes, download both and return a dict with:
        base_model: str (from config)
        config: dict
        weights_path: str (local path to adapter_final.pt)
    Otherwise return None.
    """
    try:
        files = list_repo_files(repo_id, token=token)
    except Exception:
        return None

    if 'config.json' not in files or 'adapter_final.pt' not in files:
        return None

    # Download config
    try:
        config_path = hf_hub_download(repo_id=repo_id, filename='config.json', token=token)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception:
        return None

    # Verify it's an adapter config (has 'adapter' key or 'base_model' in model)
    if 'adapter' not in config and ('model' not in config or 'base_model' not in config['model']):
        return None

    # Download weights
    try:
        weights_path = hf_hub_download(repo_id=repo_id, filename='adapter_final.pt', token=token)
    except Exception:
        return None

    # Extract base model repo
    base_model = None
    if 'model' in config and 'base_model' in config['model']:
        base_model = config['model']['base_model']
    elif 'base_model' in config:  # fallback
        base_model = config['base_model']

    if not base_model:
        return None

    return {
        'base_model': base_model,
        'config': config,
        'weights_path': weights_path
    }

def validate_dataset_columns(dataset, prompt_col, output_col):
    """Checks if dataset columns exist"""
    try:
        sample = next(iter(dataset))
        available_cols = list(sample.keys())
        if prompt_col not in available_cols:
            print(f"⚠️  Warning: Prompt column '{prompt_col}' not found.")
            print(f"   Available columns: {available_cols}")
            suggestions = ['prompt', 'input', 'question', 'instruction', 'text']
            for s in suggestions:
                if s in available_cols:
                    print(f"   Suggestion: Use --prompt-col {s}")
                    break
        if output_col not in available_cols:
            print(f"⚠️  Warning: Output column '{output_col}' not found.")
            print(f"   Available columns: {available_cols}")
            suggestions = ['response', 'answer', 'output', 'completion', 'target']
            for s in suggestions:
                if s in available_cols:
                    print(f"   Suggestion: Use --output-col {s}")
                    break
        return True
    except Exception as e:
        print(f"❌ Error validating dataset: {e}")
        return False

def get_valid_text(sample, prompt_col, output_col, max_length=None):
    """Extracts valid text from dataset sample"""
    problem = sample.get(prompt_col, "")
    solution = sample.get(output_col, "")

    if not problem and prompt_col != 'problem':
        for key in ["prompt", "input", "question", "instruction", "text"]:
            if key in sample:
                problem = sample[key]
                break
    if not solution and output_col != 'solution':
        for key in ["response", "answer", "output", "completion", "target"]:
            if key in sample:
                solution = sample[key]
                break

    if not isinstance(problem, str) or not isinstance(solution, str):
        return None
    if len(problem.strip()) == 0 or len(solution.strip()) == 0:
        return None

    text = f"{problem.strip()}\n\n### Answer:\n{solution.strip()}"
    if max_length and len(text) > max_length:
        return None
    return text

def save_config(args, vocab_size, config_path):
    """Saves configuration"""
    config = {
        "adapter_type": args.adapter_type,
        "model": {
            "base_model": args.base_model if args.base_model else args.model,
            "filename": args.filename,
            "context_size": args.context_size,
            "vocab_size": vocab_size
        },
        "adapter": {
            "adapter_dim": args.adapter_dim,
            "heads": args.heads,
            "architecture": "ExternalCorrector" if args.adapter_type == "external" else "UniversalAdapter"
        },
        "training": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "accumulation_steps": args.accumulation_steps,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "eval_steps": args.eval_steps
        },
        "dataset": {
            "dataset_id": args.dataset,
            "prompt_column": args.prompt_col,
            "output_column": args.output_col,
            "val_samples": args.val_samples
        },
        "huggingface": {
            "repo": args.hf_repo,
            "private": args.hf_private
        },
        "hardware": {
            "device": str(args.device),
            "gpu_layers": args.gpu_layers
        }
    }
    if args.adapter_type == "universal":
        config["adapter"]["top_k"] = args.top_k
        config["adapter"]["semantic_dim"] = args.semantic_dim

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✅ Configuration saved: {config_path}")
    return config

def upload_to_huggingface(output_dir, repo_name, private=True, token=None):
    """Uploads model to Hugging Face"""
    try:
        print(f"\n🤗 Uploading to Hugging Face: {repo_name}")
        if token is None:
            token = get_token()
            if not token:
                print("⚠️  No Hugging Face token found.")
                print("   Please run: huggingface-cli login")
                print("   Or specify token with --hf-token")
                return False
        try:
            user_info = whoami(token=token)
            username = user_info.get('name', '')
            print(f"   Logged in as: {username}")
            if "/" not in repo_name:
                repo_name = f"{username}/{repo_name}"
            print(f"   Repository: {repo_name}")
            print(f"   Visibility: {'private' if private else 'public'}")
            create_repo(
                repo_id=repo_name,
                private=private,
                exist_ok=True,
                token=token
            )
        except Exception as e:
            print(f"⚠️  Error creating repository: {e}")
            return False

        api = HfApi()
        print("📤 Uploading files...")
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_name,
            token=token,
            commit_message=f"Adapter Training - {os.path.basename(output_dir)}",
            ignore_patterns=["__pycache__", "*.tmp", "*.log"],
            repo_type="model"
        )
        print(f"\n✅ Upload successful!")
        print(f"   Repository: https://huggingface.co/{repo_name}")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def evaluate_model_external(llm, adapter, val_texts, device, context_size):
    """Evaluates the external adapter (full vocab)"""
    print(f"\n📊 Evaluating external adapter on {len(val_texts)} examples...")
    total_base_loss = 0.0
    total_adapt_loss = 0.0
    total_tokens = 0
    adapter.eval()

    with torch.no_grad():
        for text in tqdm(val_texts, desc="Evaluating"):
            try:
                tokens = llm.tokenize(text.encode("utf-8"), add_bos=True)
            except:
                tokens = llm.tokenize(text, add_bos=True)

            if len(tokens) > context_size:
                tokens = tokens[-context_size:]
            if len(tokens) < 2:
                continue

            llm.reset()
            llm.eval(tokens)
            if len(llm.scores) < len(tokens) - 1:
                continue

            llm_logits_list = [np.array(s) for s in llm.scores[:len(tokens)-1]]
            input_llm_logits = torch.from_numpy(np.stack(llm_logits_list)).unsqueeze(0).to(device)

            input_ids = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(device)
            target_ids = torch.tensor(tokens[1:], dtype=torch.long).to(device)

            base_logits = input_llm_logits
            base_loss_sum = F.cross_entropy(
                base_logits.view(-1, llm.n_vocab()),
                target_ids,
                reduction='sum'
            ).item()

            correction = adapter(input_ids, input_llm_logits)
            final_logits = base_logits + correction
            adapt_loss_sum = F.cross_entropy(
                final_logits.view(-1, llm.n_vocab()),
                target_ids,
                reduction='sum'
            ).item()

            n_tokens = target_ids.numel()
            total_base_loss += base_loss_sum
            total_adapt_loss += adapt_loss_sum
            total_tokens += n_tokens

    if total_tokens > 0:
        avg_base_loss = total_base_loss / total_tokens
        avg_adapt_loss = total_adapt_loss / total_tokens

        base_ppl = math.exp(min(avg_base_loss, 700))
        adapt_ppl = math.exp(min(avg_adapt_loss, 700))

        print(f"\n{'='*50}")
        print("📈 EVALUATION RESULTS (external)")
        print(f"{'='*50}")
        print(f"Base Model Perplexity:    {base_ppl:.2f}")
        print(f"Adapted Model Perplexity: {adapt_ppl:.2f}")
        print(f"Improvement:             {base_ppl - adapt_ppl:+.2f}")
        print(f"Tokens evaluated:         {total_tokens:,}")
        print(f"{'='*50}")

        return {
            "base_perplexity": float(base_ppl),
            "adapted_perplexity": float(adapt_ppl),
            "improvement": float(base_ppl - adapt_ppl),
            "total_tokens": total_tokens,
            "samples_evaluated": len(val_texts)
        }
    return None

def evaluate_model_universal(llm, adapter, mapper, val_texts, device, context_size, top_k):
    """Evaluates the universal adapter (top‑k correction)"""
    print(f"\n📊 Evaluating universal adapter on {len(val_texts)} examples...")
    total_base_loss = 0.0
    total_adapt_loss = 0.0
    total_tokens = 0
    adapter.eval()

    with torch.no_grad():
        for text in tqdm(val_texts, desc="Evaluating"):
            try:
                tokens = llm.tokenize(text.encode("utf-8"), add_bos=True)
            except:
                tokens = llm.tokenize(text, add_bos=True)

            if len(tokens) > context_size:
                tokens = tokens[-context_size:]
            if len(tokens) < 2:
                continue

            llm.reset()
            llm.eval(tokens)
            if len(llm.scores) < len(tokens) - 1:
                continue

            logits_list = [np.array(s) for s in llm.scores[:len(tokens)-1]]
            base_logits = torch.from_numpy(np.stack(logits_list)).to(device)
            targets = torch.tensor(tokens[1:], dtype=torch.long, device=device)

            # Base loss
            base_loss = F.cross_entropy(base_logits, targets, reduction='sum').item()

            # Top‑k correction
            top_v, top_i = torch.topk(base_logits, top_k, dim=-1)
            sem_embs = mapper.get_vectors(top_i).unsqueeze(0)
            corrections = adapter(sem_embs, top_v.unsqueeze(0)).squeeze(0)
            final_logits = base_logits.clone()
            final_logits.scatter_add_(1, top_i, corrections)
            adapt_loss = F.cross_entropy(final_logits, targets, reduction='sum').item()

            total_base_loss += base_loss
            total_adapt_loss += adapt_loss
            total_tokens += targets.numel()

    if total_tokens > 0:
        avg_base_loss = total_base_loss / total_tokens
        avg_adapt_loss = total_adapt_loss / total_tokens

        base_ppl = math.exp(min(avg_base_loss, 700))
        adapt_ppl = math.exp(min(avg_adapt_loss, 700))

        print(f"\n{'='*50}")
        print("📈 EVALUATION RESULTS (universal)")
        print(f"{'='*50}")
        print(f"Base Model Perplexity:    {base_ppl:.2f}")
        print(f"Adapted Model Perplexity: {adapt_ppl:.2f}")
        print(f"Improvement:             {base_ppl - adapt_ppl:+.2f}")
        print(f"Tokens evaluated:         {total_tokens:,}")
        print(f"{'='*50}")

        return {
            "base_perplexity": float(base_ppl),
            "adapted_perplexity": float(adapt_ppl),
            "improvement": float(base_ppl - adapt_ppl),
            "total_tokens": total_tokens,
            "samples_evaluated": len(val_texts)
        }
    return None

def log_metrics(csv_path, step, base_loss, adapt_loss, lr, grad_norm,
                val_base_ppl=None, val_adapt_ppl=None):
    """Append one row to the training log CSV."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['step', 'base_loss', 'adapt_loss', 'lr', 'grad_norm',
                             'val_base_ppl', 'val_adapt_ppl'])
        writer.writerow([step, f"{base_loss:.4f}", f"{adapt_loss:.4f}",
                         f"{lr:.2e}", f"{grad_norm:.4f}",
                         f"{val_base_ppl:.2f}" if val_base_ppl else '',
                         f"{val_adapt_ppl:.2f}" if val_adapt_ppl else ''])

# =========================================================
# Main Function
# =========================================================

def main():
    args = parse_arguments()
    device = setup_device()
    args.device = device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------------------------------------------------
    # Determine base model and optionally load existing adapter
    # ------------------------------------------------------------------
    base_repo = args.base_model
    adapter_weights_path = None
    adapter_config_from_repo = None

    if not base_repo:
        # Check if --model points to an adapter repository
        adapter_info = try_load_adapter_repo(args.model, token=args.hf_token)
        if adapter_info is not None:
            print(f"\n🔍 Detected adapter repository: {args.model}")
            base_repo = adapter_info['base_model']
            adapter_config_from_repo = adapter_info['config']
            adapter_weights_path = adapter_info['weights_path']
            print(f"   Base model: {base_repo}")

            # Override adapter dimensions if not set explicitly
            if 'adapter' in adapter_config_from_repo:
                repo_dim = adapter_config_from_repo['adapter'].get('adapter_dim')
                repo_heads = adapter_config_from_repo['adapter'].get('heads')
                repo_type = adapter_config_from_repo.get('adapter_type')
                if repo_type:
                    args.adapter_type = repo_type
                    print(f"   Using adapter_type={repo_type} from config")
                if repo_dim and args.adapter_dim == 256:
                    args.adapter_dim = repo_dim
                    print(f"   Using adapter_dim={repo_dim} from config")
                if repo_heads and args.heads == 8:
                    args.heads = repo_heads
                    print(f"   Using heads={repo_heads} from config")
        else:
            base_repo = args.model  # treat as base model

    # Load the base LLM
    llm = load_model(
        model_repo=base_repo,
        filename=args.filename,
        context_size=args.context_size,
        gpu_layers=args.gpu_layers,
        verbose=args.verbose
    )

    # Initialize adapter
    print(f"\n🔧 Initializing adapter (type: {args.adapter_type})...")
    print(f"   Dimension: {args.adapter_dim}")
    print(f"   Heads: {args.heads}")

    if args.adapter_type == "external":
        adapter = ExternalCorrector(
            vocab_size=llm.n_vocab(),
            embed_dim=args.adapter_dim,
            heads=args.heads
        ).to(device)
    else:  # universal
        adapter = UniversalAdapter(
            semantic_dim=args.semantic_dim,
            adapter_dim=args.adapter_dim,
            heads=args.heads
        ).to(device)
        mapper = SemanticMapper(llm, device)

    # Load adapter weights if we have a path (and no --checkpoint override)
    if args.checkpoint:
        print(f"📂 Loading checkpoint: {args.checkpoint}")
        try:
            adapter.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print("✅ Checkpoint loaded")
        except Exception as e:
            print(f"❌ Error loading: {e}")
            sys.exit(1)
    elif adapter_weights_path:
        print(f"📂 Loading existing adapter weights from {adapter_weights_path}")
        try:
            adapter.load_state_dict(torch.load(adapter_weights_path, map_location=device))
            print("✅ Adapter weights loaded")
        except Exception as e:
            print(f"❌ Error loading adapter weights: {e}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------
    print(f"\n📂 Loading dataset: {args.dataset}")
    print(f"   Prompt column: '{args.prompt_col}'")
    print(f"   Output column: '{args.output_col}'")

    try:
        stream_ds = load_dataset(args.dataset, split="train", streaming=True)
        ds_iter = iter(stream_ds)
        validate_dataset_columns(stream_ds, args.prompt_col, args.output_col)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)

    # Validation samples
    print(f"\n🔍 Collecting {args.val_samples} validation examples...")
    val_texts = []
    pbar = tqdm(total=args.val_samples, desc="Sampling validation")
    while len(val_texts) < args.val_samples:
        try:
            sample = next(ds_iter)
        except StopIteration:
            ds_iter = iter(stream_ds)
            sample = next(ds_iter)

        text = get_valid_text(sample, args.prompt_col, args.output_col, args.max_length)
        if text is not None:
            val_texts.append(text)
            pbar.update(1)
    pbar.close()
    print(f"✅ {len(val_texts)} validation examples collected")

    # Evaluation only?
    if args.eval_only:
        if args.adapter_type == "external":
            metrics = evaluate_model_external(llm, adapter, val_texts, device, args.context_size)
        else:
            metrics = evaluate_model_universal(llm, adapter, mapper, val_texts, device, args.context_size, args.top_k)
        if metrics:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "eval_metrics.json"), 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        sys.exit(0)

    # ------------------------------------------------------------------
    # Optimizer, scheduler, output dir
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=args.learning_rate * 0.1
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    config = save_config(args, llm.n_vocab(), os.path.join(output_dir, "config.json"))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n🚀 Starting training...")
    print(f"   Steps: {args.steps:,}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Gradient Accumulation: {args.accumulation_steps}")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   Checkpoint every {args.save_every} steps")
    if args.eval_steps > 0:
        print(f"   Evaluate every {args.eval_steps} steps")
    print(f"{'='*50}")

    optimizer.zero_grad()

    accum_base_loss = 0.0
    accum_adapt_loss = 0.0
    accum_count = 0
    global_step = 0

    # Resume from local checkpoint if requested
    if args.resume:
        checkpoint_files = [f for f in os.listdir(output_dir) if f.startswith('adapter_step_')]
        if checkpoint_files:
            latest = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
            checkpoint_path = os.path.join(output_dir, latest)
            print(f"↻ Resuming from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            adapter.load_state_dict(checkpoint['adapter_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            global_step = checkpoint['global_step']
            print(f"   Resumed at step: {global_step}")

    log_path = os.path.join(output_dir, args.log_file)

    pbar = tqdm(range(global_step, args.steps), desc="Training", initial=global_step)

    for step in pbar:
        # Get next sample
        try:
            sample = next(ds_iter)
        except StopIteration:
            ds_iter = iter(stream_ds)
            sample = next(ds_iter)

        text = get_valid_text(sample, args.prompt_col, args.output_col, args.max_length)
        if text is None:
            continue

        # Tokenization
        try:
            tokens = llm.tokenize(text.encode("utf-8"), add_bos=True)
        except:
            tokens = llm.tokenize(text, add_bos=True)

        if len(tokens) > args.context_size:
            tokens = tokens[-args.context_size:]
        if len(tokens) < 2:
            continue

        adapter.train()
        llm.reset()
        llm.eval(tokens)

        if len(llm.scores) < len(tokens) - 1:
            continue

        # Common: get base logits
        logits_list = [np.array(s) for s in llm.scores[:len(tokens)-1]]
        base_logits = torch.from_numpy(np.stack(logits_list)).unsqueeze(0).to(device)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(device)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long).to(device)

        if args.adapter_type == "external":
            # External adapter
            correction = adapter(input_ids, base_logits)
            final_logits = base_logits + correction
            base_loss = F.cross_entropy(base_logits.view(-1, llm.n_vocab()), target_ids.view(-1))
            adapted_loss = F.cross_entropy(final_logits.view(-1, llm.n_vocab()), target_ids.view(-1))
        else:
            # Universal adapter
            # Compute top‑k and semantic embeddings
            top_v, top_i = torch.topk(base_logits.squeeze(0), args.top_k, dim=-1)  # (seq, top_k)
            sem_embs = mapper.get_vectors(top_i).unsqueeze(0)  # (1, seq, top_k, sem_dim)
            corrections = adapter(sem_embs, top_v.unsqueeze(0))  # (1, seq, top_k)
            # Apply corrections
            final_logits = base_logits.clone()
            final_logits.scatter_add_(2, top_i.unsqueeze(0), corrections)  # scatter add over vocab dimension
            base_loss = F.cross_entropy(base_logits.view(-1, llm.n_vocab()), target_ids.view(-1))
            adapted_loss = F.cross_entropy(final_logits.view(-1, llm.n_vocab()), target_ids.view(-1))

        (adapted_loss / args.accumulation_steps).backward()

        accum_base_loss += base_loss.item()
        accum_adapt_loss += adapted_loss.item()
        accum_count += 1

        # Gradient step
        if (step + 1) % args.accumulation_steps == 0:
            grad_norm = clip_grad_norm_(adapter.parameters(), max_norm=1.0).item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if accum_count > 0:
                avg_base = accum_base_loss / accum_count
                avg_adapt = accum_adapt_loss / accum_count
                delta = avg_base - avg_adapt
                current_lr = scheduler.get_last_lr()[0]

                pbar.set_postfix({
                    'Base': f'{avg_base:.3f}',
                    'Adapt': f'{avg_adapt:.3f}',
                    'Δ': f'{delta:+.3f}',
                    'LR': f'{current_lr:.2e}',
                    '|g|': f'{grad_norm:.3f}'
                })

                log_metrics(log_path, step+1, avg_base, avg_adapt, current_lr, grad_norm)

            accum_base_loss = accum_adapt_loss = accum_count = 0.0
            global_step += 1

        # Periodic evaluation
        if args.eval_steps > 0 and (step + 1) % args.eval_steps == 0:
            print(f"\n🔍 Running evaluation at step {step+1}...")
            if args.adapter_type == "external":
                metrics = evaluate_model_external(llm, adapter, val_texts, device, args.context_size)
            else:
                metrics = evaluate_model_universal(llm, adapter, mapper, val_texts, device, args.context_size, args.top_k)
            if metrics:
                log_metrics(log_path, step+1,
                           avg_base if 'avg_base' in locals() else None,
                           avg_adapt if 'avg_adapt' in locals() else None,
                           current_lr if 'current_lr' in locals() else None,
                           grad_norm if 'grad_norm' in locals() else None,
                           metrics['base_perplexity'], metrics['adapted_perplexity'])
            print()

        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"adapter_step_{step+1:06d}.pt")
            torch.save({
                'adapter_state': adapter.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'global_step': step + 1,
                'config': config
            }, checkpoint_path)
            print(f"\n💾 Checkpoint saved: {checkpoint_path}")

        # Memory management
        if step % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()

    # ------------------------------------------------------------------
    # Final save and evaluation
    # ------------------------------------------------------------------
    print(f"\n💾 Saving final model...")
    adapter_path = os.path.join(output_dir, "adapter_final.pt")
    torch.save(adapter.state_dict(), adapter_path)

    final_checkpoint = os.path.join(output_dir, "model_final.pt")
    torch.save({
        'adapter_state': adapter.state_dict(),
        'config': config,
        'training_info': {
            'total_steps': args.steps,
            'final_loss': accum_adapt_loss / max(accum_count, 1)
        }
    }, final_checkpoint)

    print(f"✅ Adapter saved: {adapter_path}")
    print(f"✅ Complete model: {final_checkpoint}")

    print(f"\n📊 Performing final evaluation...")
    if args.adapter_type == "external":
        metrics = evaluate_model_external(llm, adapter, val_texts, device, args.context_size)
    else:
        metrics = evaluate_model_universal(llm, adapter, mapper, val_texts, device, args.context_size, args.top_k)

    if metrics:
        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                **metrics,
                'training_steps': args.steps,
                'adapter_config': {
                    'type': args.adapter_type,
                    'dim': args.adapter_dim,
                    'heads': args.heads
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ Metrics saved: {metrics_path}")

    # Hugging Face upload
    if not args.no_upload and args.hf_repo:
        upload_to_huggingface(
            output_dir=output_dir,
            repo_name=args.hf_repo,
            private=args.hf_private,
            token=args.hf_token
        )

    print(f"\n{'='*50}")
    print("🎉 TRAINING COMPLETED!")
    print(f"{'='*50}")
    print(f"Base Model:   {base_repo}")
    print(f"Dataset:      {args.dataset}")
    print(f"Steps:        {args.steps:,}")
    print(f"Adapter Type: {args.adapter_type}")
    print(f"Adapter Dim:  {args.adapter_dim}")
    print(f"Checkpoints:  {output_dir}/")
    if metrics:
        print(f"Base PPL:     {metrics['base_perplexity']:.2f}")
        print(f"Adapted PPL:  {metrics['adapted_perplexity']:.2f}")
        print(f"Improvement:  {metrics['improvement']:+.2f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
