"""GRPO training on GSM8K — end-to-end RL for math reasoning.

Uses Qwen3-0.6B as the base model. GRPO (Group Relative Policy Optimization)
with binary reward: 1 if the extracted numeric answer matches ground truth, 0 otherwise.

Usage:
    python train_grpo_gsm8k.py --model_path /path/to/qwen3-0.6b \
                                --output_dir /path/to/checkpoints

Dependencies:
    - qwen3.py, optim.py, grpo.py (this project)
    - transformers (tokenizer only — no torch model)
    - datasets (HuggingFace datasets for GSM8K)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

try:
    import nvtx
except ImportError:
    # Stub if nvtx not installed
    from contextlib import contextmanager
    class _NvtxStub:
        @staticmethod
        def annotate(name="", color=""):
            @contextmanager
            def _noop():
                yield
            return _noop()
    nvtx = _NvtxStub()

# Backend selection — mirrors qwen3.py pattern
_BACKEND = os.environ.get("TILERL_BACKEND", "").lower()

if _BACKEND == "numpy":
    import numpy as cp  # type: ignore[assignment]
else:
    try:
        import cupy as cp
    except ModuleNotFoundError:
        import numpy as cp  # type: ignore[assignment]

from qwen3 import Qwen3Config, Qwen3Model, precompute_rope
from optim import AdamW, AdamWConfig, clip_grad_norm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """GRPO training hyperparameters for GSM8K."""

    # Model
    model_path: str = ""
    model_size: str = "0.6b"  # "0.6b", "1.7b", "4b"

    # GRPO
    group_size: int = 8          # completions per prompt
    clip_eps: float = 0.2        # PPO/GRPO clipping epsilon
    beta: float = 0.04           # KL penalty coefficient
    gamma: float = 1.0           # reward discount (1.0 for bandit)

    # Optimization
    lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 50
    total_steps: int = 500
    batch_size: int = 4          # prompts per step (× group_size = total sequences)
    accumulation_steps: int = 1  # micro-steps per optimizer step (gradient accumulation)

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50

    # Logging / checkpointing
    log_every: int = 10
    eval_every: int = 50
    save_every: int = 100
    output_dir: str = "./grpo_checkpoints"
    eval_samples: int = 200      # subset of test set for fast eval during training
    resume: str | None = None     # checkpoint dir to resume from (model.npz)


# ---------------------------------------------------------------------------
# GSM8K data loading + answer extraction
# ---------------------------------------------------------------------------


def load_gsm8k() -> tuple[list[dict], list[dict]]:
    """Load GSM8K train and test splits.

    Returns:
        (train_data, test_data) where each element is
        {"question": str, "answer": str, "gold": str}.
        gold is the extracted numeric answer.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main")
        train_raw = list(ds["train"])
        test_raw = list(ds["test"])
    except ImportError:
        raise RuntimeError(
            "Install datasets: pip install datasets"
        )

    def _process(examples: list[dict]) -> list[dict]:
        out = []
        for ex in examples:
            gold = extract_answer(ex["answer"])
            out.append({
                "question": ex["question"],
                "answer": ex["answer"],
                "gold": gold,
            })
        return out

    return _process(train_raw), _process(test_raw)


def extract_answer(answer_text: str) -> str:
    """Extract the numeric answer after #### from GSM8K answer text."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return ""


def extract_model_answer(response: str) -> str:
    """Extract the final numeric answer from model's generated response.

    Looks for patterns like:
    - #### <number>
    - The answer is <number>
    - = <number> (at end of line)
    - Boxed answers: \\boxed{<number>}
    """
    # Try #### pattern first (trained models)
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", response)
    if match:
        return match.group(1).replace(",", "").strip()

    # Try \boxed{} pattern
    match = re.search(r"\\boxed\{(-?[\d,]+\.?\d*)\}", response)
    if match:
        return match.group(1).replace(",", "").strip()

    # Try "the answer is X" pattern
    match = re.search(r"[Tt]he\s+answer\s+is\s+(-?[\d,]+\.?\d*)", response)
    if match:
        return match.group(1).replace(",", "").strip()

    # Last resort: find the last number in the response
    numbers = re.findall(r"-?\d+\.?\d*", response)
    if numbers:
        return numbers[-1].strip()

    return ""


def check_answer(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    try:
        pred_val = float(predicted.replace(",", ""))
        gold_val = float(gold.replace(",", ""))
        return abs(pred_val - gold_val) < 1e-5
    except (ValueError, TypeError):
        return predicted.strip() == gold.strip()


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_prompt(question: str) -> str:
    """Format a GSM8K question into a chat-style prompt for Qwen3."""
    return (
        "<|im_start|>system\n"
        "You are a helpful math assistant. Solve the problem step by step. "
        "End your response with #### followed by the numeric answer.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Generation (autoregressive decode)
# ---------------------------------------------------------------------------


def generate(
    model: Qwen3Model,
    input_ids: cp.ndarray,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    eos_token_id: int = 151645,  # Qwen3 <|im_end|>
    pad_token_id: int = 151643,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Batched autoregressive generation with KV cache.

    Uses model.forward_inference() for O(T) per step instead of O(T²).
    Prefills the prompt in one pass, then decodes one token at a time.

    Args:
        model: Qwen3Model instance.
        input_ids: [B, T] prompt token IDs.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_k: Top-k for sampling.
        eos_token_id: Stop token.
        pad_token_id: Padding token.

    Returns:
        (full_ids, response_mask): full_ids is [B, T+gen_len],
        response_mask is [B, T+gen_len] with 1s for generated tokens.
    """
    xp = cp
    B, T = input_ids.shape
    max_len = T + max_new_tokens

    # Pre-allocate KV cache and RoPE tables
    kv_cache = model._alloc_kv_cache(B, max_len)
    rope_cos, rope_sin = precompute_rope(
        max_len, model.cfg.head_dim, model.cfg.rope_theta,
    )
    if xp is not np:
        rope_cos = xp.asarray(rope_cos)
        rope_sin = xp.asarray(rope_sin)

    # Build output buffer
    all_ids = xp.full((B, max_len), pad_token_id, dtype=xp.int32)
    all_ids[:, :T] = input_ids
    response_mask = xp.zeros((B, max_len), dtype=xp.float32)
    finished = xp.zeros(B, dtype=bool)

    # Prefill: process entire prompt at once
    logits = model.forward_inference(
        input_ids, kv_cache, seq_len=0,
        rope_cos=rope_cos, rope_sin=rope_sin,
    )  # [B, T, V]
    next_logits = logits[:, -1, :].astype(xp.float32)  # [B, V]
    seq_len = T

    for step in range(max_new_tokens):
        cur_pos = T + step

        # Sample next token
        if temperature <= 0 or temperature < 1e-8:
            next_tokens = xp.argmax(next_logits, axis=-1).astype(xp.int32)
        else:
            scaled = next_logits / temperature
            if top_k > 0:
                topk_vals = xp.sort(scaled, axis=-1)[:, -top_k:]
                threshold = topk_vals[:, 0:1]
                scaled = xp.where(scaled >= threshold, scaled, -1e30)
            probs = _softmax(scaled)
            next_tokens = _multinomial(probs).astype(xp.int32)

        # Place tokens, mark mask, check EOS
        all_ids[:, cur_pos] = xp.where(finished, pad_token_id, next_tokens)
        response_mask[:, cur_pos] = xp.where(finished, 0.0, 1.0)
        finished = finished | (next_tokens == eos_token_id)

        if bool(finished.all()):
            break

        # Decode: single new token with KV cache
        new_tok = next_tokens.reshape(B, 1)
        logits = model.forward_inference(
            new_tok, kv_cache, seq_len=seq_len,
            rope_cos=rope_cos, rope_sin=rope_sin,
        )  # [B, 1, V]
        next_logits = logits[:, 0, :].astype(xp.float32)
        seq_len += 1

    # Trim to actual length
    actual_len = T + step + 1 if step < max_new_tokens - 1 else max_len
    return all_ids[:, :actual_len], response_mask[:, :actual_len]


def generate_greedy(
    model: Qwen3Model,
    input_ids: cp.ndarray,
    max_new_tokens: int = 512,
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Greedy generation (temperature=0)."""
    return generate(
        model, input_ids, max_new_tokens,
        temperature=0.0, top_k=0,
        eos_token_id=eos_token_id, pad_token_id=pad_token_id,
    )


def _softmax(x: cp.ndarray) -> cp.ndarray:
    """Numerically stable softmax along last axis."""
    x_max = x.max(axis=-1, keepdims=True)
    e = cp.exp(x - x_max)
    return e / e.sum(axis=-1, keepdims=True)


def _multinomial(probs: cp.ndarray) -> cp.ndarray:
    """Sample one token per row from probability distribution."""
    B, V = probs.shape
    cumprobs = cp.cumsum(probs, axis=-1)
    r = cp.random.uniform(0, 1, size=(B, 1))
    # Find first index where cumprob >= r
    tokens = (cumprobs < r).sum(axis=-1).astype(cp.int32)
    tokens = cp.minimum(tokens, V - 1)
    return tokens


# ---------------------------------------------------------------------------
# Reference model log-probs (memory-efficient, no activation caching)
# ---------------------------------------------------------------------------


def gather_ref_logprobs(
    model: Qwen3Model, token_ids: cp.ndarray, response_mask: cp.ndarray,
) -> cp.ndarray:
    """Get per-token log-probs from a reference model using forward_inference.

    Uses forward_inference() instead of forward() to avoid caching activations
    (~13 GB for 28-layer model at B=8, T=562). Only allocates a KV cache (~1 GB)
    which is freed immediately after.

    Args:
        model: Reference model (frozen, no backward needed).
        token_ids: [B, T] input token IDs.
        response_mask: [B, T] mask for response tokens.

    Returns:
        log_probs: [B, T-1] masked per-token log-probs.
    """
    xp = cp
    B, T = token_ids.shape

    # Allocate temporary KV cache for the full sequence
    kv_cache = model._alloc_kv_cache(B, T)
    rope_cos, rope_sin = precompute_rope(T, model.cfg.head_dim, model.cfg.rope_theta)
    if xp is not np:
        rope_cos = xp.asarray(rope_cos)
        rope_sin = xp.asarray(rope_sin)

    # Full prefill — no activation caching, just KV cache
    logits = model.forward_inference(
        token_ids, kv_cache, seq_len=0,
        rope_cos=rope_cos, rope_sin=rope_sin,
    )  # [B, T, V]

    # Free KV cache immediately (~1 GB)
    del kv_cache, rope_cos, rope_sin

    # Compute log-probs at target positions
    from grpo import log_softmax_gather

    V = logits.shape[2]
    shift_logits = logits[:, :-1, :].reshape(B * (T - 1), V)
    del logits  # Free full logits (~2.7 GB)
    shift_targets = token_ids[:, 1:].reshape(B * (T - 1)).astype(xp.int32)

    lp_flat = log_softmax_gather(shift_logits, shift_targets)
    del shift_logits  # Free after log_softmax_gather

    mask = response_mask[:, 1:].astype(xp.float32)
    return lp_flat.reshape(B, T - 1) * mask


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: Qwen3Model,
    tokenizer,
    test_data: list[dict],
    max_samples: int = 0,
    max_new_tokens: int = 512,
    batch_size: int = 4,
) -> dict:
    """Evaluate model on GSM8K test set with greedy decoding.

    Returns:
        {"accuracy": float, "correct": int, "total": int, "examples": list}
    """
    if max_samples > 0:
        test_data = test_data[:max_samples]

    correct = 0
    total = 0
    examples = []

    for i in range(0, len(test_data), batch_size):
        batch = test_data[i : i + batch_size]
        prompts = [format_prompt(ex["question"]) for ex in batch]
        prompt_ids = [tokenizer.encode(p) for p in prompts]

        # Pad to same length
        max_prompt_len = max(len(ids) for ids in prompt_ids)
        padded = cp.full(
            (len(prompt_ids), max_prompt_len),
            tokenizer.pad_token_id or 0,
            dtype=cp.int32,
        )
        for j, ids in enumerate(prompt_ids):
            padded[j, max_prompt_len - len(ids):] = cp.array(ids, dtype=cp.int32)

        # Generate
        full_ids, _ = generate_greedy(
            model, padded, max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id or 151645,
        )

        # Decode and score
        for j, ex in enumerate(batch):
            gen_ids = full_ids[j, max_prompt_len:].get().tolist()
            # Remove padding
            gen_ids = [t for t in gen_ids if t != (tokenizer.pad_token_id or 0)]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            predicted = extract_model_answer(response)
            is_correct = check_answer(predicted, ex["gold"])

            correct += int(is_correct)
            total += 1

            if len(examples) < 5:
                examples.append({
                    "question": ex["question"][:100],
                    "gold": ex["gold"],
                    "predicted": predicted,
                    "correct": is_correct,
                })

    accuracy = correct / max(total, 1)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# GRPO training loop
# ---------------------------------------------------------------------------


def compute_rewards(
    model: Qwen3Model,
    tokenizer,
    questions: list[dict],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Generate completions and compute binary rewards.

    For each question, generate group_size completions.

    Returns:
        input_ids: [B*G, T] padded token IDs
        response_mask: [B*G, T] mask for generated tokens
        rewards: [B*G] binary rewards
    """
    B = len(questions)
    G = group_size

    all_input_ids = []
    all_response_masks = []
    all_rewards = []

    for q_idx, ex in enumerate(questions):
        prompt = format_prompt(ex["question"])
        prompt_ids = tokenizer.encode(prompt)
        prompt_len = len(prompt_ids)

        # Create batch of G identical prompts
        prompt_batch = cp.array(
            [prompt_ids] * G, dtype=cp.int32
        )  # [G, prompt_len]

        # Generate G completions
        full_ids, resp_mask = generate(
            model, prompt_batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        # Score each completion
        rewards = cp.zeros(G, dtype=cp.float32)
        for g in range(G):
            gen_ids = full_ids[g, prompt_len:].get().tolist()
            gen_ids = [t for t in gen_ids if t != (tokenizer.pad_token_id or 0)]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            predicted = extract_model_answer(response)
            if check_answer(predicted, ex["gold"]):
                rewards[g] = 1.0

        all_input_ids.append(full_ids)
        all_response_masks.append(resp_mask)
        all_rewards.append(rewards)

    # Pad all sequences to same length and stack
    max_seq_len = max(ids.shape[1] for ids in all_input_ids)
    padded_ids = cp.zeros((B * G, max_seq_len), dtype=cp.int32)
    padded_mask = cp.zeros((B * G, max_seq_len), dtype=cp.float32)

    for i, (ids, mask) in enumerate(zip(all_input_ids, all_response_masks)):
        seq_len = ids.shape[1]
        padded_ids[i * G : (i + 1) * G, :seq_len] = ids
        padded_mask[i * G : (i + 1) * G, :seq_len] = mask

    rewards = cp.concatenate(all_rewards)  # [B*G]
    return padded_ids, padded_mask, rewards


def train(cfg: TrainConfig):
    """Main GRPO training loop."""
    print("=" * 60)
    print("GRPO Training on GSM8K")
    print("=" * 60)

    # --- Setup tokenizer ---
    print("[1/5] Loading tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, trust_remote_code=True
        )
    except ImportError:
        raise RuntimeError("Install transformers: pip install transformers")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Load model ---
    print("[2/5] Loading model...")
    model_configs = {
        "0.6b": Qwen3Config.qwen3_0_6b,
        "1.7b": Qwen3Config.qwen3_1_7b,
        "4b": Qwen3Config.qwen3_4b,
    }
    model_cfg = model_configs[cfg.model_size]()
    model = Qwen3Model(model_cfg)
    model.load_weights(cfg.model_path)
    print(f"  Model: Qwen3-{cfg.model_size}, {sum(p.size for p in model.params.values()):,} params")

    # Reference model (frozen copy from original HF weights — stays at baseline)
    print("  Creating reference model (frozen copy)...")
    ref_model = Qwen3Model(model_cfg)
    for name, param in model.params.items():
        ref_model.params[name] = param.copy()

    # Resume from checkpoint (overwrite policy model, keep ref at baseline)
    if cfg.resume:
        print(f"  Resuming policy model from {cfg.resume}...")
        load_checkpoint(model, cfg.resume)
        print(f"  Policy model loaded from checkpoint (ref model stays at baseline)")
        # Skip warmup — model is already at a good optimum, warmup would
        # cause a J-curve dip as near-zero lr destabilizes greedy eval
        cfg.warmup_steps = 0
        print(f"  Warmup disabled for resume (starting at full lr={cfg.lr})")

    # --- Optimizer ---
    print("[3/5] Setting up optimizer...")
    optim_cfg = AdamWConfig(
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    optimizer = AdamW(model.params, optim_cfg)

    # --- Data ---
    print("[4/5] Loading GSM8K...")
    train_data, test_data = load_gsm8k()
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    # --- Baseline eval ---
    print("[5/5] Running baseline evaluation...")
    baseline = evaluate(
        model, tokenizer, test_data,
        max_samples=cfg.eval_samples,
        batch_size=cfg.batch_size,
    )
    print(f"  Baseline accuracy: {baseline['accuracy']:.1%} "
          f"({baseline['correct']}/{baseline['total']})")
    for ex in baseline["examples"][:3]:
        print(f"    Q: {ex['question']}")
        print(f"    Gold: {ex['gold']}, Pred: {ex['predicted']}, "
              f"{'✓' if ex['correct'] else '✗'}")

    # --- Training ---
    print("\n" + "=" * 60)
    print("Starting GRPO training")
    print(f"  Steps: {cfg.total_steps}, Batch: {cfg.batch_size}, "
          f"Group: {cfg.group_size}, Accum: {cfg.accumulation_steps}")
    print(f"  Effective episodes/step: {cfg.accumulation_steps * cfg.group_size}")
    print(f"  LR: {cfg.lr}, Clip: {cfg.clip_eps}, Beta: {cfg.beta}, "
          f"Temp: {cfg.temperature}")
    print("=" * 60)

    os.makedirs(cfg.output_dir, exist_ok=True)
    history = []
    rng = np.random.RandomState(42)

    # Import GRPO wrapper functions (auto-dispatch cuTile vs CuPy/NumPy)
    from grpo import (
        compute_advantages, clipped_surrogate_fwd, kl_fwd,
        clipped_surrogate_bwd, kl_bwd, log_softmax_gather_bwd,
        gather_logprobs,
    )

    K = cfg.accumulation_steps  # micro-steps per optimizer step

    for step in range(cfg.total_steps):
        t0 = time.time()

        # Learning rate warmup
        if step < cfg.warmup_steps:
            lr = cfg.lr * (step + 1) / cfg.warmup_steps
            optimizer.set_lr(lr)
        else:
            lr = cfg.lr

        # --- Gradient accumulation over K micro-steps ---
        # Each micro-step: sample 1 question, generate G completions, forward,
        # backward. Accumulate gradients, average before optimizer step.
        # This gives effective batch = K questions × G completions per step.
        accumulated_grads: dict[str, cp.ndarray] | None = None
        total_loss = 0.0
        total_kl = 0.0
        total_reward = 0.0

        for micro in range(K):
            # Sample 1 question per micro-step
            idx = rng.choice(len(train_data))
            batch_questions = [train_data[idx]]

            # Generate completions + compute rewards
            with nvtx.annotate("generation", color="blue"):
                input_ids, response_mask, rewards = compute_rewards(
                    model, tokenizer, batch_questions,
                    cfg.group_size, cfg.max_new_tokens,
                    cfg.temperature, cfg.top_k,
                )

            B = 1  # 1 question per micro-step
            G = cfg.group_size
            N = B * G

            # Compute group-normalized advantages
            reward_groups = rewards.reshape(B, G)
            advantages = compute_advantages(reward_groups, G).reshape(N)

            # Free GPU memory from generation phase
            if hasattr(cp, 'get_default_memory_pool'):
                cp.get_default_memory_pool().free_all_blocks()

            # --- Forward passes for log-probs ---
            with nvtx.annotate("ref_forward", color="green"):
                ref_lp = gather_ref_logprobs(ref_model, input_ids, response_mask)
            if hasattr(cp, 'get_default_memory_pool'):
                cp.get_default_memory_pool().free_all_blocks()

            with nvtx.annotate("policy_forward", color="orange"):
                new_lp, logits_2d, mask = gather_logprobs(
                    model, input_ids, response_mask, recompute_attn=True,
                )
                old_lp = new_lp.copy()

            T1 = new_lp.shape[1]

            with nvtx.annotate("loss_computation", color="red"):
                # Ratios (all 1.0 for single-epoch GRPO)
                ratios = cp.exp(new_lp - old_lp)
                adv_tok = cp.broadcast_to(advantages[:, None], (N, T1)).copy()

                # Clipped surrogate + KL
                surrogate = clipped_surrogate_fwd(ratios, adv_tok, cfg.clip_eps)
                kl = kl_fwd(new_lp, ref_lp)

                # Loss
                per_token_loss = -surrogate + cfg.beta * kl
                total_tokens = float(mask.sum())
                micro_loss = float((per_token_loss * mask).sum() / max(total_tokens, 1.0))
                micro_kl = float((kl * mask).sum() / max(total_tokens, 1.0))
                total_loss += micro_loss
                total_kl += micro_kl
                total_reward += float(rewards.mean())

            # --- Backward ---
            with nvtx.annotate("backward", color="purple"):
                grad_ptl = mask / max(total_tokens, 1.0)
                grad_ratios = clipped_surrogate_bwd(ratios, adv_tok, -grad_ptl, cfg.clip_eps)
                grad_new_lp_kl = kl_bwd(new_lp, ref_lp, cfg.beta * grad_ptl)
                grad_new_lp = grad_ratios * ratios + grad_new_lp_kl

                if hasattr(cp, 'get_default_memory_pool'):
                    cp.get_default_memory_pool().free_all_blocks()

                # Log-softmax-gather backward
                V = logits_2d.shape[2]
                shift_logits = logits_2d.reshape(N * T1, V).astype(cp.float32)
                shift_targets = input_ids[:, 1:].reshape(N * T1).astype(cp.int32)
                grad_logits_flat = log_softmax_gather_bwd(
                    shift_logits, shift_targets, grad_new_lp.reshape(N * T1),
                )

                full_grad = cp.zeros((N, input_ids.shape[1], V), dtype=cp.float32)
                full_grad[:, 1:, :] = grad_logits_flat.reshape(N, T1, V)

                grads = model.backward(full_grad)

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                for k in accumulated_grads:
                    accumulated_grads[k] += grads[k]
                del grads

            # Free everything for next micro-step (or optimizer step)
            model._cache = {}
            del full_grad, grad_logits_flat, grad_new_lp, grad_new_lp_kl
            del grad_ratios, grad_ptl
            del shift_logits, shift_targets, logits_2d
            del ref_lp, old_lp, ratios, adv_tok, surrogate, kl, per_token_loss
            if hasattr(cp, 'get_default_memory_pool'):
                cp.get_default_memory_pool().free_all_blocks()

        # Average gradients over micro-steps
        if K > 1:
            for k in accumulated_grads:
                accumulated_grads[k] /= K

        # Gradient clipping + optimizer step
        with nvtx.annotate("optimizer_step", color="cyan"):
            grad_norm = clip_grad_norm(accumulated_grads, cfg.grad_clip)
            optimizer.step(accumulated_grads)

        # Free grads + optimizer temporaries
        del accumulated_grads
        if hasattr(cp, 'get_default_memory_pool'):
            cp.get_default_memory_pool().free_all_blocks()

        dt = time.time() - t0
        loss_val = total_loss / K
        kl_val = total_kl / K
        mean_reward = total_reward / K

        metrics = {
            "step": step,
            "loss": loss_val,
            "kl": kl_val,
            "mean_reward": mean_reward,
            "grad_norm": grad_norm,
            "lr": lr,
            "time": dt,
        }
        history.append(metrics)

        if step % cfg.log_every == 0:
            print(
                f"[step {step:>4d}] loss={loss_val:.4f}  "
                f"reward={mean_reward:.3f}  kl={kl_val:.4f}  "
                f"gnorm={grad_norm:.2f}  lr={lr:.2e}  "
                f"time={dt:.1f}s"
            )

        # Periodic evaluation
        if (step + 1) % cfg.eval_every == 0:
            print(f"\n--- Eval at step {step + 1} ---")
            eval_result = evaluate(
                model, tokenizer, test_data,
                max_samples=cfg.eval_samples,
                batch_size=cfg.batch_size,
            )
            print(f"  Accuracy: {eval_result['accuracy']:.1%} "
                  f"({eval_result['correct']}/{eval_result['total']})")
            for ex in eval_result["examples"][:2]:
                print(f"    Gold={ex['gold']}, Pred={ex['predicted']}, "
                      f"{'✓' if ex['correct'] else '✗'}")
            history[-1]["eval_accuracy"] = eval_result["accuracy"]
            print()

        # Periodic checkpoint
        if (step + 1) % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.output_dir, f"step_{step + 1}")
            os.makedirs(ckpt_path, exist_ok=True)
            _save_checkpoint(model, optimizer, step + 1, history, ckpt_path)
            print(f"  Checkpoint saved to {ckpt_path}")

    # --- Final evaluation ---
    print("\n" + "=" * 60)
    print("Final evaluation on full test set")
    print("=" * 60)
    final_samples = cfg.eval_samples if cfg.eval_samples > 0 else 0
    final_eval = evaluate(
        model, tokenizer, test_data,
        max_samples=final_samples,
        batch_size=cfg.batch_size,
    )
    print(f"  Final accuracy: {final_eval['accuracy']:.1%} "
          f"({final_eval['correct']}/{final_eval['total']})")
    print(f"  Baseline was:   {baseline['accuracy']:.1%}")
    print(f"  Improvement:    {final_eval['accuracy'] - baseline['accuracy']:+.1%}")

    # Save final model
    final_path = os.path.join(cfg.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    _save_checkpoint(model, optimizer, cfg.total_steps, history, final_path)

    # Save training history
    with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Checkpoints in {cfg.output_dir}")
    return history, final_eval


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _save_checkpoint(
    model: Qwen3Model,
    optimizer: AdamW,
    step: int,
    history: list,
    path: str,
) -> None:
    """Save model parameters and optimizer state."""
    # Save model params as .npz
    params_np = {k: v.get() for k, v in model.params.items()}
    np.savez(os.path.join(path, "model.npz"), **params_np)

    # Save metadata
    meta = {
        "step": step,
        "config": {
            "hidden_size": model.cfg.hidden_size,
            "num_hidden_layers": model.cfg.num_hidden_layers,
            "vocab_size": model.cfg.vocab_size,
        },
        "last_metrics": history[-1] if history else {},
    }
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(
    model: Qwen3Model, path: str
) -> None:
    """Load model parameters from checkpoint."""
    data = np.load(os.path.join(path, "model.npz"))
    for name in model.params:
        if name in data:
            model.params[name] = cp.array(data[name])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GRPO training on GSM8K")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to Qwen3 weights (safetensors dir)")
    parser.add_argument("--model_size", type=str, default="0.6b",
                        choices=["0.6b", "1.7b", "4b"])
    parser.add_argument("--total_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation micro-steps per optimizer step")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output_dir", type=str, default="./grpo_checkpoints")
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint dir (with model.npz) to resume from")
    args = parser.parse_args()

    cfg = TrainConfig(
        model_path=args.model_path,
        model_size=args.model_size,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        accumulation_steps=args.accumulation_steps,
        lr=args.lr,
        beta=args.beta,
        temperature=args.temperature,
        output_dir=args.output_dir,
        eval_samples=args.eval_samples,
        max_new_tokens=args.max_new_tokens,
        resume=args.resume,
    )

    train(cfg)


if __name__ == "__main__":
    main()
