"""DAPO — Dynamic Actor-Policy Optimization.

Paper: "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
       (ByteDance, 2025)

Extends GRPO with four innovations:
1. Clip-Higher: asymmetric clip [1-ε_low, 1+ε_high], ε_high > ε_low
2. Dynamic Sampling: filter zero-variance groups (all correct/incorrect)
3. Token-Level Policy Gradient: per-token loss, not per-sequence
4. Overlong Reward Shaping: soft penalty for exceeding max length

Objective (per token t in completion i):
    L = -E[ min(ρ_t · Â_i, clip(ρ_t, 1-ε_low, 1+ε_high) · Â_i) ] + β · KL_t
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Backend selection — mirrors grpo.py pattern
_BACKEND = os.environ.get("TILERL_BACKEND", "").lower()

if _BACKEND == "numpy":
    import numpy as cp  # type: ignore[assignment]
else:
    try:
        import cupy as cp
    except ModuleNotFoundError:
        import numpy as cp  # type: ignore[assignment]

try:
    import cuda.tile as ct
    _HAS_CUTILE = True
except ModuleNotFoundError:
    import types
    ct = types.SimpleNamespace(
        Constant=dict, kernel=lambda f=None, **kw: f if f else (lambda g: g),
        PaddingMode=types.SimpleNamespace(ZERO="zero", NEG_INF="neg_inf"),
    )
    _HAS_CUTILE = False

_USE_CUTILE_KERNELS = _HAS_CUTILE and _BACKEND == "cutile"

ConstInt = ct.Constant[int] if _HAS_CUTILE else int

from qwen3 import Qwen3Config, Qwen3Model
from optim import AdamW

if _HAS_CUTILE:
    PAD_ZERO = ct.PaddingMode.ZERO

# ---------------------------------------------------------------------------
# cuTile kernels — forward
# ---------------------------------------------------------------------------


@ct.kernel
def log_softmax_gather_kernel(
    logits: ct.Array, targets: ct.Array, output: ct.Array,
    V: ConstInt, TILE_V: ConstInt,
):
    """Per-token log-softmax + gather at target index."""
    pid = ct.bid(0)
    num_tiles = ct.num_tiles(logits, axis=1, shape=(1, TILE_V))

    row_max = ct.full((1, TILE_V), -1e30, dtype=ct.float32)
    for j in range(num_tiles):
        tile = ct.load(logits, index=(pid, j), shape=(1, TILE_V), padding_mode=PAD_ZERO)
        row_max = ct.maximum(row_max, tile)
    row_max = ct.max(row_max, axis=1)

    sum_exp = ct.full((1, TILE_V), 0.0, dtype=ct.float32)
    for j in range(num_tiles):
        tile = ct.load(logits, index=(pid, j), shape=(1, TILE_V), padding_mode=PAD_ZERO)
        vmask = (j * TILE_V + ct.arange(TILE_V, dtype=ct.int32)) < V
        sum_exp += ct.where(vmask, ct.exp(tile - row_max), 0.0)
    log_sum_exp = ct.log(ct.sum(sum_exp, axis=1)) + row_max

    target = ct.load(targets, index=(pid,), shape=(1,))
    t_tile = target / TILE_V
    t_off = target % TILE_V
    lt = ct.load(logits, index=(pid, t_tile), shape=(1, TILE_V), padding_mode=PAD_ZERO)
    offsets = ct.arange(TILE_V, dtype=ct.int32)
    tgt_logit = ct.sum(ct.where(offsets == t_off, lt, 0.0), axis=1)
    ct.store(output, index=(pid,), tile=tgt_logit - log_sum_exp)


@ct.kernel
def dapo_group_advantage_kernel(
    rewards: ct.Array, advantages: ct.Array, valid_mask: ct.Array,
    G: ConstInt,
):
    """Group-normalize rewards + dynamic sampling.

    Sets valid_mask=0 for groups with zero variance (all same reward),
    effectively filtering them from the loss computation.
    """
    pid = ct.bid(0)
    r = ct.load(rewards, index=(pid, 0), shape=(1, G))
    mean = ct.sum(r, axis=1) / G
    diff = r - mean
    var = ct.sum(diff * diff, axis=1) / G

    # Dynamic sampling: mask out zero-variance groups
    is_valid = ct.where(var > 1e-6, 1.0, 0.0)
    ct.store(valid_mask, index=(pid,), tile=is_valid)

    std = ct.sqrt(var + 1e-8)
    adv = diff / std * is_valid  # zero out advantages for filtered groups
    ct.store(advantages, index=(pid, 0), tile=adv)


@ct.kernel
def dapo_overlong_penalty_kernel(
    rewards: ct.Array, lengths: ct.Array, max_len: ct.Array,
    penalty: ct.Array, output: ct.Array, TILE: ConstInt,
):
    """Overlong reward shaping: r -= penalty * max(0, len-max_len) / max_len."""
    pid = ct.bid(0)
    r = ct.load(rewards, index=(pid,), shape=(TILE,))
    l = ct.load(lengths, index=(pid,), shape=(TILE,))  # noqa: E741
    ml = ct.load(max_len, index=(0,), shape=(1,))
    p = ct.load(penalty, index=(0,), shape=(1,))

    excess = ct.maximum(l - ml, 0.0) / ml
    ct.store(output, index=(pid,), tile=r - p * excess)


@ct.kernel
def dapo_clip_higher_fwd_kernel(
    ratios: ct.Array, advantages: ct.Array, output: ct.Array,
    eps_low: ct.Array, eps_high: ct.Array, TILE: ConstInt,
):
    """Asymmetric clipped surrogate: clip(ρ, 1-ε_low, 1+ε_high)."""
    bid = ct.bid(0)
    num_tiles = ct.num_tiles(ratios, axis=1, shape=(1, TILE))
    el = ct.load(eps_low, index=(0,), shape=(1,))
    eh = ct.load(eps_high, index=(0,), shape=(1,))

    for j in range(num_tiles):
        r = ct.load(ratios, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        a = ct.load(advantages, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        clipped = ct.minimum(ct.maximum(r, 1.0 - el), 1.0 + eh)
        ct.store(output, index=(bid, j), tile=ct.minimum(r * a, clipped * a))


@ct.kernel
def dapo_kl_fwd_kernel(
    new_lp: ct.Array, ref_lp: ct.Array, kl_out: ct.Array, TILE: ConstInt,
):
    """Schulman KL approximation: exp(ref-new) - (ref-new) - 1."""
    bid = ct.bid(0)
    num_tiles = ct.num_tiles(new_lp, axis=1, shape=(1, TILE))
    for j in range(num_tiles):
        n = ct.load(new_lp, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        r = ct.load(ref_lp, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        log_ratio = r - n
        ct.store(kl_out, index=(bid, j), tile=ct.exp(log_ratio) - log_ratio - 1.0)


# ---------------------------------------------------------------------------
# cuTile kernels — backward
# ---------------------------------------------------------------------------


@ct.kernel
def dapo_clip_higher_bwd_kernel(
    ratios: ct.Array, advantages: ct.Array, grad_out: ct.Array,
    grad_ratios: ct.Array, eps_low: ct.Array, eps_high: ct.Array,
    TILE: ConstInt,
):
    """Backward of asymmetric clipped surrogate w.r.t. ratios."""
    bid = ct.bid(0)
    num_tiles = ct.num_tiles(ratios, axis=1, shape=(1, TILE))
    el = ct.load(eps_low, index=(0,), shape=(1,))
    eh = ct.load(eps_high, index=(0,), shape=(1,))

    for j in range(num_tiles):
        r = ct.load(ratios, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        a = ct.load(advantages, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        g = ct.load(grad_out, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        clipped = ct.minimum(ct.maximum(r, 1.0 - el), 1.0 + eh)
        use_clipped = ct.where((clipped * a) < (r * a), 1.0, 0.0)
        cond_lo = ct.where(r >= 1.0 - el, 1.0, 0.0)
        cond_hi = ct.where(r <= 1.0 + eh, 1.0, 0.0)
        in_range = cond_lo * cond_hi
        grad = ct.where(use_clipped > 0.5, in_range * a, a) * g
        ct.store(grad_ratios, index=(bid, j), tile=grad)


@ct.kernel
def dapo_kl_bwd_kernel(
    new_lp: ct.Array, ref_lp: ct.Array, grad_out: ct.Array,
    grad_new_lp: ct.Array, TILE: ConstInt,
):
    """Backward of KL w.r.t. new log-probs: -exp(ref-new) + 1."""
    bid = ct.bid(0)
    num_tiles = ct.num_tiles(new_lp, axis=1, shape=(1, TILE))
    for j in range(num_tiles):
        n = ct.load(new_lp, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        r = ct.load(ref_lp, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        g = ct.load(grad_out, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        ct.store(grad_new_lp, index=(bid, j), tile=(-ct.exp(r - n) + 1.0) * g)


@ct.kernel
def log_softmax_gather_bwd_kernel(
    logits: ct.Array, targets: ct.Array, grad_lp: ct.Array,
    grad_logits: ct.Array, V: ConstInt, TILE_V: ConstInt,
):
    """Backward of log_softmax+gather: grad * (one_hot - softmax)."""
    pid = ct.bid(0)
    num_tiles = ct.num_tiles(logits, axis=1, shape=(1, TILE_V))
    g = ct.load(grad_lp, index=(pid,), shape=(1,))
    target = ct.load(targets, index=(pid,), shape=(1,))

    row_max = ct.full((1, TILE_V), -1e30, dtype=ct.float32)
    for j in range(num_tiles):
        tile = ct.load(logits, index=(pid, j), shape=(1, TILE_V), padding_mode=PAD_ZERO)
        row_max = ct.maximum(row_max, tile)
    row_max = ct.max(row_max, axis=1)

    sum_exp = ct.full((1, TILE_V), 0.0, dtype=ct.float32)
    for j in range(num_tiles):
        tile = ct.load(logits, index=(pid, j), shape=(1, TILE_V), padding_mode=PAD_ZERO)
        vmask = (j * TILE_V + ct.arange(TILE_V, dtype=ct.int32)) < V
        sum_exp += ct.where(vmask, ct.exp(tile - row_max), 0.0)
    total = ct.sum(sum_exp, axis=1)

    for j in range(num_tiles):
        tile = ct.load(logits, index=(pid, j), shape=(1, TILE_V), padding_mode=PAD_ZERO)
        vmask = (j * TILE_V + ct.arange(TILE_V, dtype=ct.int32)) < V
        sm = ct.where(vmask, ct.exp(tile - row_max) / total, 0.0)
        offsets = j * TILE_V + ct.arange(TILE_V, dtype=ct.int32)
        oh = ct.where(offsets == target, 1.0, 0.0)
        ct.store(grad_logits, index=(pid, j), tile=g * (oh - sm))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def _get_stream():
    return cp.cuda.get_current_stream()


# ---------------------------------------------------------------------------
# Wrapper functions with CuPy/NumPy fallbacks
# ---------------------------------------------------------------------------


def log_softmax_gather(
    logits: cp.ndarray, targets: cp.ndarray,
) -> cp.ndarray:
    """Forward: per-token log-softmax + gather.

    Args:
        logits: (N, V) logits.
        targets: (N,) target indices.
    Returns:
        (N,) log-probs at target positions.
    """
    xp = cp
    N, V = logits.shape
    if _USE_CUTILE_KERNELS:
        TILE_V = min(1024, _next_pow2(V))
        out = xp.empty(N, dtype=xp.float32)
        ct.launch(_get_stream(), (N,), log_softmax_gather_kernel,
                  (logits, targets, out, V, TILE_V))
        return out
    # Fallback: stable log-softmax + gather
    logits_f = logits.astype(xp.float32)
    row_max = logits_f.max(axis=-1, keepdims=True)
    shifted = logits_f - row_max
    log_sum_exp = xp.log(xp.exp(shifted).sum(axis=-1)) + row_max.squeeze(-1)
    target_logits = logits_f[xp.arange(N), targets.astype(int)]
    return (target_logits - log_sum_exp).astype(xp.float32)


def log_softmax_gather_bwd(
    logits: cp.ndarray, targets: cp.ndarray, grad_lp: cp.ndarray,
) -> cp.ndarray:
    """Backward of log_softmax+gather: grad * (one_hot - softmax).

    Args:
        logits: (N, V) logits.
        targets: (N,) target indices.
        grad_lp: (N,) gradient w.r.t. log-probs.
    Returns:
        (N, V) gradient w.r.t. logits.
    """
    xp = cp
    N, V = logits.shape
    if _USE_CUTILE_KERNELS:
        TILE_V = min(1024, _next_pow2(V))
        grad_logits = xp.empty_like(logits)
        ct.launch(_get_stream(), (N,), log_softmax_gather_bwd_kernel,
                  (logits, targets, grad_lp, grad_logits, V, TILE_V))
        return grad_logits
    # Fallback
    logits_f = logits.astype(xp.float32)
    row_max = logits_f.max(axis=-1, keepdims=True)
    shifted = logits_f - row_max
    exp_shifted = xp.exp(shifted)
    softmax = exp_shifted / exp_shifted.sum(axis=-1, keepdims=True)
    one_hot = xp.zeros_like(logits_f)
    one_hot[xp.arange(N), targets.astype(int)] = 1.0
    return (grad_lp[:, None] * (one_hot - softmax)).astype(xp.float32)


def dapo_group_advantages(
    rewards: cp.ndarray, group_size: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Group-normalize rewards with dynamic sampling.

    Args:
        rewards: (B, G) reward matrix.
        group_size: G, number of completions per prompt.
    Returns:
        advantages: (B, G) normalized advantages (zero for filtered groups).
        valid_mask: (B,) float32, 1.0 if group has nonzero variance, else 0.0.
    """
    xp = cp
    B = rewards.shape[0]
    if _USE_CUTILE_KERNELS:
        adv = xp.empty_like(rewards)
        valid = xp.empty(B, dtype=xp.float32)
        ct.launch(_get_stream(), (B,), dapo_group_advantage_kernel,
                  (rewards, adv, valid, group_size))
        return adv, valid
    # Fallback
    mean = rewards.mean(axis=1, keepdims=True)
    diff = rewards - mean
    var = (diff * diff).mean(axis=1, keepdims=True)
    is_valid = (var.squeeze(1) > 1e-6).astype(xp.float32)
    std = xp.sqrt(var + 1e-8)
    adv = (diff / std * is_valid[:, None]).astype(xp.float32)
    return adv, is_valid


def dapo_overlong_penalty(
    rewards: cp.ndarray, lengths: cp.ndarray,
    max_len: float, penalty: float,
) -> cp.ndarray:
    """Overlong reward shaping: r -= penalty * max(0, len - max_len) / max_len.

    Args:
        rewards: (N,) reward values.
        lengths: (N,) sequence lengths.
        max_len: Maximum allowed response length.
        penalty: Penalty coefficient.
    Returns:
        (N,) shaped reward values.
    """
    xp = cp
    N = rewards.shape[0]
    if _USE_CUTILE_KERNELS:
        max_len_t = xp.array([max_len], dtype=xp.float32)
        penalty_t = xp.array([penalty], dtype=xp.float32)
        out = xp.empty_like(rewards)
        ct.launch(_get_stream(), (N,), dapo_overlong_penalty_kernel,
                  (rewards, lengths, max_len_t, penalty_t, out, 1))
        return out
    # Fallback
    rewards_f = rewards.astype(xp.float32)
    lengths_f = lengths.astype(xp.float32)
    excess = xp.maximum(lengths_f - max_len, 0.0) / max_len
    return (rewards_f - penalty * excess).astype(xp.float32)


def dapo_clip_higher_fwd(
    ratios: cp.ndarray, advantages: cp.ndarray,
    eps_low: float, eps_high: float,
) -> cp.ndarray:
    """Asymmetric clipped surrogate: min(ρ·A, clip(ρ, 1-ε_low, 1+ε_high)·A).

    Args:
        ratios: (N, T) policy ratios.
        advantages: (N, T) per-token advantages.
        eps_low: Lower clipping epsilon.
        eps_high: Upper clipping epsilon.
    Returns:
        (N, T) surrogate values.
    """
    xp = cp
    if _USE_CUTILE_KERNELS:
        N = ratios.shape[0]
        T1 = ratios.shape[1]
        TILE = min(1024, _next_pow2(T1))
        eps_low_t = xp.array([eps_low], dtype=xp.float32)
        eps_high_t = xp.array([eps_high], dtype=xp.float32)
        out = xp.empty_like(ratios)
        ct.launch(_get_stream(), (N,), dapo_clip_higher_fwd_kernel,
                  (ratios, advantages, out, eps_low_t, eps_high_t, TILE))
        return out
    # Fallback
    clipped = xp.clip(ratios, 1.0 - eps_low, 1.0 + eps_high)
    return xp.minimum(ratios * advantages, clipped * advantages).astype(xp.float32)


def dapo_clip_higher_bwd(
    ratios: cp.ndarray, advantages: cp.ndarray,
    grad_out: cp.ndarray, eps_low: float, eps_high: float,
) -> cp.ndarray:
    """Backward of asymmetric clipped surrogate w.r.t. ratios.

    Returns:
        (N, T) gradient w.r.t. ratios.
    """
    xp = cp
    if _USE_CUTILE_KERNELS:
        N = ratios.shape[0]
        T1 = ratios.shape[1]
        TILE = min(1024, _next_pow2(T1))
        eps_low_t = xp.array([eps_low], dtype=xp.float32)
        eps_high_t = xp.array([eps_high], dtype=xp.float32)
        grad_ratios = xp.empty_like(ratios)
        ct.launch(_get_stream(), (N,), dapo_clip_higher_bwd_kernel,
                  (ratios, advantages, grad_out, grad_ratios,
                   eps_low_t, eps_high_t, TILE))
        return grad_ratios
    # Fallback
    clipped = xp.clip(ratios, 1.0 - eps_low, 1.0 + eps_high)
    surr1 = ratios * advantages
    surr2 = clipped * advantages
    use_clipped = (surr2 < surr1).astype(xp.float32)
    in_range = (
        (ratios >= 1.0 - eps_low) & (ratios <= 1.0 + eps_high)
    ).astype(xp.float32)
    grad = xp.where(use_clipped > 0.5, in_range * advantages, advantages) * grad_out
    return grad.astype(xp.float32)


def dapo_kl_fwd(new_lp: cp.ndarray, ref_lp: cp.ndarray) -> cp.ndarray:
    """Schulman KL approx: exp(ref-new) - (ref-new) - 1.

    Args:
        new_lp: (N, T) new policy log-probs.
        ref_lp: (N, T) reference policy log-probs.
    Returns:
        (N, T) KL divergence estimates.
    """
    xp = cp
    if _USE_CUTILE_KERNELS:
        N = new_lp.shape[0]
        T1 = new_lp.shape[1]
        TILE = min(1024, _next_pow2(T1))
        kl = xp.empty_like(new_lp)
        ct.launch(_get_stream(), (N,), dapo_kl_fwd_kernel,
                  (new_lp, ref_lp, kl, TILE))
        return kl
    # Fallback
    log_ratio = ref_lp - new_lp
    return (xp.exp(log_ratio) - log_ratio - 1.0).astype(xp.float32)


def dapo_kl_bwd(
    new_lp: cp.ndarray, ref_lp: cp.ndarray, grad_out: cp.ndarray,
) -> cp.ndarray:
    """Backward of KL w.r.t. new log-probs: (-exp(ref-new) + 1) * grad_out.

    Returns:
        (N, T) gradient w.r.t. new_lp.
    """
    xp = cp
    if _USE_CUTILE_KERNELS:
        N = new_lp.shape[0]
        T1 = new_lp.shape[1]
        TILE = min(1024, _next_pow2(T1))
        grad_new = xp.empty_like(new_lp)
        ct.launch(_get_stream(), (N,), dapo_kl_bwd_kernel,
                  (new_lp, ref_lp, grad_out, grad_new, TILE))
        return grad_new
    # Fallback
    return ((-xp.exp(ref_lp - new_lp) + 1.0) * grad_out).astype(xp.float32)


def gather_logprobs(model: Qwen3Model, ids: cp.ndarray, mask: cp.ndarray):
    """Model forward → per-token log-probs masked to response tokens."""
    xp = cp
    logits = model.forward(ids)
    B, T, V = logits.shape
    flat = logits[:, :-1, :].reshape(B * (T - 1), V).astype(xp.float32)
    tgts = ids[:, 1:].reshape(B * (T - 1)).astype(xp.int32)

    lp_flat = log_softmax_gather(flat, tgts)

    m = mask[:, 1:].astype(xp.float32)
    return lp_flat.reshape(B, T - 1) * m, flat.reshape(B, T - 1, V), m


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class DAPOConfig:
    group_size: int = 8
    eps_low: float = 0.2
    eps_high: float = 0.28
    beta: float = 0.04
    overlong_penalty: float = 1.0
    max_response_len: int = 256
    lr: float = 1e-6
    max_steps: int = 1000
    log_every: int = 50
    grad_clip: float = 1.0


class DAPOTrainer:
    """DAPO trainer with CuPy/NumPy fallbacks. All four innovations implemented:
    clip-higher, dynamic sampling, token-level loss, overlong penalty."""

    def __init__(
        self,
        model: Qwen3Model,
        ref_model: Qwen3Model,
        optimizer: AdamW,
        reward_fn: Callable,
        config: DAPOConfig | None = None,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.cfg = config or DAPOConfig()

    def compute_loss_and_backward(
        self, ids: cp.ndarray, mask: cp.ndarray, advantages: cp.ndarray,
    ) -> float:
        """DAPO loss with token-level policy gradient + KL, then backward."""
        xp = cp
        N, T = ids.shape

        new_lp, logits, resp_mask = gather_logprobs(self.model, ids, mask)
        old_lp, _, _ = gather_logprobs(self.model, ids, mask)
        ref_lp, _, _ = gather_logprobs(self.ref_model, ids, mask)

        T1 = new_lp.shape[1]
        V = logits.shape[2]

        ratios = xp.exp(new_lp - old_lp)
        adv_tok = xp.broadcast_to(advantages[:, None], (N, T1)).copy() * resp_mask

        # Clip-higher surrogate (forward)
        surrogate = dapo_clip_higher_fwd(
            ratios, adv_tok, self.cfg.eps_low, self.cfg.eps_high,
        )

        # KL (forward)
        kl = dapo_kl_fwd(new_lp, ref_lp)

        # Token-level loss: divide by total valid tokens (not sequences)
        total_tokens = float(resp_mask.sum())
        per_token_loss = -surrogate + self.cfg.beta * kl
        loss_val = float((per_token_loss * resp_mask).sum() / max(total_tokens, 1.0))

        # --- Backward ---
        grad_ptl = resp_mask / max(total_tokens, 1.0)

        # Clip-higher backward
        grad_surr = -grad_ptl
        grad_ratios = dapo_clip_higher_bwd(
            ratios, adv_tok, grad_surr, self.cfg.eps_low, self.cfg.eps_high,
        )

        # KL backward
        grad_kl_in = self.cfg.beta * grad_ptl
        grad_new_lp_kl = dapo_kl_bwd(new_lp, ref_lp, grad_kl_in)

        grad_new_lp = grad_ratios * ratios + grad_new_lp_kl

        # log_softmax_gather backward
        tgts = ids[:, 1:].reshape(N * T1).astype(xp.int32)
        grad_logits = log_softmax_gather_bwd(
            logits.reshape(N * T1, V), tgts,
            grad_new_lp.reshape(N * T1),
        )

        full_grad = xp.zeros((N, T, V), dtype=xp.float32)
        full_grad[:, 1:, :] = grad_logits.reshape(N, T1, V)
        self.model.backward(full_grad)
        return loss_val

    def train_step(self, prompts: list, prompt_ids: cp.ndarray) -> dict[str, float]:
        """One DAPO step with all four innovations."""
        xp = cp
        B, G = len(prompts), self.cfg.group_size

        # Generate + score (reward_fn returns ids, mask, rewards, lengths)
        ids, mask, rewards, lengths = self.reward_fn(prompts, prompt_ids, G)
        rewards = rewards.astype(xp.float32)
        lengths = lengths.astype(xp.float32)

        # Overlong penalty
        shaped_rewards = dapo_overlong_penalty(
            rewards, lengths,
            float(self.cfg.max_response_len), self.cfg.overlong_penalty,
        )

        # Group advantages + dynamic sampling
        adv, valid_mask = dapo_group_advantages(
            shaped_rewards.reshape(B, G), G,
        )

        # Filter: only update on valid groups
        advantages = adv.reshape(-1)
        group_valid = xp.repeat(valid_mask, G)  # (B*G,)
        advantages = advantages * group_valid

        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            return {"loss": 0.0, "mean_reward": float(rewards.mean()),
                    "n_valid_groups": 0}

        # Policy update
        loss = self.compute_loss_and_backward(ids, mask, advantages)
        self.optimizer.step(self.model.grads)
        self.model.zero_grad()

        return {"loss": loss, "mean_reward": float(rewards.mean()),
                "n_valid_groups": n_valid,
                "mean_advantage": float(advantages[group_valid > 0.5].mean())}

    def train(self, dataloader) -> list[dict[str, float]]:
        """Full training loop."""
        history = []
        for step, batch in enumerate(dataloader):
            if step >= self.cfg.max_steps:
                break
            m = self.train_step(batch["prompts"], batch["prompt_ids"])
            history.append(m)
            if step % self.cfg.log_every == 0:
                print(f"[DAPO] step {step:>5d}  loss={m['loss']:.4f}  "
                      f"reward={m['mean_reward']:.4f}  "
                      f"valid={m['n_valid_groups']}")
        return history


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"=== DAPO Smoke Test (backend: {'cuTile' if _USE_CUTILE_KERNELS else 'CuPy/NumPy fallback'}) ===\n")

    xp = cp
    if hasattr(xp, 'random') and hasattr(xp.random, 'seed'):
        xp.random.seed(42)

    def _to_np(a):
        """Convert to NumPy array regardless of backend."""
        if hasattr(a, 'get'):
            return a.get()
        return np.asarray(a)

    def _rand(*shape):
        if hasattr(xp.random, 'randn'):
            return xp.random.randn(*shape)
        return np.random.randn(*shape)

    def _rand_uniform(*shape):
        if hasattr(xp.random, 'rand'):
            return xp.random.rand(*shape)
        return np.random.rand(*shape)

    # --- Test 1: Group advantages + dynamic sampling ---
    print("--- Group advantages + dynamic sampling ---")
    B, G = 4, 8
    rewards_raw = _rand(B, G).astype(xp.float32)
    rewards_raw[2, :] = 1.0  # all-same group → should be filtered
    adv, valid = dapo_group_advantages(rewards_raw, G)

    assert float(valid[2]) == 0.0, "Zero-variance group not filtered"
    assert float(valid[0]) == 1.0, "Valid group incorrectly filtered"
    assert float(adv[2].sum()) == 0.0, "Filtered group has non-zero advantages"
    print(f"  valid_mask: {_to_np(valid)}")
    print("  Dynamic sampling filters zero-variance groups.\n")

    # --- Test 2: Overlong penalty ---
    print("--- Overlong penalty ---")
    N = 6
    rewards_1d = xp.ones(N, dtype=xp.float32)
    lengths = xp.array([100, 200, 300, 256, 512, 128], dtype=xp.float32)
    shaped = dapo_overlong_penalty(rewards_1d, lengths, max_len=256.0, penalty=1.0)

    expected = np.array([
        1.0 - max(0, l - 256) / 256
        for l in [100, 200, 300, 256, 512, 128]
    ], dtype=np.float32)
    np.testing.assert_allclose(_to_np(shaped), expected, atol=1e-4)
    print(f"  shaped rewards: {_to_np(shaped)}")
    print("  Overlong penalty correct.\n")

    # --- Test 3: Asymmetric clipped surrogate ---
    print("--- Clip-higher (asymmetric) ---")
    N, T = 8, 32
    eps_low, eps_high = 0.2, 0.28
    ratios = (0.7 + 0.6 * _rand_uniform(N, T)).astype(xp.float32)
    advantages = _rand(N, T).astype(xp.float32)

    surr = dapo_clip_higher_fwd(ratios, advantages, eps_low, eps_high)

    clipped = xp.clip(ratios, 1 - eps_low, 1 + eps_high)
    ref = xp.minimum(ratios * advantages, clipped * advantages)
    np.testing.assert_allclose(_to_np(surr), _to_np(ref), atol=1e-4)
    print("  Clip-higher matches.\n")

    # --- Test 4: Clip-higher backward ---
    print("--- Clip-higher backward ---")
    grad_out = xp.ones_like(ratios) / (N * T)
    grad_ratios = dapo_clip_higher_bwd(ratios, advantages, grad_out, eps_low, eps_high)
    assert grad_ratios.shape == ratios.shape
    print("  Clip-higher backward runs correctly.\n")

    # --- Test 5: KL divergence ---
    print("--- KL divergence ---")
    new_lp = (_rand(N, T) * 0.1).astype(xp.float32)
    ref_lp = (_rand(N, T) * 0.1).astype(xp.float32)
    kl = dapo_kl_fwd(new_lp, ref_lp)

    log_ratio = ref_lp - new_lp
    kl_ref = xp.exp(log_ratio) - log_ratio - 1.0
    np.testing.assert_allclose(_to_np(kl), _to_np(kl_ref), atol=1e-4)
    print("  KL matches.\n")

    # --- Test 6: KL backward ---
    print("--- KL backward ---")
    grad_kl_out = xp.ones_like(new_lp) / (N * T)
    grad_new = dapo_kl_bwd(new_lp, ref_lp, grad_kl_out)

    grad_new_ref = (-xp.exp(ref_lp - new_lp) + 1.0) * grad_kl_out
    np.testing.assert_allclose(_to_np(grad_new), _to_np(grad_new_ref), atol=1e-4)
    print("  KL backward matches.\n")

    # --- Test 7: log_softmax_gather ---
    print("--- log_softmax_gather ---")
    V = 64
    logits_2d = _rand(N, V).astype(xp.float32)
    targets = xp.array([i % V for i in range(N)], dtype=xp.int32)

    lp = log_softmax_gather(logits_2d, targets)

    logits_np = _to_np(logits_2d)
    targets_np = _to_np(targets)
    max_l = logits_np.max(axis=-1, keepdims=True)
    shifted_np = logits_np - max_l
    lse = np.log(np.exp(shifted_np).sum(axis=-1)) + max_l.squeeze(-1)
    lp_ref = logits_np[np.arange(N), targets_np] - lse
    np.testing.assert_allclose(_to_np(lp), lp_ref, atol=1e-4)
    print("  log_softmax_gather matches.\n")

    # --- Test 8: log_softmax_gather backward ---
    print("--- log_softmax_gather backward ---")
    grad_lp_1d = xp.ones(N, dtype=xp.float32) / N
    grad_logits = log_softmax_gather_bwd(logits_2d, targets, grad_lp_1d)
    assert grad_logits.shape == (N, V)
    print("  log_softmax_gather backward runs correctly.\n")

    print("=== DAPO Smoke Test Complete ===")
