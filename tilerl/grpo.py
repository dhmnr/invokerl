"""GRPO — Group Relative Policy Optimization.

Paper: https://arxiv.org/abs/2402.03300 (Shao et al., DeepSeekMath, 2024)

Key idea: Estimate baselines from group-level reward statistics instead of
training a separate critic model. For each prompt, sample G completions,
score them, normalize advantages within the group, then optimize a clipped
surrogate objective with KL regularization.

Objective (per token t):
    L = -E[ min(ratio_t · Â_i, clip(ratio_t, 1-ε, 1+ε) · Â_i) ] + β · KL_t

where:
    ratio_t = π_θ(a_t|s_t) / π_old(a_t|s_t)
    Â_i = (r_i - mean(r_group)) / std(r_group)
    KL_t ≈ exp(ref - new) - (ref - new) - 1   (Schulman approximation)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Backend selection — mirrors qwen3.py pattern
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
    """Per-token log-softmax + gather. Three passes: max, log-sum-exp, gather."""
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
    target_tile_idx = target / TILE_V
    target_offset = target % TILE_V
    logit_tile = ct.load(logits, index=(pid, target_tile_idx), shape=(1, TILE_V), padding_mode=PAD_ZERO)
    offsets = ct.arange(TILE_V, dtype=ct.int32)
    target_logit = ct.sum(ct.where(offsets == target_offset, logit_tile, 0.0), axis=1)
    ct.store(output, index=(pid,), tile=target_logit - log_sum_exp)


@ct.kernel
def grpo_advantage_kernel(
    rewards: ct.Array, advantages: ct.Array, G: ConstInt,
):
    """Group-normalize rewards: Â_i = (r_i - mean(r)) / (std(r) + eps)."""
    pid = ct.bid(0)
    r = ct.load(rewards, index=(pid, 0), shape=(1, G))
    mean = ct.sum(r, axis=1) / G
    diff = r - mean
    var = ct.sum(diff * diff, axis=1) / G
    std = ct.sqrt(var + 1e-8)
    ct.store(advantages, index=(pid, 0), tile=diff / std)


@ct.kernel
def grpo_clipped_surrogate_fwd_kernel(
    ratios: ct.Array, advantages: ct.Array, output: ct.Array,
    eps: ct.Array, TILE: ConstInt,
):
    """Clipped surrogate: min(ratio * adv, clip(ratio, 1±ε) * adv)."""
    bid = ct.bid(0)
    num_tiles = ct.num_tiles(ratios, axis=1, shape=(1, TILE))
    e = ct.load(eps, index=(0,), shape=(1,))

    for j in range(num_tiles):
        r = ct.load(ratios, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        a = ct.load(advantages, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        clipped_r = ct.minimum(ct.maximum(r, 1.0 - e), 1.0 + e)
        ct.store(output, index=(bid, j), tile=ct.minimum(r * a, clipped_r * a))


@ct.kernel
def grpo_kl_fwd_kernel(
    new_lp: ct.Array, ref_lp: ct.Array, kl_out: ct.Array, TILE: ConstInt,
):
    """Schulman KL approx: exp(ref-new) - (ref-new) - 1."""
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
def grpo_clipped_surrogate_bwd_kernel(
    ratios: ct.Array, advantages: ct.Array, grad_out: ct.Array,
    grad_ratios: ct.Array, eps: ct.Array, TILE: ConstInt,
):
    """Backward of clipped surrogate w.r.t. ratios."""
    bid = ct.bid(0)
    num_tiles = ct.num_tiles(ratios, axis=1, shape=(1, TILE))
    e = ct.load(eps, index=(0,), shape=(1,))

    for j in range(num_tiles):
        r = ct.load(ratios, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        a = ct.load(advantages, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        g = ct.load(grad_out, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)

        clipped_r = ct.minimum(ct.maximum(r, 1.0 - e), 1.0 + e)
        surr1 = r * a
        surr2 = clipped_r * a
        # Gradient flows through the min branch
        use_clipped = ct.where(surr2 < surr1, 1.0, 0.0)
        cond_lo = ct.where(r >= 1.0 - e, 1.0, 0.0)
        cond_hi = ct.where(r <= 1.0 + e, 1.0, 0.0)
        in_range = cond_lo * cond_hi
        grad = ct.where(use_clipped > 0.5, in_range * a, a) * g
        ct.store(grad_ratios, index=(bid, j), tile=grad)


@ct.kernel
def grpo_kl_bwd_kernel(
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
    total_sum_exp = ct.sum(sum_exp, axis=1)

    for j in range(num_tiles):
        tile = ct.load(logits, index=(pid, j), shape=(1, TILE_V), padding_mode=PAD_ZERO)
        vmask = (j * TILE_V + ct.arange(TILE_V, dtype=ct.int32)) < V
        softmax_tile = ct.where(vmask, ct.exp(tile - row_max) / total_sum_exp, 0.0)
        offsets = j * TILE_V + ct.arange(TILE_V, dtype=ct.int32)
        one_hot = ct.where(offsets == target, 1.0, 0.0)
        ct.store(grad_logits, index=(pid, j), tile=g * (one_hot - softmax_tile))


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


def compute_advantages(rewards: cp.ndarray, group_size: int) -> cp.ndarray:
    """Group-normalize rewards → advantages.

    Args:
        rewards: (B, G) reward matrix.
    Returns:
        (B, G) normalized advantages.
    """
    xp = cp
    if _USE_CUTILE_KERNELS:
        B = rewards.shape[0]
        adv = xp.empty_like(rewards)
        ct.launch(_get_stream(), (B,), grpo_advantage_kernel,
                  (rewards, adv, group_size))
        return adv
    # Fallback: pure array ops
    mean = rewards.mean(axis=1, keepdims=True)
    diff = rewards - mean
    var = (diff * diff).mean(axis=1, keepdims=True)
    std = xp.sqrt(var + 1e-8)
    return (diff / std).astype(xp.float32)


def clipped_surrogate_fwd(
    ratios: cp.ndarray, advantages: cp.ndarray, eps: float,
) -> cp.ndarray:
    """Clipped surrogate: min(ratio * adv, clip(ratio, 1±ε) * adv).

    Args:
        ratios: (N, T) policy ratios.
        advantages: (N, T) per-token advantages.
        eps: Clipping epsilon.
    Returns:
        (N, T) surrogate values.
    """
    xp = cp
    if _USE_CUTILE_KERNELS:
        N = ratios.shape[0]
        T1 = ratios.shape[1]
        TILE = min(1024, _next_pow2(T1))
        eps_t = xp.array([eps], dtype=xp.float32)
        out = xp.empty_like(ratios)
        ct.launch(_get_stream(), (N,), grpo_clipped_surrogate_fwd_kernel,
                  (ratios, advantages, out, eps_t, TILE))
        return out
    # Fallback
    clipped = xp.clip(ratios, 1.0 - eps, 1.0 + eps)
    return xp.minimum(ratios * advantages, clipped * advantages).astype(xp.float32)


def kl_fwd(new_lp: cp.ndarray, ref_lp: cp.ndarray) -> cp.ndarray:
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
        ct.launch(_get_stream(), (N,), grpo_kl_fwd_kernel,
                  (new_lp, ref_lp, kl, TILE))
        return kl
    # Fallback
    log_ratio = ref_lp - new_lp
    return (xp.exp(log_ratio) - log_ratio - 1.0).astype(xp.float32)


def clipped_surrogate_bwd(
    ratios: cp.ndarray, advantages: cp.ndarray,
    grad_out: cp.ndarray, eps: float,
) -> cp.ndarray:
    """Backward of clipped surrogate w.r.t. ratios.

    Returns:
        (N, T) gradient w.r.t. ratios.
    """
    xp = cp
    if _USE_CUTILE_KERNELS:
        N = ratios.shape[0]
        T1 = ratios.shape[1]
        TILE = min(1024, _next_pow2(T1))
        eps_t = xp.array([eps], dtype=xp.float32)
        grad_ratios = xp.empty_like(ratios)
        ct.launch(_get_stream(), (N,), grpo_clipped_surrogate_bwd_kernel,
                  (ratios, advantages, grad_out, grad_ratios, eps_t, TILE))
        return grad_ratios
    # Fallback
    clipped = xp.clip(ratios, 1.0 - eps, 1.0 + eps)
    surr1 = ratios * advantages
    surr2 = clipped * advantages
    use_clipped = (surr2 < surr1).astype(xp.float32)
    in_range = ((ratios >= 1.0 - eps) & (ratios <= 1.0 + eps)).astype(xp.float32)
    grad = xp.where(use_clipped > 0.5, in_range * advantages, advantages) * grad_out
    return grad.astype(xp.float32)


def kl_bwd(
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
        ct.launch(_get_stream(), (N,), grpo_kl_bwd_kernel,
                  (new_lp, ref_lp, grad_out, grad_new, TILE))
        return grad_new
    # Fallback
    return ((-xp.exp(ref_lp - new_lp) + 1.0) * grad_out).astype(xp.float32)


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
    # Fallback — memory-efficient: reuses single (N, V) buffer via in-place ops.
    # Original allocated ~7 × (N, V) temporaries; this uses ~1 × (N, V) extra.
    # For N=1600, V=151936 this saves ~5.8 GB of peak GPU memory.
    buf = logits.astype(xp.float32)       # (N, V) — will be reused in-place
    row_max = buf.max(axis=-1, keepdims=True)
    buf -= row_max                         # buf = shifted logits
    xp.exp(buf, out=buf)                   # buf = exp(shifted)
    denom = buf.sum(axis=-1, keepdims=True)
    buf /= denom                           # buf = softmax
    buf *= -grad_lp[:, None]               # buf = -grad_lp * softmax
    buf[xp.arange(N), targets.astype(int)] += grad_lp  # + grad_lp at target positions
    return buf


def gather_logprobs(
    model: Qwen3Model, token_ids: cp.ndarray, response_mask: cp.ndarray,
    recompute_attn: bool = False, no_cache: bool = False,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Model forward → per-token log-probs masked to response tokens.

    Args:
        recompute_attn: Skip caching attention weights (saves ~4.5 GB).
        no_cache: Skip ALL activation caching (saves ~13 GB). Use for
            reference models where backward() won't be called.
    """
    xp = cp
    logits = model.forward(token_ids, recompute_attn=recompute_attn, no_cache=no_cache)
    B, T, V = logits.shape
    shift_logits = logits[:, :-1, :].reshape(B * (T - 1), V).astype(xp.float32)
    shift_targets = token_ids[:, 1:].reshape(B * (T - 1)).astype(xp.int32)

    lp_flat = log_softmax_gather(shift_logits, shift_targets)

    mask = response_mask[:, 1:].astype(xp.float32)
    return lp_flat.reshape(B, T - 1) * mask, shift_logits.reshape(B, T - 1, V), mask


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    """Hyperparameters for GRPO training."""
    group_size: int = 8
    clip_eps: float = 0.2
    beta: float = 0.04
    lr: float = 1e-6
    max_steps: int = 1000
    log_every: int = 50
    grad_clip: float = 1.0
    max_new_tokens: int = 128
    temperature: float = 0.7


class GRPOTrainer:
    """Pure cuTile GRPO trainer. No torch, no autograd.

    Single-epoch per batch: old_lp is computed from the current model before
    the update step, so ratios start at 1.0. For multi-epoch training, cache
    old_lp before the optimization loop.
    """

    def __init__(
        self,
        model: Qwen3Model,
        ref_model: Qwen3Model,
        optimizer: AdamW,
        reward_fn: Callable,
        config: GRPOConfig | None = None,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.cfg = config or GRPOConfig()

    def compute_loss_and_backward(
        self,
        input_ids: cp.ndarray,
        response_mask: cp.ndarray,
        advantages: cp.ndarray,
    ) -> float:
        """Compute GRPO loss and backprop through model.

        Args:
            input_ids: Shape (N, T), int32.
            response_mask: Shape (N, T), float32.
            advantages: Shape (N,), float32, group-normalized.

        Returns:
            Scalar loss value.
        """
        xp = cp
        N, T = input_ids.shape

        # Forward: get log-probs from policy, old policy, and reference
        new_lp, logits_2d, mask = gather_logprobs(self.model, input_ids, response_mask)
        old_lp, _, _ = gather_logprobs(self.model, input_ids, response_mask)
        ref_lp, _, _ = gather_logprobs(self.ref_model, input_ids, response_mask)

        T1 = new_lp.shape[1]

        # Policy ratios
        ratios = xp.exp(new_lp - old_lp)  # (N, T1)
        adv_tok = xp.broadcast_to(advantages[:, None], (N, T1)).copy()

        # Clipped surrogate (forward)
        surrogate = clipped_surrogate_fwd(ratios, adv_tok, self.cfg.clip_eps)

        # KL divergence (forward)
        kl = kl_fwd(new_lp, ref_lp)

        # Loss: -surrogate + β·KL, averaged over response tokens
        per_token_loss = -surrogate + self.cfg.beta * kl
        total_tokens = float(mask.sum())
        loss_val = float((per_token_loss * mask).sum() / max(total_tokens, 1.0))

        # --- Backward ---
        grad_ptl = mask / max(total_tokens, 1.0)

        # Surrogate backward → grad w.r.t. ratios
        grad_surr = -grad_ptl
        grad_ratios = clipped_surrogate_bwd(ratios, adv_tok, grad_surr, self.cfg.clip_eps)

        # KL backward → grad w.r.t. new_lp
        grad_kl_input = self.cfg.beta * grad_ptl
        grad_new_lp_kl = kl_bwd(new_lp, ref_lp, grad_kl_input)

        # ratio = exp(new_lp - old_lp) → d(ratio)/d(new_lp) = ratio
        grad_new_lp = grad_ratios * ratios + grad_new_lp_kl

        # log_softmax_gather backward → grad w.r.t. logits
        V = logits_2d.shape[2]
        shift_targets = input_ids[:, 1:].reshape(N * T1).astype(xp.int32)
        grad_logits_flat = log_softmax_gather_bwd(
            logits_2d.reshape(N * T1, V), shift_targets,
            grad_new_lp.reshape(N * T1),
        )

        # Backprop through model
        full_grad = xp.zeros((N, T, V), dtype=xp.float32)
        full_grad[:, 1:, :] = grad_logits_flat.reshape(N, T1, V)
        self.model.backward(full_grad)

        return loss_val

    def train_step(
        self, prompts: list, prompt_ids: cp.ndarray,
    ) -> dict[str, float]:
        """One GRPO step: generate → score → normalize advantages → update."""
        B, G = len(prompts), self.cfg.group_size

        # Step 1: Generate G completions per prompt (placeholder — needs model.generate)
        # In practice: for each prompt, sample G responses, build input_ids + response_mask
        # For now, assume reward_fn handles generation or pre-generated data is passed
        input_ids, response_mask, rewards = self.reward_fn(prompts, prompt_ids, G)
        rewards = rewards.astype(cp.float32)

        # Step 2: Group-normalized advantages
        adv = compute_advantages(rewards.reshape(B, G), G)
        advantages = adv.reshape(-1)

        # Step 3: Policy update
        loss = self.compute_loss_and_backward(input_ids, response_mask, advantages)
        self.optimizer.step(self.model.grads)
        self.model.zero_grad()

        return {"loss": loss, "mean_reward": float(rewards.mean()),
                "mean_advantage": float(advantages.mean())}

    def train(self, prompt_dataloader) -> list[dict[str, float]]:
        """Full training loop. Batches yield {prompts, prompt_ids}."""
        history: list[dict[str, float]] = []
        for step, batch in enumerate(prompt_dataloader):
            if step >= self.cfg.max_steps:
                break
            metrics = self.train_step(batch["prompts"], batch["prompt_ids"])
            history.append(metrics)
            if step % self.cfg.log_every == 0:
                print(f"[GRPO] step {step:>5d}  loss={metrics['loss']:.4f}  "
                      f"reward={metrics['mean_reward']:.4f}")
        return history


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"=== GRPO Smoke Test (backend: {'cuTile' if _USE_CUTILE_KERNELS else 'CuPy/NumPy fallback'}) ===\n")

    xp = cp
    if hasattr(xp, 'random') and hasattr(xp.random, 'seed'):
        xp.random.seed(42)

    # --- Test 1: Advantage normalization (via wrapper) ---
    print("--- Advantage normalization ---")
    B, G = 3, 8
    rewards = xp.random.randn(B, G).astype(xp.float32) if hasattr(xp.random, 'randn') else \
              np.random.randn(B, G).astype(np.float32)
    adv = compute_advantages(rewards, G)

    mean_r = rewards.mean(axis=1, keepdims=True)
    var_r = ((rewards - mean_r) ** 2).mean(axis=1, keepdims=True)
    adv_ref = (rewards - mean_r) / xp.sqrt(var_r + 1e-8)

    np.testing.assert_allclose(
        np.asarray(adv) if hasattr(adv, 'get') else adv,
        np.asarray(adv_ref) if hasattr(adv_ref, 'get') else adv_ref,
        atol=1e-4,
    )
    print("  Advantage computation matches.\n")

    # --- Test 2: Clipped surrogate (via wrapper) ---
    print("--- Clipped surrogate ---")
    N, T = 8, 32
    eps = 0.2
    ratios = (0.8 + 0.4 * xp.random.rand(N, T)).astype(xp.float32) if hasattr(xp.random, 'rand') else \
             (0.8 + 0.4 * np.random.rand(N, T)).astype(np.float32)
    advantages = xp.random.randn(N, T).astype(xp.float32) if hasattr(xp.random, 'randn') else \
                 np.random.randn(N, T).astype(np.float32)

    surr = clipped_surrogate_fwd(ratios, advantages, eps)

    clipped = xp.clip(ratios, 1 - eps, 1 + eps)
    surr_ref = xp.minimum(ratios * advantages, clipped * advantages)
    np.testing.assert_allclose(
        np.asarray(surr) if hasattr(surr, 'get') else surr,
        np.asarray(surr_ref) if hasattr(surr_ref, 'get') else surr_ref,
        atol=1e-4,
    )
    print("  Clipped surrogate matches.\n")

    # --- Test 3: KL divergence (via wrapper) ---
    print("--- KL divergence ---")
    new_lp = (xp.random.randn(N, T) * 0.1).astype(xp.float32) if hasattr(xp.random, 'randn') else \
             (np.random.randn(N, T) * 0.1).astype(np.float32)
    ref_lp = (xp.random.randn(N, T) * 0.1).astype(xp.float32) if hasattr(xp.random, 'randn') else \
             (np.random.randn(N, T) * 0.1).astype(np.float32)
    kl = kl_fwd(new_lp, ref_lp)

    log_ratio = ref_lp - new_lp
    kl_ref = xp.exp(log_ratio) - log_ratio - 1.0
    np.testing.assert_allclose(
        np.asarray(kl) if hasattr(kl, 'get') else kl,
        np.asarray(kl_ref) if hasattr(kl_ref, 'get') else kl_ref,
        atol=1e-4,
    )
    print("  KL divergence matches.\n")

    # --- Test 4: Backward kernels (via wrappers) ---
    print("--- Backward: clipped surrogate ---")
    grad_out = xp.ones_like(ratios) / (N * T)
    grad_ratios = clipped_surrogate_bwd(ratios, advantages, grad_out, eps)
    assert grad_ratios.shape == ratios.shape
    print("  Surrogate backward runs correctly.\n")

    print("--- Backward: KL ---")
    grad_kl_out = xp.ones_like(new_lp) / (N * T)
    grad_new = kl_bwd(new_lp, ref_lp, grad_kl_out)

    grad_new_ref = (-xp.exp(ref_lp - new_lp) + 1.0) * grad_kl_out
    np.testing.assert_allclose(
        np.asarray(grad_new) if hasattr(grad_new, 'get') else grad_new,
        np.asarray(grad_new_ref) if hasattr(grad_new_ref, 'get') else grad_new_ref,
        atol=1e-4,
    )
    print("  KL backward matches.\n")

    # --- Test 5: log_softmax_gather (via wrapper) ---
    print("--- log_softmax_gather ---")
    V = 64
    logits = xp.random.randn(N, V).astype(xp.float32) if hasattr(xp.random, 'randn') else \
             np.random.randn(N, V).astype(np.float32)
    targets = xp.array([i % V for i in range(N)], dtype=xp.int32)

    lp = log_softmax_gather(logits, targets)

    # Reference: log_softmax + gather
    logits_np = np.asarray(logits) if hasattr(logits, 'get') else np.array(logits)
    targets_np = np.asarray(targets) if hasattr(targets, 'get') else np.array(targets)
    max_l = logits_np.max(axis=-1, keepdims=True)
    shifted = logits_np - max_l
    lse = np.log(np.exp(shifted).sum(axis=-1)) + max_l.squeeze(-1)
    lp_ref = logits_np[np.arange(N), targets_np] - lse

    np.testing.assert_allclose(
        np.asarray(lp) if hasattr(lp, 'get') else lp,
        lp_ref, atol=1e-4,
    )
    print("  log_softmax_gather matches.\n")

    # --- Test 6: log_softmax_gather backward (via wrapper) ---
    print("--- log_softmax_gather backward ---")
    grad_lp = xp.ones(N, dtype=xp.float32) / N
    grad_logits = log_softmax_gather_bwd(logits, targets, grad_lp)
    assert grad_logits.shape == (N, V)
    print("  log_softmax_gather backward runs correctly.\n")

    print("=== GRPO Smoke Test Complete ===")
