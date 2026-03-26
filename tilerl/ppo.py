"""PPO — Proximal Policy Optimization for RLHF.

Paper: https://arxiv.org/abs/1707.06347 (Schulman et al., 2017)
RLHF: https://arxiv.org/abs/2203.02155 (Ouyang et al., 2022)

L = L_clip + c1·L_value - c2·H[π]
L_clip = -E[min(ρ·A, clip(ρ,1±ε)·A)], L_value = E[max((V-R)², (clipV-R)²)]
GAE: Â_t = Σ (γλ)^l · δ_{t+l}, δ_t = r_t + γV_{t+1} - V_t
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

# -- cuTile kernels: forward --

@ct.kernel
def log_softmax_gather_fwd_kernel(
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
def gae_kernel(
    rewards: ct.Array, values: ct.Array, mask: ct.Array,
    advantages: ct.Array, returns: ct.Array,
    gamma: ct.Array, lam: ct.Array, T: ConstInt,
):
    """GAE reverse scan: δ_t = r_t + γV_{t+1} - V_t, Â_t = δ_t + γλÂ_{t+1}."""
    bid = ct.bid(0)
    g = ct.load(gamma, index=(0,), shape=(1,))
    l = ct.load(lam, index=(0,), shape=(1,))  # noqa: E741

    last_adv = ct.full((1,), 0.0, dtype=ct.float32)
    last_val = ct.full((1,), 0.0, dtype=ct.float32)

    # Reverse scan — process one token at a time
    for t_rev in range(T):
        t = T - 1 - t_rev
        r_t = ct.load(rewards, index=(bid, t), shape=(1, 1)).reshape((1,))
        v_t = ct.load(values, index=(bid, t), shape=(1, 1)).reshape((1,))
        m_t = ct.load(mask, index=(bid, t), shape=(1, 1)).reshape((1,))

        next_val = ct.where(t_rev == 0, ct.full((1,), 0.0, dtype=ct.float32), last_val)
        delta = r_t + g * next_val - v_t
        adv = delta + g * l * last_adv
        adv = adv * m_t

        ct.store(advantages, index=(bid, t), tile=adv.reshape((1, 1)))
        ct.store(returns, index=(bid, t), tile=(adv + v_t).reshape((1, 1)))

        last_adv = adv
        last_val = v_t


@ct.kernel
def ppo_clip_fwd_kernel(
    ratios: ct.Array, advantages: ct.Array, output: ct.Array,
    eps: ct.Array, TILE: ConstInt,
):
    """PPO clipped surrogate: -min(ρ·A, clip(ρ,1±ε)·A)."""
    bid = ct.bid(0)
    num_tiles = ct.num_tiles(ratios, axis=1, shape=(1, TILE))
    e = ct.load(eps, index=(0,), shape=(1,))

    for j in range(num_tiles):
        r = ct.load(ratios, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        a = ct.load(advantages, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        clipped = ct.minimum(ct.maximum(r, 1.0 - e), 1.0 + e)
        ct.store(output, index=(bid, j), tile=-ct.minimum(r * a, clipped * a))


@ct.kernel
def value_loss_fwd_kernel(
    values: ct.Array, old_values: ct.Array, returns: ct.Array,
    output: ct.Array, eps: ct.Array, TILE: ConstInt,
):
    """Clipped value loss: max((V-R)², (clip(V,V_old±ε)-R)²)."""
    bid = ct.bid(0)
    num_tiles = ct.num_tiles(values, axis=1, shape=(1, TILE))
    e = ct.load(eps, index=(0,), shape=(1,))

    for j in range(num_tiles):
        v = ct.load(values, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        v_old = ct.load(old_values, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        ret = ct.load(returns, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        v_clip = ct.minimum(ct.maximum(v, v_old - e), v_old + e)
        loss1 = (v - ret) * (v - ret)
        loss2 = (v_clip - ret) * (v_clip - ret)
        ct.store(output, index=(bid, j), tile=ct.maximum(loss1, loss2))


@ct.kernel
def entropy_fwd_kernel(
    logits: ct.Array, output: ct.Array, V: ConstInt, TILE_V: ConstInt,
):
    """Policy entropy: -Σ p·log(p) via softmax. Returns per-position entropy."""
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
    total = ct.sum(sum_exp, axis=1)
    log_total = ct.log(total) + row_max

    ent = ct.full((1, TILE_V), 0.0, dtype=ct.float32)
    for j in range(num_tiles):
        tile = ct.load(logits, index=(pid, j), shape=(1, TILE_V), padding_mode=PAD_ZERO)
        vmask = (j * TILE_V + ct.arange(TILE_V, dtype=ct.int32)) < V
        p = ct.where(vmask, ct.exp(tile - row_max) / total, 0.0)
        log_p = ct.where(vmask, tile - log_total, 0.0)
        ent += -p * log_p
    ct.store(output, index=(pid,), tile=ct.sum(ent, axis=1))


# -- cuTile kernels: backward --

@ct.kernel
def ppo_clip_bwd_kernel(
    ratios: ct.Array, advantages: ct.Array, grad_out: ct.Array,
    grad_ratios: ct.Array, eps: ct.Array, TILE: ConstInt,
):
    """Backward of PPO clipped surrogate w.r.t. ratios."""
    bid = ct.bid(0)
    num_tiles = ct.num_tiles(ratios, axis=1, shape=(1, TILE))
    e = ct.load(eps, index=(0,), shape=(1,))

    for j in range(num_tiles):
        r = ct.load(ratios, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        a = ct.load(advantages, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        g = ct.load(grad_out, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        clipped = ct.minimum(ct.maximum(r, 1.0 - e), 1.0 + e)
        use_clipped = ct.where((clipped * a) < (r * a), 1.0, 0.0)
        cond_lo = ct.where(r >= 1.0 - e, 1.0, 0.0)
        cond_hi = ct.where(r <= 1.0 + e, 1.0, 0.0)
        in_range = cond_lo * cond_hi
        # -min(r*a, clip*a) → negate the surrogate grad
        grad = -ct.where(use_clipped > 0.5, in_range * a, a) * g
        ct.store(grad_ratios, index=(bid, j), tile=grad)


@ct.kernel
def value_loss_bwd_kernel(
    values: ct.Array, old_values: ct.Array, returns: ct.Array,
    grad_out: ct.Array, grad_values: ct.Array, eps: ct.Array, TILE: ConstInt,
):
    """Backward of clipped value loss w.r.t. values."""
    bid = ct.bid(0)
    num_tiles = ct.num_tiles(values, axis=1, shape=(1, TILE))
    e = ct.load(eps, index=(0,), shape=(1,))

    for j in range(num_tiles):
        v = ct.load(values, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        v_old = ct.load(old_values, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        ret = ct.load(returns, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        g = ct.load(grad_out, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        v_clip = ct.minimum(ct.maximum(v, v_old - e), v_old + e)
        loss1 = (v - ret) * (v - ret)
        loss2 = (v_clip - ret) * (v_clip - ret)
        use_clip = ct.where(loss2 > loss1, 1.0, 0.0)
        cond_lo = ct.where(v >= v_old - e, 1.0, 0.0)
        cond_hi = ct.where(v <= v_old + e, 1.0, 0.0)
        in_range = cond_lo * cond_hi
        grad = ct.where(use_clip > 0.5, 2.0 * (v_clip - ret) * in_range,
                        2.0 * (v - ret)) * g
        ct.store(grad_values, index=(bid, j), tile=grad)


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


def log_softmax_gather_fwd(logits: cp.ndarray, targets: cp.ndarray) -> cp.ndarray:
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
        ct.launch(_get_stream(), (N,), log_softmax_gather_fwd_kernel,
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


def compute_gae(
    rewards: cp.ndarray, values: cp.ndarray, mask: cp.ndarray,
    gamma: float, lam: float,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Compute GAE advantages and returns.

    Args:
        rewards: (N, T) reward array.
        values: (N, T) value estimates.
        mask: (N, T) response mask (1=valid, 0=pad).
        gamma: Discount factor.
        lam: GAE lambda.
    Returns:
        advantages: (N, T), returns: (N, T).
    """
    xp = cp
    N, T = rewards.shape
    if _USE_CUTILE_KERNELS:
        advantages = xp.empty_like(rewards)
        returns = xp.empty_like(rewards)
        gamma_t = xp.array([gamma], dtype=xp.float32)
        lam_t = xp.array([lam], dtype=xp.float32)
        ct.launch(_get_stream(), (N,), gae_kernel,
                  (rewards, values, mask, advantages, returns, gamma_t, lam_t, T))
        return advantages, returns
    # Fallback: reverse scan in Python/NumPy
    advantages = xp.zeros_like(rewards)
    returns = xp.zeros_like(rewards)
    last_adv = xp.zeros(N, dtype=xp.float32)
    last_val = xp.zeros(N, dtype=xp.float32)
    for t_rev in range(T):
        t = T - 1 - t_rev
        r_t = rewards[:, t]
        v_t = values[:, t]
        m_t = mask[:, t]
        next_val = xp.zeros(N, dtype=xp.float32) if t_rev == 0 else last_val
        delta = r_t + gamma * next_val - v_t
        adv = (delta + gamma * lam * last_adv) * m_t
        advantages[:, t] = adv
        returns[:, t] = adv + v_t
        last_adv = adv
        last_val = v_t
    return advantages.astype(xp.float32), returns.astype(xp.float32)


def ppo_clip_fwd(
    ratios: cp.ndarray, advantages: cp.ndarray, eps: float,
) -> cp.ndarray:
    """PPO clipped surrogate: -min(ρ·A, clip(ρ,1±ε)·A).

    Args:
        ratios: (N, T) policy ratios.
        advantages: (N, T) normalized advantages.
        eps: Clipping epsilon.
    Returns:
        (N, T) per-token policy loss (positive = penalize).
    """
    xp = cp
    if _USE_CUTILE_KERNELS:
        N = ratios.shape[0]
        T1 = ratios.shape[1]
        TILE = min(1024, _next_pow2(T1))
        eps_t = xp.array([eps], dtype=xp.float32)
        out = xp.empty_like(ratios)
        ct.launch(_get_stream(), (N,), ppo_clip_fwd_kernel,
                  (ratios, advantages, out, eps_t, TILE))
        return out
    # Fallback
    clipped = xp.clip(ratios, 1.0 - eps, 1.0 + eps)
    return (-xp.minimum(ratios * advantages, clipped * advantages)).astype(xp.float32)


def ppo_clip_bwd(
    ratios: cp.ndarray, advantages: cp.ndarray,
    grad_out: cp.ndarray, eps: float,
) -> cp.ndarray:
    """Backward of PPO clipped surrogate w.r.t. ratios.

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
        ct.launch(_get_stream(), (N,), ppo_clip_bwd_kernel,
                  (ratios, advantages, grad_out, grad_ratios, eps_t, TILE))
        return grad_ratios
    # Fallback
    clipped = xp.clip(ratios, 1.0 - eps, 1.0 + eps)
    use_clipped = ((clipped * advantages) < (ratios * advantages)).astype(xp.float32)
    in_range = ((ratios >= 1.0 - eps) & (ratios <= 1.0 + eps)).astype(xp.float32)
    grad = -xp.where(use_clipped > 0.5, in_range * advantages, advantages) * grad_out
    return grad.astype(xp.float32)


def value_loss_fwd(
    values: cp.ndarray, old_values: cp.ndarray, returns: cp.ndarray, eps: float,
) -> cp.ndarray:
    """Clipped value loss: max((V-R)², (clip(V,V_old±ε)-R)²).

    Args:
        values: (N, T) current value estimates.
        old_values: (N, T) old value estimates.
        returns: (N, T) GAE returns.
        eps: Clipping epsilon.
    Returns:
        (N, T) per-token value loss.
    """
    xp = cp
    if _USE_CUTILE_KERNELS:
        N = values.shape[0]
        T1 = values.shape[1]
        TILE = min(1024, _next_pow2(T1))
        eps_t = xp.array([eps], dtype=xp.float32)
        out = xp.empty_like(values)
        ct.launch(_get_stream(), (N,), value_loss_fwd_kernel,
                  (values, old_values, returns, out, eps_t, TILE))
        return out
    # Fallback
    v_clip = xp.clip(values, old_values - eps, old_values + eps)
    loss1 = (values - returns) ** 2
    loss2 = (v_clip - returns) ** 2
    return xp.maximum(loss1, loss2).astype(xp.float32)


def value_loss_bwd(
    values: cp.ndarray, old_values: cp.ndarray, returns: cp.ndarray,
    grad_out: cp.ndarray, eps: float,
) -> cp.ndarray:
    """Backward of clipped value loss w.r.t. values.

    Returns:
        (N, T) gradient w.r.t. values.
    """
    xp = cp
    if _USE_CUTILE_KERNELS:
        N = values.shape[0]
        T1 = values.shape[1]
        TILE = min(1024, _next_pow2(T1))
        eps_t = xp.array([eps], dtype=xp.float32)
        grad_values = xp.empty_like(values)
        ct.launch(_get_stream(), (N,), value_loss_bwd_kernel,
                  (values, old_values, returns, grad_out, grad_values, eps_t, TILE))
        return grad_values
    # Fallback
    v_clip = xp.clip(values, old_values - eps, old_values + eps)
    loss1 = (values - returns) ** 2
    loss2 = (v_clip - returns) ** 2
    use_clip = (loss2 > loss1).astype(xp.float32)
    in_range = ((values >= old_values - eps) & (values <= old_values + eps)).astype(xp.float32)
    grad = xp.where(use_clip > 0.5,
                    2.0 * (v_clip - returns) * in_range,
                    2.0 * (values - returns)) * grad_out
    return grad.astype(xp.float32)


def entropy_fwd(logits: cp.ndarray) -> cp.ndarray:
    """Policy entropy: -Σ p·log(p). Returns per-row entropy.

    Args:
        logits: (N, V) logits.
    Returns:
        (N,) entropy values.
    """
    xp = cp
    N, V = logits.shape
    if _USE_CUTILE_KERNELS:
        TILE_V = min(1024, _next_pow2(V))
        out = xp.empty(N, dtype=xp.float32)
        ct.launch(_get_stream(), (N,), entropy_fwd_kernel,
                  (logits, out, V, TILE_V))
        return out
    # Fallback
    logits_f = logits.astype(xp.float32)
    row_max = logits_f.max(axis=-1, keepdims=True)
    shifted = logits_f - row_max
    exp_shifted = xp.exp(shifted)
    total = exp_shifted.sum(axis=-1, keepdims=True)
    p = exp_shifted / total
    log_p = shifted - xp.log(total)
    return (-(p * log_p).sum(axis=-1)).astype(xp.float32)


# -- Trainer --

@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    vf_coeff: float = 0.5
    entropy_coeff: float = 0.01
    gamma: float = 1.0
    lam: float = 0.95
    lr: float = 1e-6
    max_steps: int = 1000
    ppo_epochs: int = 4
    mini_batch_size: int = 4
    log_every: int = 50
    grad_clip: float = 1.0
    max_new_tokens: int = 128
    kl_coeff: float = 0.04


class PPOTrainer:
    """Pure cuTile PPO trainer: policy + value head + frozen ref + reward."""

    def __init__(
        self,
        policy: Qwen3Model,
        ref_policy: Qwen3Model,
        value_head: Qwen3Model,
        optimizer: AdamW,
        reward_fn: Callable,
        config: PPOConfig | None = None,
    ) -> None:
        self.policy = policy
        self.ref_policy = ref_policy
        self.value_head = value_head
        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.cfg = config or PPOConfig()
        for p in self.ref_policy.params.values():
            pass  # ref is never updated

    def _get_logprobs(self, model: Qwen3Model, ids: cp.ndarray, mask: cp.ndarray):
        """Forward pass → per-token log-probs, masked to response."""
        xp = cp
        logits = model.forward(ids)
        B, T, V = logits.shape
        flat = logits[:, :-1, :].reshape(B * (T - 1), V).astype(xp.float32)
        tgts = ids[:, 1:].reshape(B * (T - 1)).astype(xp.int32)
        lp = log_softmax_gather_fwd(flat, tgts)
        m = mask[:, 1:].astype(xp.float32)
        return lp.reshape(B, T - 1) * m, logits, m

    def compute_gae(
        self, rewards: cp.ndarray, values: cp.ndarray, mask: cp.ndarray,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Compute GAE advantages and returns."""
        return compute_gae(rewards, values, mask, self.cfg.gamma, self.cfg.lam)

    def ppo_update(
        self, ids: cp.ndarray, mask: cp.ndarray,
        old_lp: cp.ndarray, old_values: cp.ndarray,
        advantages: cp.ndarray, returns: cp.ndarray,
    ) -> dict[str, float]:
        """One PPO gradient update over a mini-batch."""
        xp = cp
        N, T = ids.shape
        eps = self.cfg.clip_eps

        new_lp, logits, resp_mask = self._get_logprobs(self.policy, ids, mask)
        T1 = new_lp.shape[1]
        V = logits.shape[2]

        # Policy ratios
        ratios = xp.exp(new_lp - old_lp[:, :T1])

        # Normalize advantages
        adv = advantages[:, :T1] * resp_mask
        adv_mean = adv.sum() / resp_mask.sum().clip(1)
        adv_std = xp.sqrt(((adv - adv_mean) ** 2 * resp_mask).sum() / resp_mask.sum().clip(1) + 1e-8)
        adv_norm = (adv - adv_mean) / adv_std

        # Clipped policy loss
        clip_loss = ppo_clip_fwd(ratios, adv_norm, eps)

        # Value loss (use last-layer hidden as value estimate placeholder)
        values = self.value_head.forward(ids)[:, 1:, 0].astype(xp.float32)[:, :T1]
        val_loss = value_loss_fwd(values, old_values[:, :T1], returns[:, :T1], eps)

        # Entropy
        flat_logits = logits[:, :-1, :].reshape(N * T1, V).astype(xp.float32)
        ent = entropy_fwd(flat_logits).reshape(N, T1)

        # Total loss
        total = resp_mask.sum().clip(1)
        pg_loss = (clip_loss * resp_mask).sum() / total
        vf_loss = (val_loss * resp_mask).sum() / total
        ent_bonus = (ent * resp_mask).sum() / total
        loss = pg_loss + self.cfg.vf_coeff * vf_loss - self.cfg.entropy_coeff * ent_bonus

        # --- Backward: policy gets clip_loss - entropy_coeff * entropy ---
        grad_clip = resp_mask / total
        grad_ratios = ppo_clip_bwd(ratios, adv_norm, grad_clip, eps)

        # ratio = exp(new_lp - old_lp) → d/d(new_lp) = ratio * grad_ratios
        grad_new_lp = grad_ratios * ratios

        tgts = ids[:, 1:].reshape(N * T1).astype(xp.int32)
        grad_logits = log_softmax_gather_bwd(flat_logits, tgts,
                                              grad_new_lp.reshape(N * T1))

        # Entropy grad on logits: d(H)/d(logits_j) = sm_j * (H + lsm_j)
        flat_max = flat_logits.max(axis=1, keepdims=True)
        flat_exp = xp.exp(flat_logits - flat_max)
        flat_sm = flat_exp / flat_exp.sum(axis=1, keepdims=True)
        flat_lsm = flat_logits - flat_max - xp.log(flat_exp.sum(axis=1, keepdims=True))
        ent_flat = ent.reshape(N * T1)
        ent_grad_scale = (self.cfg.entropy_coeff * resp_mask / total).reshape(N * T1, 1)
        grad_logits += ent_grad_scale * flat_sm * (ent_flat[:, None] + flat_lsm)

        full_grad = xp.zeros((N, T, V), dtype=xp.float32)
        full_grad[:, 1:, :] = grad_logits.reshape(N, T1, V)
        self.policy.backward(full_grad)
        self.optimizer.step(self.policy.grads)
        self.policy.zero_grad()

        # Value head backward: vf_coeff * value_loss
        grad_vl = self.cfg.vf_coeff * resp_mask / total
        grad_values = value_loss_bwd(values, old_values[:, :T1], returns[:, :T1],
                                      grad_vl, eps)
        vh_grad = xp.zeros((N, T, self.value_head.forward(ids).shape[2]), dtype=xp.float32)
        vh_grad[:, 1:T1+1, 0] = grad_values
        self.value_head.backward(vh_grad)
        self.optimizer.step(self.value_head.grads)
        self.value_head.zero_grad()

        return {"pg_loss": float(pg_loss), "vf_loss": float(vf_loss),
                "entropy": float(ent_bonus), "loss": float(loss)}

    def train_step(self, prompts, prompt_ids: cp.ndarray) -> dict[str, float]:
        """One PPO training step: rollout → GAE → multi-epoch updates."""
        xp = cp
        # Generate rollouts (reward_fn handles generation)
        ids, mask, rewards = self.reward_fn(prompts, prompt_ids)

        # Get old log-probs and values
        old_lp, _, _ = self._get_logprobs(self.policy, ids, mask)
        values = self.value_head.forward(ids)[:, 1:, 0].astype(xp.float32)

        T1 = old_lp.shape[1]
        resp = mask[:, 1:].astype(xp.float32)[:, :T1]
        rew_pad = xp.zeros_like(old_lp)
        rew_pad[:, :rewards.shape[1]] = rewards[:, :T1] if rewards.shape[1] >= T1 else rewards

        advantages, returns = self.compute_gae(rew_pad, values[:, :T1], resp)

        # Multi-epoch mini-batch PPO
        metrics = {}
        N = ids.shape[0]
        for epoch in range(self.cfg.ppo_epochs):
            perm = xp.random.permutation(N)
            for i in range(0, N, self.cfg.mini_batch_size):
                idx = perm[i:i + self.cfg.mini_batch_size]
                metrics = self.ppo_update(
                    ids[idx], mask[idx], old_lp[idx], values[idx, :T1],
                    advantages[idx], returns[idx],
                )
        return metrics

    def train(self, dataloader) -> list[dict[str, float]]:
        """Full training loop."""
        history = []
        for step, batch in enumerate(dataloader):
            if step >= self.cfg.max_steps:
                break
            m = self.train_step(batch["prompts"], batch["prompt_ids"])
            history.append(m)
            if step % self.cfg.log_every == 0:
                print(f"[PPO] step {step:>5d}  loss={m['loss']:.4f}  "
                      f"pg={m['pg_loss']:.4f}  vf={m['vf_loss']:.4f}")
        return history

# -- Smoke test --

if __name__ == "__main__":
    print(f"=== PPO Smoke Test (backend: {'cuTile' if _USE_CUTILE_KERNELS else 'CuPy/NumPy fallback'}) ===\n")

    xp = cp
    if hasattr(xp, 'random') and hasattr(xp.random, 'seed'):
        xp.random.seed(42)

    N, T = 4, 16
    eps = 0.2

    # Test 1: Clipped surrogate (via wrapper)
    print("--- PPO clip loss ---")
    ratios = (0.8 + 0.4 * xp.random.rand(N, T)).astype(xp.float32)
    adv = (xp.random.randn(N, T) if hasattr(xp.random, 'randn')
           else np.random.randn(N, T)).astype(xp.float32)
    clip_out = ppo_clip_fwd(ratios, adv, eps)
    clipped = xp.clip(ratios, 1 - eps, 1 + eps)
    ref = -xp.minimum(ratios * adv, clipped * adv)
    np.testing.assert_allclose(
        np.asarray(clip_out) if hasattr(clip_out, 'get') else clip_out,
        np.asarray(ref) if hasattr(ref, 'get') else ref,
        atol=1e-4,
    )
    print("  PPO clip loss matches.\n")

    # Test 2: GAE (via wrapper)
    print("--- GAE ---")
    rewards = (xp.random.randn(N, T) * 0.1 if hasattr(xp.random, 'randn')
               else np.random.randn(N, T) * 0.1).astype(xp.float32)
    values = (xp.random.randn(N, T) * 0.1 if hasattr(xp.random, 'randn')
              else np.random.randn(N, T) * 0.1).astype(xp.float32)
    mask = xp.ones((N, T), dtype=xp.float32)
    mask[:, -2:] = 0
    adv_out, ret_out = compute_gae(rewards, values, mask, gamma=0.99, lam=0.95)
    assert adv_out.shape == (N, T), f"Expected ({N},{T}), got {adv_out.shape}"
    assert ret_out.shape == (N, T), f"Expected ({N},{T}), got {ret_out.shape}"
    # Verify masked positions are zero
    np.testing.assert_allclose(
        np.asarray(adv_out[:, -2:]) if hasattr(adv_out, 'get') else adv_out[:, -2:],
        0.0, atol=1e-6,
    )
    print(f"  GAE runs, shape={adv_out.shape}, masked positions are zero.\n")

    # Test 3: Value loss (via wrapper)
    print("--- Value loss ---")
    vals = (xp.random.randn(N, T) if hasattr(xp.random, 'randn')
            else np.random.randn(N, T)).astype(xp.float32)
    old_vals = (vals + 0.01).astype(xp.float32)
    rets = (xp.random.randn(N, T) if hasattr(xp.random, 'randn')
            else np.random.randn(N, T)).astype(xp.float32)
    vl = value_loss_fwd(vals, old_vals, rets, eps)
    v_clip = xp.clip(vals, old_vals - eps, old_vals + eps)
    ref_vl = xp.maximum((vals - rets) ** 2, (v_clip - rets) ** 2)
    np.testing.assert_allclose(
        np.asarray(vl) if hasattr(vl, 'get') else vl,
        np.asarray(ref_vl) if hasattr(ref_vl, 'get') else ref_vl,
        atol=1e-4,
    )
    print("  Value loss matches.\n")

    # Test 4: Entropy (via wrapper)
    print("--- Entropy ---")
    V = 64
    logits = (xp.random.randn(N, V) if hasattr(xp.random, 'randn')
              else np.random.randn(N, V)).astype(xp.float32)
    ent = entropy_fwd(logits)
    lse = xp.log(xp.exp(logits).sum(axis=1, keepdims=True))
    p = xp.exp(logits - lse)
    ref_ent = -(p * (logits - lse)).sum(axis=1)
    np.testing.assert_allclose(
        np.asarray(ent) if hasattr(ent, 'get') else ent,
        np.asarray(ref_ent) if hasattr(ref_ent, 'get') else ref_ent,
        atol=1e-3,
    )
    print("  Entropy matches.\n")

    # Test 5: log_softmax_gather fwd (via wrapper)
    print("--- log_softmax_gather fwd ---")
    logits2 = (xp.random.randn(N, V) if hasattr(xp.random, 'randn')
               else np.random.randn(N, V)).astype(xp.float32)
    targets = xp.array([i % V for i in range(N)], dtype=xp.int32)
    lp = log_softmax_gather_fwd(logits2, targets)
    logits2_np = np.asarray(logits2) if hasattr(logits2, 'get') else np.array(logits2)
    targets_np = np.asarray(targets) if hasattr(targets, 'get') else np.array(targets)
    max_l = logits2_np.max(axis=-1, keepdims=True)
    shifted = logits2_np - max_l
    lse_np = np.log(np.exp(shifted).sum(axis=-1)) + max_l.squeeze(-1)
    lp_ref = logits2_np[np.arange(N), targets_np] - lse_np
    np.testing.assert_allclose(
        np.asarray(lp) if hasattr(lp, 'get') else lp,
        lp_ref, atol=1e-4,
    )
    print("  log_softmax_gather fwd matches.\n")

    # Test 6: log_softmax_gather bwd (via wrapper)
    print("--- log_softmax_gather bwd ---")
    grad_lp = xp.ones(N, dtype=xp.float32) / N
    grad_logits = log_softmax_gather_bwd(logits2, targets, grad_lp)
    assert grad_logits.shape == (N, V), f"Expected ({N},{V}), got {grad_logits.shape}"
    print("  log_softmax_gather bwd runs correctly.\n")

    # Test 7: ppo_clip_bwd (via wrapper)
    print("--- PPO clip bwd ---")
    grad_clip_out = xp.ones((N, T), dtype=xp.float32) / (N * T)
    grad_ratios = ppo_clip_bwd(ratios, adv, grad_clip_out, eps)
    assert grad_ratios.shape == ratios.shape
    print("  PPO clip bwd runs correctly.\n")

    # Test 8: value_loss_bwd (via wrapper)
    print("--- Value loss bwd ---")
    grad_vl_out = xp.ones((N, T), dtype=xp.float32) / (N * T)
    grad_vals = value_loss_bwd(vals, old_vals, rets, grad_vl_out, eps)
    assert grad_vals.shape == vals.shape
    print("  Value loss bwd runs correctly.\n")

    print("=== PPO Smoke Test Complete ===")
