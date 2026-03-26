"""SimPO — Simple Preference Optimization with a Reference-Free Reward.

Paper: https://arxiv.org/abs/2405.14734 (Meng et al., NeurIPS 2024)

Key idea: Use length-normalized average log-probability as implicit reward,
eliminating the need for a reference model. A target reward margin γ ensures
sufficient separation between preferred and dispreferred responses.

Loss:
    L_SimPO = -E[ log σ( β/|y_w| · log π(y_w|x) - β/|y_l| · log π(y_l|x) - γ ) ]

where β scales the reward, γ is the margin, and |y| is response length.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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
    """Per-token log_softmax(logits)[target]. 3-pass: max, log-sum-exp, gather."""
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
def simpo_avg_logprob_fwd_kernel(
    logprobs: ct.Array, mask: ct.Array, lengths: ct.Array,
    output: ct.Array, TILE: ConstInt,
):
    """Length-normalized avg log-prob: sum(logprobs * mask) / length."""
    bid = ct.bid(0)
    num_tiles = ct.num_tiles(logprobs, axis=1, shape=(1, TILE))

    acc = ct.full((1, TILE), 0.0, dtype=ct.float32)
    for j in range(num_tiles):
        lp = ct.load(logprobs, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        m = ct.load(mask, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        acc += lp * m

    total = ct.sum(acc, axis=1)
    length = ct.load(lengths, index=(bid,), shape=(1,))
    ct.store(output, index=(bid,), tile=total / length.astype(ct.float32))


@ct.kernel
def simpo_loss_fwd_kernel(
    avg_w: ct.Array, avg_l: ct.Array, losses: ct.Array,
    beta: ct.Array, gamma: ct.Array,
):
    """Per-pair SimPO loss: -log σ(β·avg_w - β·avg_l - γ).

    Uses numerically stable softplus: max(z,0) + log(1+exp(-|z|)).
    """
    pid = ct.bid(0)
    w = ct.load(avg_w, index=(pid,), shape=(1,))
    l = ct.load(avg_l, index=(pid,), shape=(1,))  # noqa: E741
    b = ct.load(beta, index=(0,), shape=(1,))
    g = ct.load(gamma, index=(0,), shape=(1,))

    logit = b * w - b * l - g
    neg_logit = -logit
    loss = ct.maximum(neg_logit, 0.0) + ct.log(1.0 + ct.exp(-ct.abs(neg_logit)))
    ct.store(losses, index=(pid,), tile=loss)


# ---------------------------------------------------------------------------
# cuTile kernels — backward
# ---------------------------------------------------------------------------


@ct.kernel
def simpo_loss_bwd_kernel(
    avg_w: ct.Array, avg_l: ct.Array,
    beta: ct.Array, gamma: ct.Array,
    grad_avg_w: ct.Array, grad_avg_l: ct.Array,
    B: ConstInt,
):
    """Backward of mean SimPO loss: d/d(avg) = ±β·(σ(logit)-1)/B."""
    pid = ct.bid(0)
    w = ct.load(avg_w, index=(pid,), shape=(1,))
    l = ct.load(avg_l, index=(pid,), shape=(1,))  # noqa: E741
    b = ct.load(beta, index=(0,), shape=(1,))
    g = ct.load(gamma, index=(0,), shape=(1,))

    logit = b * w - b * l - g
    # σ(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + ct.exp(-logit))
    d_logit = (sig - 1.0) / B

    ct.store(grad_avg_w, index=(pid,), tile=b * d_logit)
    ct.store(grad_avg_l, index=(pid,), tile=-b * d_logit)


@ct.kernel
def avg_logprob_bwd_kernel(
    grad_avg: ct.Array, mask: ct.Array, lengths: ct.Array,
    grad_logprobs: ct.Array, TILE: ConstInt,
):
    """Backward: gradient of avg log-prob w.r.t. per-token log-probs.

    d(avg)/d(lp_t) = mask_t / length
    """
    bid = ct.bid(0)
    num_tiles = ct.num_tiles(mask, axis=1, shape=(1, TILE))
    g = ct.load(grad_avg, index=(bid,), shape=(1,))
    length = ct.load(lengths, index=(bid,), shape=(1,))

    for j in range(num_tiles):
        m = ct.load(mask, index=(bid, j), shape=(1, TILE), padding_mode=PAD_ZERO)
        grad = g * m / length.astype(ct.float32)
        ct.store(grad_logprobs, index=(bid, j), tile=grad)


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

    # Recompute row_max and log_sum_exp
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

    # Compute gradient: g * (one_hot[target] - softmax)
    for j in range(num_tiles):
        tile = ct.load(logits, index=(pid, j), shape=(1, TILE_V), padding_mode=PAD_ZERO)
        vmask = (j * TILE_V + ct.arange(TILE_V, dtype=ct.int32)) < V
        softmax_tile = ct.where(vmask, ct.exp(tile - row_max) / total_sum_exp, 0.0)
        offsets = j * TILE_V + ct.arange(TILE_V, dtype=ct.int32)
        one_hot = ct.where(offsets == target, 1.0, 0.0)
        grad = g * (one_hot - softmax_tile)
        ct.store(grad_logits, index=(pid, j), tile=grad)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _next_pow2(n: int) -> int:
    """Next power of 2 >= n."""
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


def simpo_avg_logprob_fwd(
    logprobs: cp.ndarray, mask: cp.ndarray, lengths: cp.ndarray,
) -> cp.ndarray:
    """Length-normalized avg log-prob: sum(logprobs * mask) / length.

    Args:
        logprobs: (B, T) per-token log-probs.
        mask: (B, T) response mask.
        lengths: (B,) sequence lengths.
    Returns:
        (B,) average log-probs.
    """
    xp = cp
    B = logprobs.shape[0]
    T = logprobs.shape[1]
    if _USE_CUTILE_KERNELS:
        TILE = min(1024, _next_pow2(T))
        out = xp.empty(B, dtype=xp.float32)
        ct.launch(_get_stream(), (B,), simpo_avg_logprob_fwd_kernel,
                  (logprobs, mask, lengths, out, TILE))
        return out
    # Fallback
    return ((logprobs * mask).sum(axis=1) / lengths).astype(xp.float32)


def simpo_avg_logprob_bwd(
    grad_avg: cp.ndarray, mask: cp.ndarray, lengths: cp.ndarray,
) -> cp.ndarray:
    """Backward of avg log-prob w.r.t. per-token log-probs.

    Args:
        grad_avg: (B,) upstream gradient.
        mask: (B, T) response mask.
        lengths: (B,) sequence lengths.
    Returns:
        (B, T) gradient w.r.t. log-probs.
    """
    xp = cp
    B = mask.shape[0]
    T = mask.shape[1]
    if _USE_CUTILE_KERNELS:
        TILE = min(1024, _next_pow2(T))
        grad_logprobs = xp.empty_like(mask)
        ct.launch(_get_stream(), (B,), avg_logprob_bwd_kernel,
                  (grad_avg, mask, lengths, grad_logprobs, TILE))
        return grad_logprobs
    # Fallback
    return (grad_avg[:, None] * mask / lengths[:, None]).astype(xp.float32)


def simpo_loss_fwd(
    avg_w: cp.ndarray, avg_l: cp.ndarray, beta: float, gamma: float,
) -> cp.ndarray:
    """Per-pair SimPO loss: -log σ(β·avg_w - β·avg_l - γ).

    Args:
        avg_w: (B,) avg log-probs for chosen sequences.
        avg_l: (B,) avg log-probs for rejected sequences.
        beta: reward scaling factor.
        gamma: target margin.
    Returns:
        (B,) per-pair losses.
    """
    xp = cp
    B = avg_w.shape[0]
    if _USE_CUTILE_KERNELS:
        beta_t = xp.array([beta], dtype=xp.float32)
        gamma_t = xp.array([gamma], dtype=xp.float32)
        losses = xp.empty(B, dtype=xp.float32)
        ct.launch(_get_stream(), (B,), simpo_loss_fwd_kernel,
                  (avg_w, avg_l, losses, beta_t, gamma_t))
        return losses
    # Fallback: numerically stable -log σ(x) = softplus(-x)
    logit = beta * avg_w - beta * avg_l - gamma
    neg_logit = -logit
    return (xp.maximum(neg_logit, 0.0) + xp.log(1.0 + xp.exp(-xp.abs(neg_logit)))).astype(xp.float32)


def simpo_loss_bwd(
    avg_w: cp.ndarray, avg_l: cp.ndarray, beta: float, gamma: float, B: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Backward of mean SimPO loss w.r.t. avg log-probs.

    Args:
        avg_w: (B,) avg log-probs for chosen sequences.
        avg_l: (B,) avg log-probs for rejected sequences.
        beta: reward scaling factor.
        gamma: target margin.
        B: batch size (divisor for mean).
    Returns:
        (grad_avg_w, grad_avg_l) each shape (B,).
    """
    xp = cp
    if _USE_CUTILE_KERNELS:
        beta_t = xp.array([beta], dtype=xp.float32)
        gamma_t = xp.array([gamma], dtype=xp.float32)
        grad_avg_w = xp.empty(B, dtype=xp.float32)
        grad_avg_l = xp.empty(B, dtype=xp.float32)
        ct.launch(_get_stream(), (B,), simpo_loss_bwd_kernel,
                  (avg_w, avg_l, beta_t, gamma_t, grad_avg_w, grad_avg_l, B))
        return grad_avg_w, grad_avg_l
    # Fallback
    logit = beta * avg_w - beta * avg_l - gamma
    sig = 1.0 / (1.0 + xp.exp(-logit))
    d_logit = (sig - 1.0) / B
    grad_avg_w = (beta * d_logit).astype(xp.float32)
    grad_avg_l = (-beta * d_logit).astype(xp.float32)
    return grad_avg_w, grad_avg_l


def gather_logprobs(
    model: Qwen3Model, token_ids: cp.ndarray, response_mask: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Run model forward, compute per-token log-probs for response tokens.

    Args:
        model: Qwen3Model instance.
        token_ids: Shape (B, T), int32.
        response_mask: Shape (B, T), float32, 1 for response tokens.

    Returns:
        (logprobs, logits) where logprobs shape (B, T-1), logits shape (B, T-1, V).
    """
    xp = cp
    logits = model.forward(token_ids)  # (B, T, V)
    B, T, V = logits.shape

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].reshape(B * (T - 1), V).astype(xp.float32)
    shift_targets = token_ids[:, 1:].reshape(B * (T - 1)).astype(xp.int32)

    lp_flat = log_softmax_gather(shift_logits, shift_targets)

    logprobs = lp_flat.reshape(B, T - 1)
    # Apply response mask (shifted by 1)
    mask = response_mask[:, 1:].astype(xp.float32)
    logprobs = logprobs * mask
    return logprobs, shift_logits.reshape(B, T - 1, V), mask


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class SimPOConfig:
    """Hyperparameters for SimPO training."""
    beta: float = 2.0
    gamma: float = 1.0
    lr: float = 1e-6
    max_steps: int = 1000
    log_every: int = 50
    grad_clip: float = 1.0


class SimPOTrainer:
    """SimPO trainer supporting cuTile, CuPy, and NumPy backends."""

    def __init__(
        self, model: Qwen3Model, optimizer: AdamW,
        config: SimPOConfig | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.cfg = config or SimPOConfig()

    def train_step(
        self,
        chosen_ids: cp.ndarray, chosen_mask: cp.ndarray,
        rejected_ids: cp.ndarray, rejected_mask: cp.ndarray,
    ) -> float:
        """One SimPO training step.

        Args:
            chosen_ids: Preferred token ids, shape (B, T_w), int32.
            chosen_mask: Response mask for chosen, shape (B, T_w), float32.
            rejected_ids: Dispreferred token ids, shape (B, T_l), int32.
            rejected_mask: Response mask for rejected, shape (B, T_l), float32.

        Returns:
            Scalar loss value.
        """
        xp = cp
        B = chosen_ids.shape[0]

        # --- Forward: get per-token log-probs ---
        lp_w, logits_w, mask_w = gather_logprobs(self.model, chosen_ids, chosen_mask)
        lp_l, logits_l, mask_l = gather_logprobs(self.model, rejected_ids, rejected_mask)

        # Sequence lengths
        len_w = mask_w.sum(axis=1).astype(xp.float32)  # (B,)
        len_l = mask_l.sum(axis=1).astype(xp.float32)

        # Average log-probs
        avg_w = simpo_avg_logprob_fwd(lp_w, mask_w, len_w)
        avg_l = simpo_avg_logprob_fwd(lp_l, mask_l, len_l)

        # SimPO loss
        losses = simpo_loss_fwd(avg_w, avg_l, self.cfg.beta, self.cfg.gamma)
        loss_val = float(losses.mean())

        # --- Backward ---
        # 1. Loss → avg log-probs
        grad_avg_w, grad_avg_l = simpo_loss_bwd(
            avg_w, avg_l, self.cfg.beta, self.cfg.gamma, B)

        # 2. Avg log-probs → per-token log-probs
        grad_lp_w = simpo_avg_logprob_bwd(grad_avg_w, mask_w, len_w)
        grad_lp_l = simpo_avg_logprob_bwd(grad_avg_l, mask_l, len_l)

        # 3. Per-token log-probs → logits (log_softmax_gather backward)
        B_w, T_w_1, V = logits_w.shape
        B_l, T_l_1, V_l = logits_l.shape

        shift_targets_w = chosen_ids[:, 1:].reshape(B * T_w_1).astype(xp.int32)
        shift_targets_l = rejected_ids[:, 1:].reshape(B * T_l_1).astype(xp.int32)

        grad_logits_w = log_softmax_gather_bwd(
            logits_w.reshape(B * T_w_1, V), shift_targets_w,
            grad_lp_w.reshape(B * T_w_1),
        )
        grad_logits_l = log_softmax_gather_bwd(
            logits_l.reshape(B * T_l_1, V), shift_targets_l,
            grad_lp_l.reshape(B * T_l_1),
        )

        # 4. Backprop through model
        # Pad grad_logits back to (B, T, V) — zero grad for first position
        def _pad_grad(g: cp.ndarray, B: int, T: int, V: int) -> cp.ndarray:
            full = xp.zeros((B, T, V), dtype=xp.float32)
            full[:, 1:, :] = g.reshape(B, T - 1, V)
            return full

        grad_logits_w_full = _pad_grad(grad_logits_w, B, chosen_ids.shape[1], V)
        grad_logits_l_full = _pad_grad(grad_logits_l, B, rejected_ids.shape[1], V)

        # Accumulate gradients from both chosen and rejected
        self.model.backward(grad_logits_w_full)
        self.model.backward(grad_logits_l_full)

        # 5. Optimizer step
        self.optimizer.step(self.model.grads)
        self.model.zero_grad()

        return loss_val

    def train(self, dataloader) -> list[float]:
        """Full training loop.

        Each batch: {chosen_ids, chosen_mask, rejected_ids, rejected_mask}.
        """
        losses: list[float] = []
        for step, batch in enumerate(dataloader):
            if step >= self.cfg.max_steps:
                break
            loss = self.train_step(
                batch["chosen_ids"], batch["chosen_mask"],
                batch["rejected_ids"], batch["rejected_mask"],
            )
            losses.append(loss)
            if step % self.cfg.log_every == 0:
                print(f"[SimPO] step {step:>5d}  loss={loss:.4f}")
        return losses


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"=== SimPO Smoke Test (backend: {'cuTile' if _USE_CUTILE_KERNELS else 'CuPy/NumPy fallback'}) ===\n")

    xp = cp
    if hasattr(xp, 'random') and hasattr(xp.random, 'seed'):
        xp.random.seed(42)

    B, T, V = 4, 32, 128
    beta, gamma = 2.5, 1.0

    # Synthetic per-token log-probs and masks
    if hasattr(xp.random, 'randn'):
        logprobs_w = (xp.random.randn(B, T) * 0.1).astype(xp.float32)
        logprobs_l = (xp.random.randn(B, T) * 0.1).astype(xp.float32)
    else:
        logprobs_w = (np.random.randn(B, T) * 0.1).astype(np.float32)
        logprobs_l = (np.random.randn(B, T) * 0.1).astype(np.float32)

    mask_w = xp.ones((B, T), dtype=xp.float32)
    mask_l = xp.ones((B, T), dtype=xp.float32)

    if hasattr(xp.random, 'randint'):
        lengths_w = xp.random.randint(T // 2, T, (B,)).astype(xp.float32)
        lengths_l = xp.random.randint(T // 2, T, (B,)).astype(xp.float32)
    else:
        lengths_w = np.random.randint(T // 2, T, (B,)).astype(np.float32)
        lengths_l = np.random.randint(T // 2, T, (B,)).astype(np.float32)

    for i in range(B):
        k_w, k_l = int(lengths_w[i]), int(lengths_l[i])
        logprobs_w[i, k_w:] = 0.0
        logprobs_l[i, k_l:] = 0.0
        mask_w[i, k_w:] = 0.0
        mask_l[i, k_l:] = 0.0

    # --- Forward: avg log-probs ---
    print("--- avg log-prob kernel ---")
    avg_w = simpo_avg_logprob_fwd(logprobs_w, mask_w, lengths_w)
    avg_l = simpo_avg_logprob_fwd(logprobs_l, mask_l, lengths_l)

    # Reference
    avg_w_ref = (logprobs_w * mask_w).sum(axis=1) / lengths_w
    avg_l_ref = (logprobs_l * mask_l).sum(axis=1) / lengths_l

    print(f"avg_w result: {avg_w}")
    print(f"avg_w ref:    {avg_w_ref}")

    def _to_np(arr):
        return np.asarray(arr) if hasattr(arr, 'get') else np.array(arr)

    np.testing.assert_allclose(_to_np(avg_w), _to_np(avg_w_ref), atol=1e-4)
    print("avg_logprob kernel matches.\n")

    # --- Forward: SimPO loss ---
    print("--- SimPO loss kernel ---")
    losses = simpo_loss_fwd(avg_w, avg_l, beta, gamma)

    logit_ref = beta * avg_w_ref - beta * avg_l_ref - gamma
    loss_ref = xp.logaddexp(xp.zeros_like(logit_ref), -logit_ref)  # -log σ(x) = log(1+exp(-x))

    print(f"loss result: {losses}")
    print(f"loss ref:    {loss_ref}")
    np.testing.assert_allclose(_to_np(losses), _to_np(loss_ref), atol=1e-4)
    print("SimPO loss kernel matches.\n")

    # --- Backward: loss → avg log-probs ---
    print("--- loss backward kernel ---")
    grad_avg_w, grad_avg_l = simpo_loss_bwd(avg_w, avg_l, beta, gamma, B)

    sig_ref = 1.0 / (1.0 + xp.exp(-logit_ref))
    grad_avg_w_ref = (beta * (sig_ref - 1.0) / B).astype(xp.float32)
    grad_avg_l_ref = (-beta * (sig_ref - 1.0) / B).astype(xp.float32)

    np.testing.assert_allclose(_to_np(grad_avg_w), _to_np(grad_avg_w_ref), atol=1e-5)
    np.testing.assert_allclose(_to_np(grad_avg_l), _to_np(grad_avg_l_ref), atol=1e-5)
    print("Loss backward kernel matches.\n")

    # --- Backward: avg log-probs → per-token log-probs ---
    print("--- avg logprob backward kernel ---")
    grad_lp_w = simpo_avg_logprob_bwd(grad_avg_w, mask_w, lengths_w)

    grad_lp_w_ref = (grad_avg_w[:, None] * mask_w) / lengths_w[:, None]
    np.testing.assert_allclose(_to_np(grad_lp_w), _to_np(grad_lp_w_ref), atol=1e-5)
    print("Avg logprob backward kernel matches.\n")

    print("=== SimPO Smoke Test Complete ===")
