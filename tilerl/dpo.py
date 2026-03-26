"""DPO -- Direct Preference Optimization.

Paper: https://arxiv.org/abs/2305.18290 (Rafailov et al., NeurIPS 2023)

Key idea: Bypass explicit reward modelling by directly optimizing a policy
from preference pairs using the closed-form relationship between the optimal
policy and the reward function under a KL-constrained objective.

Loss:
    L_DPO = -E[ log sigma( beta * ( sum_t log pi(y_w_t|x) - log pi_ref(y_w_t|x)
                                   - sum_t log pi(y_l_t|x) + log pi_ref(y_l_t|x) ) ) ]

where (y_w, y_l) is a (chosen, rejected) pair, pi_ref is the frozen reference
policy, and beta controls deviation from pi_ref.

Supports cuTile, CuPy, and NumPy backends via TILERL_BACKEND env var.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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
from optim import AdamW, AdamWConfig, clip_grad_norm

# ---------------------------------------------------------------------------
# cuTile kernels — log-softmax + gather (reused from sft.py pattern)
# ---------------------------------------------------------------------------


@ct.kernel
def log_softmax_gather_fwd_kernel(
    logits: ct.Array,
    labels: ct.Array,
    output: ct.Array,
    TILE_V: ConstInt,
):
    """Fused log-softmax + gather per row: log p(label | logits).

    Args:
        logits: Shape (N, V).
        labels: Shape (N,).
        output: Shape (N,).
        TILE_V: Vocab tile width.
    """
    bid = ct.bid(0)
    nt = ct.num_tiles(logits, axis=1, shape=(1, TILE_V))
    mx = ct.full((1, 1), float("-inf"), dtype=ct.float32)
    for j in range(nt):
        t = ct.load(logits, index=(bid, j), shape=(1, TILE_V),
                    padding_mode=ct.PaddingMode.NEG_INF)
        mx = ct.maximum(mx, ct.max(t, axis=1, keepdims=True))
    se = ct.full((1, 1), 0.0, dtype=ct.float32)
    for j in range(nt):
        t = ct.load(logits, index=(bid, j), shape=(1, TILE_V),
                    padding_mode=ct.PaddingMode.NEG_INF)
        se += ct.sum(ct.exp(t - mx), axis=1, keepdims=True)
    lse = ct.log(se)
    a = ct.load(labels, index=(bid,), shape=(1,))
    tgt = ct.gather(logits, (bid, a)).astype(ct.float32)
    ct.store(output, index=(bid,), tile=tgt - mx.reshape((1,)) - lse.reshape((1,)))


@ct.kernel
def log_softmax_gather_bwd_kernel(
    logits: ct.Array,
    labels: ct.Array,
    grad_out: ct.Array,
    grad_logits: ct.Array,
    TILE_V: ConstInt,
):
    """Backward: grad = (one_hot(label) - softmax) * grad_out per row.

    Args:
        logits: Shape (N, V).
        labels: Shape (N,).
        grad_out: Shape (N,).
        grad_logits: Shape (N, V).
        TILE_V: Vocab tile width.
    """
    bid = ct.bid(0)
    nt = ct.num_tiles(logits, axis=1, shape=(1, TILE_V))
    mx = ct.full((1, 1), float("-inf"), dtype=ct.float32)
    for j in range(nt):
        t = ct.load(logits, index=(bid, j), shape=(1, TILE_V),
                    padding_mode=ct.PaddingMode.NEG_INF)
        mx = ct.maximum(mx, ct.max(t, axis=1, keepdims=True))
    se = ct.full((1, 1), 0.0, dtype=ct.float32)
    for j in range(nt):
        t = ct.load(logits, index=(bid, j), shape=(1, TILE_V),
                    padding_mode=ct.PaddingMode.NEG_INF)
        se += ct.sum(ct.exp(t - mx), axis=1, keepdims=True)
    g = ct.load(grad_out, index=(bid,), shape=(1,))
    a = ct.load(labels, index=(bid,), shape=(1,))
    for j in range(nt):
        t = ct.load(logits, index=(bid, j), shape=(1, TILE_V),
                    padding_mode=ct.PaddingMode.NEG_INF)
        softmax = ct.exp(t - mx) / se
        tile_offset = j * TILE_V
        idx = ct.arange(TILE_V, dtype=ct.int32) + tile_offset
        oh = ct.where(idx == a, 1.0, 0.0).reshape((1, TILE_V))
        grad = (oh - softmax) * g
        ct.store(grad_logits, index=(bid, j), tile=grad.astype(grad_logits.dtype))


# ---------------------------------------------------------------------------
# cuTile kernels — DPO-specific
# ---------------------------------------------------------------------------


@ct.kernel
def dpo_log_ratio_fwd_kernel(
    policy_lp: ct.Array,
    ref_lp: ct.Array,
    mask: ct.Array,
    output: ct.Array,
    TILE: ConstInt,
):
    """Per-sequence sum of log-ratios: sum_t (log pi - log pi_ref) * mask_t.

    Args:
        policy_lp: Policy per-token log-probs, shape (B, T).
        ref_lp: Reference per-token log-probs, shape (B, T).
        mask: Response mask, shape (B, T).
        output: Per-sequence log-ratio sum, shape (B,).
        TILE: Tile width along T.
    """
    bid = ct.bid(0)
    nt = ct.num_tiles(policy_lp, axis=1, shape=(1, TILE))
    acc = ct.full((1, TILE), 0.0, dtype=ct.float32)
    for j in range(nt):
        p = ct.load(policy_lp, index=(bid, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO)
        r = ct.load(ref_lp, index=(bid, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO)
        m = ct.load(mask, index=(bid, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO)
        acc += (p - r) * m
    total = ct.sum(acc, axis=1)
    ct.store(output, index=(bid,), tile=total)


@ct.kernel
def dpo_loss_fwd_kernel(
    log_ratio_w: ct.Array,
    log_ratio_l: ct.Array,
    beta: ct.Array,
    losses: ct.Array,
    TILE: ConstInt,
):
    """DPO loss: -log sigma(beta * (Delta_w - Delta_l)).

    Numerically stable: softplus(-x) = max(-x, 0) + log(1 + exp(-|x|))

    Args:
        log_ratio_w: Chosen log-ratio sums, shape (B,).
        log_ratio_l: Rejected log-ratio sums, shape (B,).
        beta: Scaling scalar, shape (1,).
        losses: Output losses, shape (B,).
        TILE: Batch tile size.
    """
    pid = ct.bid(0)
    w = ct.load(log_ratio_w, index=(pid,), shape=(TILE,))
    l = ct.load(log_ratio_l, index=(pid,), shape=(TILE,))  # noqa: E741
    b = ct.load(beta, index=(0,), shape=(1,))
    logit = b * (w - l)
    neg_logit = -logit
    loss = ct.maximum(neg_logit, 0.0) + ct.log(1.0 + ct.exp(-ct.abs(neg_logit)))
    ct.store(losses, index=(pid,), tile=loss)


@ct.kernel
def dpo_loss_bwd_kernel(
    log_ratio_w: ct.Array,
    log_ratio_l: ct.Array,
    beta: ct.Array,
    grad_out: ct.Array,
    grad_w: ct.Array,
    grad_l: ct.Array,
    TILE: ConstInt,
):
    """Backward for DPO loss.

    d(loss)/d(logit) = sigma(logit) - 1
    d(logit)/d(w) = beta, d(logit)/d(l) = -beta

    Args:
        log_ratio_w: Chosen log-ratio sums, shape (B,).
        log_ratio_l: Rejected log-ratio sums, shape (B,).
        beta: Scaling scalar, shape (1,).
        grad_out: Upstream gradient, shape (B,).
        grad_w: Gradient w.r.t. chosen log-ratios, shape (B,).
        grad_l: Gradient w.r.t. rejected log-ratios, shape (B,).
        TILE: Batch tile size.
    """
    pid = ct.bid(0)
    w = ct.load(log_ratio_w, index=(pid,), shape=(TILE,))
    l = ct.load(log_ratio_l, index=(pid,), shape=(TILE,))  # noqa: E741
    b = ct.load(beta, index=(0,), shape=(1,))
    g = ct.load(grad_out, index=(pid,), shape=(TILE,))

    logit = b * (w - l)
    # sigma(x) = 1 / (1 + exp(-x)); stable: exp(x)/(1+exp(x)) if x>0
    sig = 1.0 / (1.0 + ct.exp(-logit))
    d_logit = (sig - 1.0) * g
    ct.store(grad_w, index=(pid,), tile=b * d_logit)
    ct.store(grad_l, index=(pid,), tile=-b * d_logit)


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


def log_softmax_gather_fwd(
    logits: cp.ndarray, labels: cp.ndarray,
) -> cp.ndarray:
    """Forward: per-token log-softmax + gather.

    Args:
        logits: (N, V) logits.
        labels: (N,) target indices.
    Returns:
        (N,) log-probs at target positions.
    """
    xp = cp
    N, V = logits.shape
    if _USE_CUTILE_KERNELS:
        TILE_V = min(1024, _next_pow2(V))
        out = xp.empty(N, dtype=xp.float32)
        ct.launch(_get_stream(), (N,), log_softmax_gather_fwd_kernel,
                  (logits, labels, out, TILE_V))
        return out
    # Fallback: stable log-softmax + gather
    logits_f = logits.astype(xp.float32)
    row_max = logits_f.max(axis=-1, keepdims=True)
    shifted = logits_f - row_max
    log_sum_exp = xp.log(xp.exp(shifted).sum(axis=-1)) + row_max.squeeze(-1)
    target_logits = logits_f[xp.arange(N), labels.astype(int)]
    return (target_logits - log_sum_exp).astype(xp.float32)


def log_softmax_gather_bwd(
    logits: cp.ndarray, labels: cp.ndarray, grad_out: cp.ndarray,
) -> cp.ndarray:
    """Backward of log_softmax+gather: grad * (one_hot - softmax).

    Args:
        logits: (N, V) logits.
        labels: (N,) target indices.
        grad_out: (N,) upstream gradient w.r.t. log-probs.
    Returns:
        (N, V) gradient w.r.t. logits.
    """
    xp = cp
    N, V = logits.shape
    if _USE_CUTILE_KERNELS:
        TILE_V = min(1024, _next_pow2(V))
        grad_logits = xp.zeros((N, V), dtype=logits.dtype)
        ct.launch(_get_stream(), (N,), log_softmax_gather_bwd_kernel,
                  (logits, labels, grad_out, grad_logits, TILE_V))
        return grad_logits
    # Fallback
    logits_f = logits.astype(xp.float32)
    row_max = logits_f.max(axis=-1, keepdims=True)
    shifted = logits_f - row_max
    exp_shifted = xp.exp(shifted)
    softmax = exp_shifted / exp_shifted.sum(axis=-1, keepdims=True)
    one_hot = xp.zeros_like(logits_f)
    one_hot[xp.arange(N), labels.astype(int)] = 1.0
    return (grad_out[:, None] * (one_hot - softmax)).astype(logits.dtype)


def dpo_log_ratio_fwd(
    policy_lp: cp.ndarray, ref_lp: cp.ndarray, mask: cp.ndarray,
) -> cp.ndarray:
    """Per-sequence sum of log-ratios: sum_t (log pi - log pi_ref) * mask_t.

    Args:
        policy_lp: (B, T) policy per-token log-probs.
        ref_lp: (B, T) reference per-token log-probs.
        mask: (B, T) response mask.
    Returns:
        (B,) per-sequence log-ratio sums.
    """
    xp = cp
    B, T = policy_lp.shape
    if _USE_CUTILE_KERNELS:
        TILE = min(1024, _next_pow2(T))
        out = xp.empty(B, dtype=xp.float32)
        ct.launch(_get_stream(), (B,), dpo_log_ratio_fwd_kernel,
                  (policy_lp, ref_lp, mask, out, TILE))
        return out
    # Fallback: pure array ops
    return ((policy_lp - ref_lp) * mask).sum(axis=1).astype(xp.float32)


def dpo_loss_fwd(
    log_ratio_w: cp.ndarray, log_ratio_l: cp.ndarray, beta: float,
) -> cp.ndarray:
    """DPO loss: -log sigma(beta * (Delta_w - Delta_l)).

    Numerically stable via softplus: max(-x, 0) + log(1 + exp(-|x|))

    Args:
        log_ratio_w: (B,) chosen log-ratio sums.
        log_ratio_l: (B,) rejected log-ratio sums.
        beta: Scaling factor.
    Returns:
        (B,) per-sample losses.
    """
    xp = cp
    B = log_ratio_w.shape[0]
    if _USE_CUTILE_KERNELS:
        beta_t = xp.array([beta], dtype=xp.float32)
        losses = xp.empty(B, dtype=xp.float32)
        ct.launch(_get_stream(), (B,), dpo_loss_fwd_kernel,
                  (log_ratio_w, log_ratio_l, beta_t, losses, 1))
        return losses
    # Fallback: numerically stable softplus(-logit)
    logit = beta * (log_ratio_w - log_ratio_l)
    neg_logit = -logit
    return (xp.maximum(neg_logit, 0.0) + xp.log(1.0 + xp.exp(-xp.abs(neg_logit)))).astype(xp.float32)


def dpo_loss_bwd(
    log_ratio_w: cp.ndarray, log_ratio_l: cp.ndarray,
    beta: float, grad_out: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Backward for DPO loss.

    d(loss)/d(logit) = sigma(logit) - 1
    d(logit)/d(w) = beta, d(logit)/d(l) = -beta

    Args:
        log_ratio_w: (B,) chosen log-ratio sums.
        log_ratio_l: (B,) rejected log-ratio sums.
        beta: Scaling factor.
        grad_out: (B,) upstream gradient.
    Returns:
        (grad_w, grad_l): gradients w.r.t. chosen and rejected log-ratios.
    """
    xp = cp
    B = log_ratio_w.shape[0]
    if _USE_CUTILE_KERNELS:
        beta_t = xp.array([beta], dtype=xp.float32)
        grad_w = xp.empty(B, dtype=xp.float32)
        grad_l = xp.empty(B, dtype=xp.float32)
        ct.launch(_get_stream(), (B,), dpo_loss_bwd_kernel,
                  (log_ratio_w, log_ratio_l, beta_t, grad_out,
                   grad_w, grad_l, 1))
        return grad_w, grad_l
    # Fallback
    logit = beta * (log_ratio_w - log_ratio_l)
    sig = 1.0 / (1.0 + xp.exp(-logit))
    d_logit = (sig - 1.0) * grad_out
    grad_w = (beta * d_logit).astype(xp.float32)
    grad_l = (-beta * d_logit).astype(xp.float32)
    return grad_w, grad_l


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class DPOConfig:
    """Hyperparameters for DPO training."""

    beta: float = 0.1
    lr: float = 1e-6
    max_steps: int = 1000
    log_every: int = 50
    grad_clip: float = 1.0
    weight_decay: float = 0.01


class DPOTrainer:
    """DPO trainer — supports cuTile, CuPy, and NumPy backends.

    Trains a Qwen3 policy against a frozen reference model using the DPO
    preference loss. No explicit reward model needed.
    """

    def __init__(
        self,
        model: Qwen3Model,
        ref_model: Qwen3Model,
        config: DPOConfig | None = None,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.cfg = config or DPOConfig()
        self.optimizer = AdamW(
            model.parameters(),
            config=AdamWConfig(lr=self.cfg.lr, weight_decay=self.cfg.weight_decay),
        )

    def _get_logprobs_and_mask(
        self, model: Qwen3Model, input_ids: cp.ndarray, labels: cp.ndarray
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Forward through model, compute per-token log-probs.

        Returns:
            logprobs: shape (B, T-1), masked to response tokens.
            mask: shape (B, T-1).
        """
        B, T = input_ids.shape
        logits = model.forward(input_ids)  # (B, T, V)
        V = logits.shape[2]

        shift_logits = logits[:, :-1, :].reshape(-1, V)
        shift_labels = labels[:, 1:]
        mask = (shift_labels != -100).astype(cp.float32)
        safe_labels = cp.clip(shift_labels, 0, V - 1).astype(cp.int32)
        flat_labels = safe_labels.reshape(-1)

        logprobs_flat = log_softmax_gather_fwd(shift_logits, flat_labels)

        logprobs = logprobs_flat.reshape(B, T - 1) * mask
        return logprobs, mask

    def train_step(
        self,
        chosen_ids: cp.ndarray,
        chosen_labels: cp.ndarray,
        rejected_ids: cp.ndarray,
        rejected_labels: cp.ndarray,
    ) -> dict[str, float]:
        """One DPO training step.

        Args:
            chosen_ids: Preferred token ids, shape (B, T_w).
            chosen_labels: Preferred labels, shape (B, T_w).
            rejected_ids: Dispreferred token ids, shape (B, T_l).
            rejected_labels: Dispreferred labels, shape (B, T_l).

        Returns:
            Metrics dict with loss, margin, accuracy.
        """
        V = self.model.config.vocab_size
        B, T_w = chosen_ids.shape
        _, T_l = rejected_ids.shape

        # Reference model log-probs (no backward needed, cache irrelevant)
        ref_lp_w, _ = self._get_logprobs_and_mask(self.ref_model, chosen_ids, chosen_labels)
        ref_lp_l, _ = self._get_logprobs_and_mask(self.ref_model, rejected_ids, rejected_labels)

        # --- Chosen: forward -> compute logprobs -> backward (cache-safe) ---
        logits_w = self.model.forward(chosen_ids)  # caches chosen activations
        shift_logits_w = logits_w[:, :-1, :].reshape(-1, V)
        shift_labels_w = cp.clip(chosen_labels[:, 1:], 0, V - 1).astype(cp.int32)
        mask_w = (chosen_labels[:, 1:] != -100).astype(cp.float32)
        flat_labels_w = shift_labels_w.reshape(-1)
        N_w = shift_logits_w.shape[0]

        pol_lp_w_flat = log_softmax_gather_fwd(shift_logits_w, flat_labels_w)
        pol_lp_w = pol_lp_w_flat.reshape(B, T_w - 1) * mask_w

        # --- Rejected: forward -> compute logprobs (defer backward) ---
        logits_l = self.model.forward(rejected_ids)  # caches rejected activations
        shift_logits_l = logits_l[:, :-1, :].reshape(-1, V)
        shift_labels_l = cp.clip(rejected_labels[:, 1:], 0, V - 1).astype(cp.int32)
        mask_l = (rejected_labels[:, 1:] != -100).astype(cp.float32)
        flat_labels_l = shift_labels_l.reshape(-1)
        N_l = shift_logits_l.shape[0]

        pol_lp_l_flat = log_softmax_gather_fwd(shift_logits_l, flat_labels_l)
        pol_lp_l = pol_lp_l_flat.reshape(B, T_l - 1) * mask_l

        # Per-sequence log-ratio sums
        T1_w, T1_l = T_w - 1, T_l - 1

        lr_w = dpo_log_ratio_fwd(pol_lp_w, ref_lp_w, mask_w)
        lr_l = dpo_log_ratio_fwd(pol_lp_l, ref_lp_l, mask_l)

        # DPO loss
        losses = dpo_loss_fwd(lr_w, lr_l, self.cfg.beta)
        loss_val = float(losses.mean())

        # Metrics
        margin = float((self.cfg.beta * (lr_w - lr_l)).mean())
        accuracy = float((lr_w > lr_l).astype(cp.float32).mean())

        # ---- Backward ----
        grad_out = cp.full(B, 1.0 / B, dtype=cp.float32)
        grad_lr_w, grad_lr_l = dpo_loss_bwd(lr_w, lr_l, self.cfg.beta, grad_out)

        # Backward through log-ratio: d(lr)/d(policy_lp_t) = mask_t
        grad_pol_lp_w = grad_lr_w[:, None] * mask_w
        grad_pol_lp_l = grad_lr_l[:, None] * mask_l

        # Rejected backward first (cache still holds rejected activations)
        grad_slogits_l = log_softmax_gather_bwd(
            shift_logits_l, flat_labels_l,
            (grad_pol_lp_l * mask_l).reshape(-1))
        grad_logits_l = cp.zeros_like(logits_l)
        grad_logits_l[:, :-1, :] = grad_slogits_l.reshape(B, T1_l, V)
        grads_l = self.model.backward(grad_logits_l)  # uses rejected cache

        # Chosen: re-forward to restore cache, then backward
        logits_w = self.model.forward(chosen_ids)  # restores chosen cache
        shift_logits_w = logits_w[:, :-1, :].reshape(-1, V)
        grad_slogits_w = log_softmax_gather_bwd(
            shift_logits_w, flat_labels_w,
            (grad_pol_lp_w * mask_w).reshape(-1))
        grad_logits_w = cp.zeros_like(logits_w)
        grad_logits_w[:, :-1, :] = grad_slogits_w.reshape(B, T1_w, V)
        grads_w = self.model.backward(grad_logits_w)  # uses chosen cache

        # Accumulate gradients from both paths
        grads = {k: grads_w.get(k, 0) + grads_l.get(k, 0) for k in grads_w}

        clip_grad_norm(grads, self.cfg.grad_clip)
        self.optimizer.step(grads)

        return {"loss": loss_val, "margin": margin, "accuracy": accuracy}

    def train(self, data_iter) -> list[dict[str, float]]:
        """Full training loop. Batches yield dicts with keys:
        chosen_ids, chosen_labels, rejected_ids, rejected_labels."""
        history: list[dict[str, float]] = []
        for step, batch in enumerate(data_iter):
            if step >= self.cfg.max_steps:
                break
            metrics = self.train_step(
                batch["chosen_ids"], batch["chosen_labels"],
                batch["rejected_ids"], batch["rejected_labels"],
            )
            history.append(metrics)
            if step % self.cfg.log_every == 0:
                print(f"[DPO] step {step:>5d}  loss={metrics['loss']:.4f}  "
                      f"margin={metrics['margin']:.4f}  acc={metrics['accuracy']:.2f}")
        return history


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _backend_label = "cuTile" if _USE_CUTILE_KERNELS else "CuPy/NumPy fallback"
    print(f"=== DPO Smoke Test (backend: {_backend_label}) ===\n")

    xp = cp
    np.random.seed(42)

    # --- Test 1: Log-ratio (via wrapper) ---
    print("--- Log-ratio ---")
    B, T = 4, 32
    pol_lp = np.random.randn(B, T).astype(np.float32) * 0.2
    ref_lp_np = np.random.randn(B, T).astype(np.float32) * 0.2
    mask = np.ones((B, T), dtype=np.float32)
    for i in range(B):
        plen = np.random.randint(4, T // 2)
        mask[i, :plen] = 0.0
        pol_lp[i, :plen], ref_lp_np[i, :plen] = 0.0, 0.0
    ref_lr = ((pol_lp - ref_lp_np) * mask).sum(axis=1)

    result = dpo_log_ratio_fwd(
        xp.array(pol_lp), xp.array(ref_lp_np), xp.array(mask))
    result_np = np.asarray(result) if hasattr(result, 'get') else np.array(result)
    if np.allclose(result_np, ref_lr, atol=1e-4):
        print("  Log-ratio matches.\n")
    else:
        print(f"  MISMATCH -- max diff: {np.abs(result_np - ref_lr).max():.6f}\n")

    # --- Test 2: DPO loss (via wrapper) ---
    print("--- DPO loss ---")
    beta = 0.1
    lr_w_np = np.random.randn(B).astype(np.float32)
    lr_l_np = np.random.randn(B).astype(np.float32)
    logit = beta * (lr_w_np - lr_l_np)
    ref_loss = np.log(1 + np.exp(-logit))  # -log sigmoid(logit)

    result = dpo_loss_fwd(xp.array(lr_w_np), xp.array(lr_l_np), beta)
    result_np = np.asarray(result) if hasattr(result, 'get') else np.array(result)
    if np.allclose(result_np, ref_loss, atol=1e-4):
        print("  DPO loss matches.\n")
    else:
        print(f"  MISMATCH -- max diff: {np.abs(result_np - ref_loss).max():.6f}\n")

    # --- Test 3: DPO loss backward (via wrapper) ---
    print("--- DPO loss backward ---")
    sig = 1 / (1 + np.exp(-logit))
    grad_out_np = (np.ones(B) / B).astype(np.float32)
    ref_gw = beta * (sig - 1) * grad_out_np
    ref_gl = -beta * (sig - 1) * grad_out_np

    gw, gl = dpo_loss_bwd(
        xp.array(lr_w_np), xp.array(lr_l_np), beta, xp.array(grad_out_np))
    gw_np = np.asarray(gw) if hasattr(gw, 'get') else np.array(gw)
    gl_np = np.asarray(gl) if hasattr(gl, 'get') else np.array(gl)
    if np.allclose(gw_np, ref_gw, atol=1e-5) and np.allclose(gl_np, ref_gl, atol=1e-5):
        print("  DPO backward matches.")
    else:
        print(f"  MISMATCH -- max diff w: {np.abs(gw_np - ref_gw).max():.6f}")
    # Check d_w = -d_l
    if np.allclose(gw_np, -gl_np, atol=1e-6):
        print("  Chosen and rejected gradients are opposite (correct).\n")
    else:
        print("  WARNING: gradients are not opposite.\n")

    # --- Test 4: log_softmax_gather (via wrapper) ---
    print("--- log_softmax_gather ---")
    V = 64
    logits_t = xp.array(np.random.randn(B, V).astype(np.float32))
    targets_t = xp.array([i % V for i in range(B)], dtype=xp.int32)

    lp = log_softmax_gather_fwd(logits_t, targets_t)

    logits_ref = np.asarray(logits_t) if hasattr(logits_t, 'get') else np.array(logits_t)
    targets_ref = np.asarray(targets_t) if hasattr(targets_t, 'get') else np.array(targets_t)
    max_l = logits_ref.max(axis=-1, keepdims=True)
    shifted = logits_ref - max_l
    lse = np.log(np.exp(shifted).sum(axis=-1)) + max_l.squeeze(-1)
    lp_ref = logits_ref[np.arange(B), targets_ref] - lse

    lp_np = np.asarray(lp) if hasattr(lp, 'get') else np.array(lp)
    np.testing.assert_allclose(lp_np, lp_ref, atol=1e-4)
    print("  log_softmax_gather matches.\n")

    # --- Test 5: log_softmax_gather backward (via wrapper) ---
    print("--- log_softmax_gather backward ---")
    grad_lp = xp.ones(B, dtype=xp.float32) / B
    grad_logits = log_softmax_gather_bwd(logits_t, targets_t, grad_lp)
    assert grad_logits.shape == (B, V)
    print("  log_softmax_gather backward runs correctly.\n")

    print("=== DPO Smoke Test Complete ===")
