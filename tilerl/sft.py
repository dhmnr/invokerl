"""SFT -- Supervised Fine-Tuning.

Paper: Standard practice for instruction-tuning LLMs.
       See Ouyang et al. "Training language models to follow instructions with human
       feedback" (NeurIPS 2022) for context on SFT as the first post-training stage.

Loss:
    L_SFT = -1/|y| * sum_t log pi(y_t | x, y_{<t})

where x is the prompt, y is the target response, and |y| is the number of
response tokens (prompt tokens are masked out with label = -100).

Pure cuTile + CuPy. No PyTorch.
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
# cuTile kernels — log-softmax + gather (shared pattern)
# ---------------------------------------------------------------------------


@ct.kernel
def log_softmax_gather_fwd_kernel(
    logits: ct.Array,
    labels: ct.Array,
    output: ct.Array,
    TILE_V: ConstInt,
):
    """Fused log-softmax + gather: log p(label | logits) per position.

    Two-pass online softmax over vocabulary tiles, then gather at label index.
    Grid: one block per (B*T) position.

    Args:
        logits: Logits, shape (N, V) where N = B*(T-1).
        labels: Label indices, shape (N,). -100 for masked positions.
        output: Per-position log-probs, shape (N,). 0.0 for masked.
        TILE_V: Tile width along vocab dimension (power-of-2).
    """
    bid = ct.bid(0)
    nt = ct.num_tiles(logits, axis=1, shape=(1, TILE_V))

    # Pass 1: row-wise max for numerical stability
    mx = ct.full((1, 1), float("-inf"), dtype=ct.float32)
    for j in range(nt):
        t = ct.load(logits, index=(bid, j), shape=(1, TILE_V),
                    padding_mode=ct.PaddingMode.NEG_INF)
        mx = ct.maximum(mx, ct.max(t, axis=1, keepdims=True))

    # Pass 2: sum of exp(logit - max)
    se = ct.full((1, 1), 0.0, dtype=ct.float32)
    for j in range(nt):
        t = ct.load(logits, index=(bid, j), shape=(1, TILE_V),
                    padding_mode=ct.PaddingMode.NEG_INF)
        se += ct.sum(ct.exp(t - mx), axis=1, keepdims=True)
    lse = ct.log(se)

    # Gather logit at label position
    a = ct.load(labels, index=(bid,), shape=(1,))
    tgt = ct.gather(logits, (bid, a)).astype(ct.float32)

    # log_softmax(logits)[label] = logit[label] - max - log(sum_exp)
    result = tgt - mx.reshape((1,)) - lse.reshape((1,))
    ct.store(output, index=(bid,), tile=result)


@ct.kernel
def log_softmax_gather_bwd_kernel(
    logits: ct.Array,
    labels: ct.Array,
    grad_out: ct.Array,
    grad_logits: ct.Array,
    TILE_V: ConstInt,
):
    """Backward: d(log_softmax(x)[k]) / d(x_j) = (1_{j=k} - softmax(x)_j) * g.

    Each block handles one row. Writes gradient to grad_logits.

    Args:
        logits: Original logits, shape (N, V).
        labels: Label indices, shape (N,).
        grad_out: Upstream gradient, shape (N,).
        grad_logits: Output gradient, shape (N, V).
        TILE_V: Tile width along vocab (power-of-2).
    """
    bid = ct.bid(0)
    nt = ct.num_tiles(logits, axis=1, shape=(1, TILE_V))

    # Recompute row max + sum-exp (same as forward)
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

    # Gradient: (one_hot(label) - softmax) * grad_out, per vocab tile
    for j in range(nt):
        t = ct.load(logits, index=(bid, j), shape=(1, TILE_V),
                    padding_mode=ct.PaddingMode.NEG_INF)
        softmax = ct.exp(t - mx) / se
        # One-hot at label position within this tile
        tile_offset = j * TILE_V
        idx = ct.arange(TILE_V, dtype=ct.int32) + tile_offset
        oh = ct.where(idx == a, 1.0, 0.0).reshape((1, TILE_V))
        grad = (oh - softmax) * g
        ct.store(grad_logits, index=(bid, j), tile=grad.astype(grad_logits.dtype))


# ---------------------------------------------------------------------------
# cuTile kernels — SFT loss
# ---------------------------------------------------------------------------


@ct.kernel
def sft_masked_nll_fwd_kernel(
    logprobs: ct.Array,
    mask: ct.Array,
    losses: ct.Array,
    TILE: ConstInt,
):
    """Per-sequence mean NLL over response tokens.

    loss_b = -sum_t(logprobs[b,t] * mask[b,t]) / sum_t(mask[b,t])

    Args:
        logprobs: Per-token log-probs at label positions, shape (B, T).
        mask: Binary mask (1.0 for response tokens), shape (B, T).
        losses: Per-sequence mean NLL output, shape (B,).
        TILE: Tile width along T (power-of-2).
    """
    bid = ct.bid(0)
    num_tiles = ct.num_tiles(logprobs, axis=1, shape=(1, TILE))

    acc_nll = ct.full((1, TILE), 0.0, dtype=ct.float32)
    acc_count = ct.full((1, TILE), 0.0, dtype=ct.float32)

    for j in range(num_tiles):
        lp = ct.load(logprobs, index=(bid, j), shape=(1, TILE),
                     padding_mode=ct.PaddingMode.ZERO)
        m = ct.load(mask, index=(bid, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO)
        acc_nll += -lp * m
        acc_count += m

    total_nll = ct.sum(acc_nll, axis=1)
    count = ct.sum(acc_count, axis=1)
    loss = total_nll / (count + 1e-8)
    ct.store(losses, index=(bid,), tile=loss)


@ct.kernel
def sft_masked_nll_bwd_kernel(
    mask: ct.Array,
    count: ct.Array,
    grad_out: ct.Array,
    grad_logprobs: ct.Array,
    TILE: ConstInt,
):
    """Backward for masked mean NLL: d/d(lp[b,t]) = -mask[b,t] / count_b * g_b.

    Args:
        mask: Response mask, shape (B, T).
        count: Per-sequence token count, shape (B,).
        grad_out: Upstream gradient per sequence, shape (B,).
        grad_logprobs: Output gradient, shape (B, T).
        TILE: Tile width along T.
    """
    bid = ct.bid(0)
    num_tiles = ct.num_tiles(mask, axis=1, shape=(1, TILE))

    cnt = ct.load(count, index=(bid,), shape=(1,))
    g = ct.load(grad_out, index=(bid,), shape=(1,))

    for j in range(num_tiles):
        m = ct.load(mask, index=(bid, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO)
        grad = -m / (cnt + 1e-8) * g
        ct.store(grad_logprobs, index=(bid, j), tile=grad)


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
    """Forward: fused log-softmax + gather → per-position log-probs.

    Args:
        logits: (N, V) logits.
        labels: (N,) label indices. -100 for masked positions.
    Returns:
        (N,) log-probs at label positions. 0.0 for masked.
    """
    xp = cp
    N, V = logits.shape
    if _USE_CUTILE_KERNELS:
        out = xp.empty(N, dtype=xp.float32)
        TILE_V = min(1024, _next_pow2(V))
        ct.launch(_get_stream(), (N,), log_softmax_gather_fwd_kernel,
                  (logits, labels, out, TILE_V))
        return out
    # Fallback: stable log-softmax + gather
    logits_f = logits.astype(xp.float32)
    row_max = logits_f.max(axis=-1, keepdims=True)
    shifted = logits_f - row_max
    lse = xp.log(xp.exp(shifted).sum(axis=-1, keepdims=True))
    log_softmax = shifted - lse  # (N, V)
    return log_softmax[xp.arange(N), labels].astype(xp.float32)


def log_softmax_gather_backward(
    logits: cp.ndarray, labels: cp.ndarray, grad_out: cp.ndarray,
) -> cp.ndarray:
    """Backward of log_softmax+gather: (one_hot - softmax) * grad_out.

    Args:
        logits: (N, V) original logits.
        labels: (N,) label indices.
        grad_out: (N,) upstream gradient.
    Returns:
        (N, V) gradient w.r.t. logits.
    """
    xp = cp
    N, V = logits.shape
    if _USE_CUTILE_KERNELS:
        grad_logits = xp.zeros_like(logits)
        TILE_V = min(1024, _next_pow2(V))
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
    one_hot[xp.arange(N), labels] = 1.0
    return (grad_out[:, None] * (one_hot - softmax)).astype(logits.dtype)


def sft_masked_nll_fwd(
    logprobs: cp.ndarray, mask: cp.ndarray,
) -> cp.ndarray:
    """Per-sequence mean NLL over response tokens.

    loss_b = -sum_t(logprobs[b,t] * mask[b,t]) / sum_t(mask[b,t])

    Args:
        logprobs: (B, T) per-token log-probs at label positions.
        mask: (B, T) binary mask (1.0 for response tokens).
    Returns:
        (B,) per-sequence mean NLL.
    """
    xp = cp
    B, T = logprobs.shape
    if _USE_CUTILE_KERNELS:
        losses = xp.empty(B, dtype=xp.float32)
        TILE_T = min(1024, _next_pow2(T))
        ct.launch(_get_stream(), (B,), sft_masked_nll_fwd_kernel,
                  (logprobs, mask, losses, TILE_T))
        return losses
    # Fallback: pure array ops
    total_nll = (-logprobs * mask).sum(axis=1)
    count = mask.sum(axis=1)
    return (total_nll / (count + 1e-8)).astype(xp.float32)


def sft_masked_nll_bwd(
    mask: cp.ndarray, count: cp.ndarray, grad_out: cp.ndarray,
) -> cp.ndarray:
    """Backward for masked mean NLL: d/d(lp[b,t]) = -mask[b,t] / count_b * g_b.

    Args:
        mask: (B, T) response mask.
        count: (B,) per-sequence token count.
        grad_out: (B,) upstream gradient per sequence.
    Returns:
        (B, T) gradient w.r.t. logprobs.
    """
    xp = cp
    B, T = mask.shape
    if _USE_CUTILE_KERNELS:
        grad_logprobs = xp.empty((B, T), dtype=xp.float32)
        TILE_T = min(1024, _next_pow2(T))
        ct.launch(_get_stream(), (B,), sft_masked_nll_bwd_kernel,
                  (mask, count, grad_out, grad_logprobs, TILE_T))
        return grad_logprobs
    # Fallback: pure array ops
    return (-mask / (count[:, None] + 1e-8) * grad_out[:, None]).astype(xp.float32)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class SFTConfig:
    """Hyperparameters for SFT training."""

    lr: float = 2e-5
    max_steps: int = 1000
    log_every: int = 50
    grad_clip: float = 1.0
    weight_decay: float = 0.01


class SFTTrainer:
    """Supervised Fine-Tuning trainer — pure cuTile + CuPy.

    Trains a Qwen3 model to minimize per-token cross-entropy on
    instruction-response pairs. Prompt tokens are masked so only
    response tokens contribute to the loss.
    """

    def __init__(
        self,
        model: Qwen3Model,
        config: SFTConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = config or SFTConfig()
        self.optimizer = AdamW(
            model.parameters(),
            config=AdamWConfig(lr=self.cfg.lr, weight_decay=self.cfg.weight_decay),
        )

    @staticmethod
    def _prepare_shifted(
        labels: cp.ndarray,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Shift labels for next-token prediction, build response mask.

        Args:
            labels: Target ids, shape (B, T). -100 for prompt/padding.

        Returns:
            shift_labels: Shifted labels (B, T-1). Clipped to [0, V).
            mask: Response mask (B, T-1). 1.0 where label != -100.
        """
        shift_labels = labels[:, 1:]  # (B, T-1)
        mask = (shift_labels != -100).astype(cp.float32)
        safe_labels = cp.clip(shift_labels, 0, None).astype(cp.int32)
        return safe_labels, mask

    def train_step(
        self, input_ids: cp.ndarray, labels: cp.ndarray
    ) -> dict[str, float]:
        """One full train step: forward → loss → backward → update.

        Args:
            input_ids: Token ids, shape (B, T).
            labels: Labels (-100 for prompt/padding), shape (B, T).

        Returns:
            Metrics dict.
        """
        B, T = input_ids.shape

        # Forward: model → logits
        logits = self.model.forward(input_ids)  # (B, T, V)
        V = logits.shape[2]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].reshape(-1, V)  # (B*(T-1), V)
        safe_labels, mask = self._prepare_shifted(labels)
        flat_labels = safe_labels.reshape(-1).astype(cp.int32)  # (B*(T-1),)
        N = shift_logits.shape[0]  # B*(T-1)

        # Fused log-softmax + gather → per-token log-probs
        logprobs_flat = log_softmax_gather_fwd(shift_logits, flat_labels)

        # Apply mask and reshape to (B, T-1)
        logprobs = logprobs_flat.reshape(B, T - 1) * mask
        T1 = T - 1

        # Masked mean NLL → per-sequence losses
        losses = sft_masked_nll_fwd(logprobs, mask)
        loss_val = float(losses.mean())

        # ---- Backward ----
        # d(mean_loss) / d(loss_b) = 1/B
        grad_out = cp.full(B, 1.0 / B, dtype=cp.float32)
        count = mask.sum(axis=1).astype(cp.float32)  # (B,)

        # Backward through masked NLL → grad w.r.t. logprobs
        grad_logprobs = sft_masked_nll_bwd(mask, count, grad_out)

        # Flatten for log-softmax backward
        grad_lp_flat = (grad_logprobs * mask).reshape(-1)  # (N,)

        # Backward through log-softmax + gather → grad w.r.t. logits
        grad_shift_logits = log_softmax_gather_backward(
            shift_logits, flat_labels, grad_lp_flat)

        # Reconstruct full logit gradient (B, T, V) — first position is zero
        grad_logits = cp.zeros_like(logits)
        grad_logits[:, :-1, :] = grad_shift_logits.reshape(B, T1, V)

        # Backward through model → parameter gradients
        grads = self.model.backward(grad_logits)

        # Clip and update
        clip_grad_norm(grads, self.cfg.grad_clip)
        self.optimizer.step(grads)

        return {"loss": loss_val}

    def train(self, data_iter) -> list[dict[str, float]]:
        """Full training loop.

        ``data_iter`` yields dicts with keys ``input_ids``, ``labels``
        (both ``cp.ndarray``).
        """
        history: list[dict[str, float]] = []
        for step, batch in enumerate(data_iter):
            if step >= self.cfg.max_steps:
                break
            metrics = self.train_step(batch["input_ids"], batch["labels"])
            history.append(metrics)
            if step % self.cfg.log_every == 0:
                print(f"[SFT] step {step:>5d}  loss={metrics['loss']:.4f}")
        return history


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _backend_label = "cuTile" if _USE_CUTILE_KERNELS else "CuPy/NumPy fallback"
    print(f"=== SFT Smoke Test (backend: {_backend_label}) ===\n")

    xp = cp
    np.random.seed(42)

    # --- Test 1: log-softmax + gather (via wrapper) ---
    print("--- log-softmax + gather ---")
    N, V = 8, 256
    logits_np = np.random.randn(N, V).astype(np.float32)
    labels_np = np.random.randint(0, V, (N,)).astype(np.int32)

    # NumPy reference
    mx = logits_np.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(logits_np - mx).sum(axis=-1, keepdims=True))
    ref_lp = (logits_np - mx - lse)[np.arange(N), labels_np]

    logits_arr = xp.array(logits_np)
    labels_arr = xp.array(labels_np)
    result_arr = log_softmax_gather_fwd(logits_arr, labels_arr)
    result = np.asarray(result_arr) if hasattr(result_arr, 'get') else np.array(result_arr)
    if np.allclose(result, ref_lp, atol=1e-4):
        print("-> Forward matches NumPy reference.\n")
    else:
        print(f"FAIL Max diff: {np.abs(result - ref_lp).max():.6f}\n")

    # --- Test 2: log-softmax + gather backward (via wrapper) ---
    print("--- log-softmax + gather backward ---")
    grad_out_np = np.random.randn(N).astype(np.float32)
    # NumPy reference: (one_hot - softmax) * grad_out
    sm = np.exp(logits_np - mx) / np.exp(logits_np - mx).sum(axis=-1, keepdims=True)
    oh = np.zeros_like(logits_np)
    oh[np.arange(N), labels_np] = 1.0
    ref_grad = (oh - sm) * grad_out_np[:, None]

    grad_out_arr = xp.array(grad_out_np)
    grad_logits_arr = log_softmax_gather_backward(logits_arr, labels_arr, grad_out_arr)
    grad_result = np.asarray(grad_logits_arr) if hasattr(grad_logits_arr, 'get') else np.array(grad_logits_arr)
    if np.allclose(grad_result, ref_grad, atol=1e-4):
        print("-> Backward matches NumPy reference.\n")
    else:
        print(f"FAIL Max diff: {np.abs(grad_result - ref_grad).max():.6f}\n")

    # --- Test 3: Masked NLL (via wrapper) ---
    print("--- Masked NLL ---")
    B, T = 4, 32
    logprobs_np = np.random.randn(B, T).astype(np.float32) * 0.5
    mask_np = np.ones((B, T), dtype=np.float32)
    for i in range(B):
        plen = np.random.randint(T // 4, T // 2)
        mask_np[i, :plen] = 0.0
        logprobs_np[i, :plen] = 0.0

    ref_nll = (-logprobs_np * mask_np).sum(axis=1) / np.maximum(mask_np.sum(axis=1), 1e-8)

    lp_arr = xp.array(logprobs_np)
    mask_arr = xp.array(mask_np)
    losses_arr = sft_masked_nll_fwd(lp_arr, mask_arr)
    result = np.asarray(losses_arr) if hasattr(losses_arr, 'get') else np.array(losses_arr)
    if np.allclose(result, ref_nll, atol=1e-4):
        print("-> Masked NLL matches.\n")
    else:
        print(f"FAIL Max diff: {np.abs(result - ref_nll).max():.6f}\n")

    # --- Test 4: NLL backward (via wrapper) ---
    print("--- NLL backward ---")
    count_np = mask_np.sum(axis=1).astype(np.float32)
    grad_out_np2 = (np.ones(B) / B).astype(np.float32)
    ref_grad_lp = -mask_np / np.maximum(count_np[:, None], 1e-8) * grad_out_np2[:, None]

    count_arr = xp.array(count_np)
    grad_out_arr2 = xp.array(grad_out_np2)
    grad_lp_arr = sft_masked_nll_bwd(mask_arr, count_arr, grad_out_arr2)
    grad_result = np.asarray(grad_lp_arr) if hasattr(grad_lp_arr, 'get') else np.array(grad_lp_arr)
    if np.allclose(grad_result, ref_grad_lp, atol=1e-5):
        print("-> NLL backward matches analytical reference.\n")
    else:
        print(f"FAIL Max diff: {np.abs(grad_result - ref_grad_lp).max():.6f}\n")

    print("=== SFT Smoke Test Complete ===")
