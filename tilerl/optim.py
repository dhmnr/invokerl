"""optim -- AdamW optimizer implemented as cuTile kernels.

Decoupled weight decay (Loshchilov & Hutter, 2019):
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta_{t-1})

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

_TILE = 1024  # power-of-2 tile for flat parameter processing

# ---------------------------------------------------------------------------
# cuTile kernels
# ---------------------------------------------------------------------------


@ct.kernel
def adamw_step_kernel(
    param: ct.Array,
    grad: ct.Array,
    m: ct.Array,
    v: ct.Array,
    lr: ct.Array,
    beta1: ct.Array,
    beta2: ct.Array,
    eps: ct.Array,
    wd: ct.Array,
    bc1: ct.Array,
    bc2: ct.Array,
    TILE: ConstInt,
):
    """In-place AdamW update for one tile of a flattened parameter.

    Each block processes TILE contiguous elements. Padding handles
    the final partial tile (reads zero, writes are ignored for OOB).

    Args:
        param: Flattened parameter, shape (N,). Updated in-place.
        grad: Flattened gradient, shape (N,).
        m: First moment buffer, shape (N,). Updated in-place.
        v: Second moment buffer, shape (N,). Updated in-place.
        lr: Learning rate scalar, shape (1,).
        beta1: First moment decay scalar, shape (1,).
        beta2: Second moment decay scalar, shape (1,).
        eps: Denominator epsilon scalar, shape (1,).
        wd: Weight decay scalar, shape (1,).
        bc1: Bias correction 1 = (1 - beta1^t), shape (1,).
        bc2: Bias correction 2 = (1 - beta2^t), shape (1,).
        TILE: Tile width (power-of-2).
    """
    pid = ct.bid(0)

    p = ct.load(param, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    g = ct.load(grad, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    m_old = ct.load(m, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    v_old = ct.load(v, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)

    lr_s = ct.load(lr, index=(0,), shape=(1,))
    b1 = ct.load(beta1, index=(0,), shape=(1,))
    b2 = ct.load(beta2, index=(0,), shape=(1,))
    e = ct.load(eps, index=(0,), shape=(1,))
    w = ct.load(wd, index=(0,), shape=(1,))
    c1 = ct.load(bc1, index=(0,), shape=(1,))
    c2 = ct.load(bc2, index=(0,), shape=(1,))

    # Moment updates
    m_new = b1 * m_old + (1.0 - b1) * g
    v_new = b2 * v_old + (1.0 - b2) * g * g

    # Bias-corrected estimates
    m_hat = m_new / c1
    v_hat = v_new / c2

    # Decoupled weight decay + Adam update
    p_new = p - lr_s * (m_hat / (ct.sqrt(v_hat) + e) + w * p)

    ct.store(param, index=(pid,), tile=p_new)
    ct.store(m, index=(pid,), tile=m_new)
    ct.store(v, index=(pid,), tile=v_new)


# ---------------------------------------------------------------------------
# Optimizer class
# ---------------------------------------------------------------------------


@dataclass
class AdamWConfig:
    """Hyperparameters for AdamW."""

    lr: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01


class AdamW:
    """AdamW optimizer with cuTile kernel for parameter updates.

    All parameters and state live as CuPy arrays on GPU. The update kernel
    runs one block per TILE-sized chunk of each flattened parameter.
    """

    def __init__(
        self,
        params: dict[str, cp.ndarray],
        config: AdamWConfig | None = None,
    ) -> None:
        self.cfg = config or AdamWConfig()
        self.params = params
        self.step_count = 0

        # First / second moment buffers (zeros, matching each param)
        self.m: dict[str, cp.ndarray] = {
            name: cp.zeros(p.size, dtype=cp.float32) for name, p in params.items()
        }
        self.v: dict[str, cp.ndarray] = {
            name: cp.zeros(p.size, dtype=cp.float32) for name, p in params.items()
        }

        # Pre-allocate scalar hyperparameter arrays (GPU if cuTile, else CPU)
        xp = cp
        self._lr = xp.array([self.cfg.lr], dtype=xp.float32)
        self._beta1 = xp.array([self.cfg.beta1], dtype=xp.float32)
        self._beta2 = xp.array([self.cfg.beta2], dtype=xp.float32)
        self._eps = xp.array([self.cfg.eps], dtype=xp.float32)
        self._wd = xp.array([self.cfg.weight_decay], dtype=xp.float32)

    def set_lr(self, lr: float) -> None:
        """Update learning rate (e.g., for scheduling)."""
        self.cfg.lr = lr
        self._lr = cp.array([lr], dtype=cp.float32)  # works for both CuPy and NumPy

    def step(self, grads: dict[str, cp.ndarray]) -> None:
        """Perform one AdamW update for all parameters with provided gradients.

        Args:
            grads: Gradient dict ``{param_name: gradient_array}``.
                   Names must match those passed to ``__init__``.
        """
        self.step_count += 1
        t = self.step_count

        xp = cp  # CuPy or NumPy depending on backend
        beta1, beta2 = self.cfg.beta1, self.cfg.beta2
        bc1_val = 1.0 - beta1 ** t
        bc2_val = 1.0 - beta2 ** t
        lr_val = self.cfg.lr
        eps_val = self.cfg.eps
        wd_val = self.cfg.weight_decay

        if _USE_CUTILE_KERNELS:
            # cuTile kernel path
            bc1 = xp.array([bc1_val], dtype=xp.float32)
            bc2 = xp.array([bc2_val], dtype=xp.float32)
            stream = xp.cuda.get_current_stream()

            for name, param in self.params.items():
                if name not in grads:
                    continue

                grad = grads[name]
                numel = param.size
                p_flat = param.reshape(-1).astype(xp.float32, copy=False)
                g_flat = grad.reshape(-1).astype(xp.float32, copy=False)
                m_flat = self.m[name]
                v_flat = self.v[name]
                num_blocks = (numel + _TILE - 1) // _TILE

                ct.launch(
                    stream, (num_blocks,), adamw_step_kernel,
                    (p_flat, g_flat, m_flat, v_flat,
                     self._lr, self._beta1, self._beta2, self._eps, self._wd,
                     bc1, bc2, _TILE),
                )
        else:
            # CuPy/NumPy fallback — pure array ops
            for name, param in self.params.items():
                if name not in grads:
                    continue

                grad = grads[name]
                p_flat = param.reshape(-1).astype(xp.float32)
                g_flat = grad.reshape(-1).astype(xp.float32)
                m_flat = self.m[name]
                v_flat = self.v[name]

                # Moment updates
                m_flat[:] = beta1 * m_flat + (1.0 - beta1) * g_flat
                v_flat[:] = beta2 * v_flat + (1.0 - beta2) * g_flat * g_flat

                # Bias-corrected estimates
                m_hat = m_flat / bc1_val
                v_hat = v_flat / bc2_val

                # Decoupled weight decay + Adam update
                p_flat[:] = p_flat - lr_val * (m_hat / (xp.sqrt(v_hat) + eps_val) + wd_val * p_flat)

                # Write back to original param shape
                self.params[name] = p_flat.reshape(param.shape)

    def zero_grad(self) -> None:
        """No-op: gradients are passed explicitly to ``step()``, not accumulated."""
        pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def clip_grad_norm(
    grads: dict[str, cp.ndarray], max_norm: float
) -> float:
    """Clip gradient global norm in-place.

    Computes the L2 norm across all gradient arrays. If it exceeds
    ``max_norm``, scales all gradients down proportionally.

    Args:
        grads: Gradient dict (modified in-place if clipped).
        max_norm: Maximum allowed L2 norm.

    Returns:
        The original (unclipped) global norm.
    """
    total_norm_sq = 0.0
    for g in grads.values():
        total_norm_sq += float(cp.sum(g * g))
    total_norm = total_norm_sq ** 0.5

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for name in grads:
            grads[name] = grads[name] * scale
    return total_norm


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"=== AdamW Optimizer Smoke Test (backend: {'cuTile' if _USE_CUTILE_KERNELS else 'CuPy/NumPy fallback'}) ===\n")

    has_cuda = hasattr(cp, 'cuda') and cp.cuda.runtime.getDeviceCount() > 0
    if not has_cuda:
        print("No CUDA device — running NumPy-only reference test.\n")

    np.random.seed(42)

    # --- Config ---
    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
    N = 2048  # parameter size
    steps = 5

    # --- NumPy reference ---
    param_np = np.random.randn(N).astype(np.float32)
    grads_np = [np.random.randn(N).astype(np.float32) * 0.1 for _ in range(steps)]
    m_np = np.zeros(N, dtype=np.float32)
    v_np = np.zeros(N, dtype=np.float32)

    param_ref = param_np.copy()
    m_ref = m_np.copy()
    v_ref = v_np.copy()

    for t in range(1, steps + 1):
        g = grads_np[t - 1]
        m_ref = beta1 * m_ref + (1 - beta1) * g
        v_ref = beta2 * v_ref + (1 - beta2) * g ** 2
        m_hat = m_ref / (1 - beta1 ** t)
        v_hat = v_ref / (1 - beta2 ** t)
        param_ref = param_ref - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * param_ref)

    print(f"NumPy reference param (first 5): {param_ref[:5]}")
    print(f"NumPy reference m     (first 5): {m_ref[:5]}")

    # --- Optimizer test (uses cuTile or fallback depending on backend) ---
    xp = cp
    param_cu = xp.array(param_np.copy())
    config = AdamWConfig(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=wd)
    optimizer = AdamW({"test": param_cu}, config=config)

    for t in range(steps):
        g_cu = xp.array(grads_np[t])
        optimizer.step({"test": g_cu})

    param_result = np.asarray(optimizer.params["test"].reshape(-1))
    m_result = np.asarray(optimizer.m["test"])

    backend_name = "cuTile" if _USE_CUTILE_KERNELS else "fallback"
    print(f"{backend_name} param  (first 5): {param_result[:5]}")
    print(f"{backend_name} m      (first 5): {m_result[:5]}")

    if np.allclose(param_result, param_ref, atol=1e-5, rtol=1e-4):
        print(f"\n-> AdamW parameter update matches NumPy reference ({backend_name}).")
    else:
        diff = np.abs(param_result - param_ref).max()
        print(f"\n  Max param difference: {diff:.8f}")

    if np.allclose(m_result, m_ref, atol=1e-5, rtol=1e-4):
        print(f"-> First moment matches NumPy reference ({backend_name}).")
    else:
        diff = np.abs(m_result - m_ref).max()
        print(f"  Max moment difference: {diff:.8f}")

    # --- Gradient clipping test ---
    print("\n--- Gradient clipping ---")
    g1 = xp.ones(100, dtype=xp.float32) * 10.0  # norm = 100
    g2 = xp.ones(100, dtype=xp.float32) * 10.0  # norm = 100
    grads_dict = {"a": g1, "b": g2}
    # Total norm = sqrt(100*100 + 100*100) = sqrt(20000) ~ 141.4
    orig_norm = clip_grad_norm(grads_dict, max_norm=1.0)
    clipped_norm_sq = sum(float(xp.sum(g * g)) for g in grads_dict.values())
    clipped_norm = clipped_norm_sq ** 0.5
    print(f"Original norm: {orig_norm:.2f}")
    print(f"Clipped norm:  {clipped_norm:.4f}")
    if clipped_norm <= 1.0 + 1e-4:
        print("-> Gradient clipping works correctly.")
    else:
        print("  Clipping failed.")

    print("\n=== AdamW Smoke Test Complete ===")
