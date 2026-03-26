"""Qwen3 Transformer — forward and backward passes in cuTile kernels.

Architecture: RMSNorm, RoPE (θ=1e6), GQA + QK-Norm, SwiGLU FFN, tied embeddings.
Configs: 0.6B, 1.7B, 4B (from HuggingFace Qwen3 family).

Pure cuTile + CuPy. No PyTorch.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

# Backend selection via TILERL_BACKEND env var:
#   "numpy"  — Force NumPy CPU ops (always correct, slow). Use on sm_120/Blackwell.
#   "cupy"   — CuPy GPU ops (cuBLAS + JIT kernels). Use on A100/H100.
#   "cutile" — cuTile kernels (fastest, but has OOB write bugs on non-pow2 dims).
#   ""       — Auto: CuPy if available, else NumPy. cuTile kernels disabled by default.
_BACKEND = os.environ.get("TILERL_BACKEND", "").lower()

if _BACKEND == "numpy":
    cp = np  # type: ignore[assignment] — forced CPU mode
else:
    try:
        import cupy as cp
    except ModuleNotFoundError:
        cp = np  # type: ignore[assignment] — CPU fallback

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

ConstInt = ct.Constant[int] if _HAS_CUTILE else int

# cuTile kernels are only used when explicitly requested (TILERL_BACKEND=cutile).
# Known OOB write bugs on non-power-of-2 dimensions.
_USE_CUTILE_KERNELS = _HAS_CUTILE and _BACKEND == "cutile"

# ── Configuration ───────────────────────────────────────────────────────────


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    intermediate_size: int = 3072
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    tie_word_embeddings: bool = True
    max_seq_len: int = 4096

    @classmethod
    def qwen3_0_6b(cls):
        return cls(hidden_size=1024, num_hidden_layers=28, num_attention_heads=16,
                   num_key_value_heads=8, intermediate_size=3072)

    @classmethod
    def qwen3_1_7b(cls):
        return cls(hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
                   num_key_value_heads=8, intermediate_size=6144)

    @classmethod
    def qwen3_4b(cls):
        return cls(hidden_size=2560, num_hidden_layers=36, num_attention_heads=32,
                   num_key_value_heads=8, intermediate_size=9728)

    @classmethod
    def tiny(cls):
        """Tiny config for smoke tests."""
        return cls(vocab_size=256, hidden_size=64, num_hidden_layers=2,
                   num_attention_heads=4, num_key_value_heads=2,
                   intermediate_size=128, head_dim=32, max_seq_len=64)


# ── cuTile Kernels ──────────────────────────────────────────────────────────


@ct.kernel
def rms_norm_fwd_kernel(X, W, Y, Rstd, EpsK: ConstInt, N: ConstInt, TILE_N: ConstInt):
    """RMSNorm forward: y = x * rsqrt(mean(x²)+eps) * w.  One block per row."""
    bid = ct.bid(0)
    nt = ct.num_tiles(X, axis=1, shape=(1, TILE_N))
    # Compute mean(x²)
    ss = ct.full((1, 1), 0.0, dtype=ct.float32)
    for j in range(nt):
        x = ct.load(X, index=(bid, j), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        ss += ct.sum(x * x, axis=1, keepdims=True)
    ms = ss / N.astype(ct.float32)
    rs = 1.0 / ct.sqrt(ms + EpsK.astype(ct.float32) / 1e9)
    ct.store(Rstd, index=(bid,), tile=rs.reshape((1,)))
    # Normalize and scale
    for j in range(nt):
        x = ct.load(X, index=(bid, j), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        w = ct.load(W, index=(j,), shape=(TILE_N,), padding_mode=ct.PaddingMode.ZERO)
        ct.store(Y, index=(bid, j), tile=(x * rs * w).astype(Y.dtype))


@ct.kernel
def rms_norm_bwd_kernel(DY, X, W, Rstd, DX, DW,
                        N: ConstInt, TILE_N: ConstInt):
    """RMSNorm backward: computes dX, accumulates dW.  One block per row.

    dx = rstd * (w * dy - x * rstd² * mean(w * x * dy))
    dw += rstd * x * dy  (accumulated across rows)
    """
    bid = ct.bid(0)
    nt = ct.num_tiles(X, axis=1, shape=(1, TILE_N))
    rs = ct.load(Rstd, index=(bid,), shape=(1,)).reshape((1, 1))
    # Compute c = mean(w * x * dy)
    c = ct.full((1, 1), 0.0, dtype=ct.float32)
    for j in range(nt):
        dy = ct.load(DY, index=(bid, j), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        x = ct.load(X, index=(bid, j), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        w = ct.load(W, index=(j,), shape=(TILE_N,), padding_mode=ct.PaddingMode.ZERO)
        c += ct.sum(w * x * dy, axis=1, keepdims=True)
    c = c / N.astype(ct.float32)
    # Compute dx and accumulate dw
    for j in range(nt):
        dy = ct.load(DY, index=(bid, j), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        x = ct.load(X, index=(bid, j), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        w = ct.load(W, index=(j,), shape=(TILE_N,), padding_mode=ct.PaddingMode.ZERO)
        dx = rs * (w * dy - x * rs * rs * c)
        ct.store(DX, index=(bid, j), tile=dx.astype(DX.dtype))
        # Accumulate dW (atomic add across rows — simplified: store partial)
        dw_partial = rs * x * dy
        old = ct.load(DW, index=(j,), shape=(TILE_N,), padding_mode=ct.PaddingMode.ZERO)
        ct.store(DW, index=(j,), tile=(old + dw_partial.reshape((TILE_N,))).astype(DW.dtype))


@ct.kernel
def rope_kernel(X, Cos, Sin, Out, D: ConstInt, TILE_D: ConstInt):
    """Apply RoPE: rotate pairs of dimensions.  One block per (batch, head, seq) position.

    For each pair (x0, x1): out0 = x0*cos - x1*sin, out1 = x0*sin + x1*cos
    X shape: [N, D], Cos/Sin shape: [N, D//2], Out shape: [N, D]
    """
    bid = ct.bid(0)
    half = D // 2
    nt = ct.num_tiles(X, axis=1, shape=(1, TILE_D))
    half_nt = nt // 2
    for j in range(half_nt):
        x0 = ct.load(X, index=(bid, j), shape=(1, TILE_D), padding_mode=ct.PaddingMode.ZERO)
        x1 = ct.load(X, index=(bid, j + half_nt), shape=(1, TILE_D),
                      padding_mode=ct.PaddingMode.ZERO)
        cs = ct.load(Cos, index=(bid, j), shape=(1, TILE_D), padding_mode=ct.PaddingMode.ZERO)
        sn = ct.load(Sin, index=(bid, j), shape=(1, TILE_D), padding_mode=ct.PaddingMode.ZERO)
        ct.store(Out, index=(bid, j), tile=(x0 * cs - x1 * sn).astype(Out.dtype))
        ct.store(Out, index=(bid, j + half_nt), tile=(x0 * sn + x1 * cs).astype(Out.dtype))


@ct.kernel
def matmul_kernel(A, B, C, TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt):
    """Tiled GEMM: C[M,N] = A[M,K] @ B[K,N].  Grid = (ceil(M/TM), ceil(N/TN))."""
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    nk = ct.num_tiles(A, axis=1, shape=(TILE_M, TILE_K))
    acc = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
    for k in range(nk):
        a = ct.load(A, index=(bid_m, k), shape=(TILE_M, TILE_K),
                    padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, bid_n), shape=(TILE_K, TILE_N),
                    padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, b, acc)
    ct.store(C, index=(bid_m, bid_n), tile=acc.astype(C.dtype))


@ct.kernel
def attention_fwd_kernel(Q, K, V, Out, Scale,
                         TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
    """Causal fused attention: softmax(causal_mask(Q@K^T * scale)) @ V.

    One block per (batch*heads, query-tile).  Flash-style tiled accumulation.
    Q,K,V: [N, T, D], Out: [N, T, D] where N = batch * num_heads.
    Applies causal mask: position i can only attend to positions j <= i.
    """
    bid_n = ct.bid(0)  # batch*head index
    bid_m = ct.bid(1)  # query tile index
    T = K.shape[1]
    nt = ct.num_tiles(K, axis=1, shape=(1, TILE_N, TILE_D))
    # Online softmax accumulators
    m_i = ct.full((TILE_M, 1), float("-inf"), dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)
    scale = Scale.astype(ct.float32) / 1000.0
    # Row/col indices for causal mask
    row_idx = ct.arange(TILE_M, dtype=ct.int32).reshape((TILE_M, 1)) + bid_m * TILE_M
    for j in range(nt):
        q = ct.load(Q, index=(bid_n, bid_m, 0), shape=(1, TILE_M, TILE_D),
                    padding_mode=ct.PaddingMode.ZERO).reshape((TILE_M, TILE_D))
        k = ct.load(K, index=(bid_n, j, 0), shape=(1, TILE_N, TILE_D),
                    padding_mode=ct.PaddingMode.ZERO).reshape((TILE_N, TILE_D))
        kt = ct.transpose(k)  # [N, D] -> [D, N] for Q @ K^T
        qk = ct.mma(q, kt, ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32))
        qk = qk * scale
        # Causal + padding mask: -inf where col > row OR col >= T
        col_idx = ct.arange(TILE_N, dtype=ct.int32).reshape((1, TILE_N)) + j * TILE_N
        causal = ct.where(col_idx > row_idx, float("-inf"), 0.0)
        pad_mask = ct.where(col_idx >= T, float("-inf"), 0.0)
        qk = qk + causal + pad_mask
        # Online softmax update
        m_ij = ct.max(qk, axis=1, keepdims=True)
        m_new = ct.maximum(m_i, m_ij)
        alpha = ct.exp(m_i - m_new)
        p = ct.exp(qk - m_new)
        l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=True)
        acc = acc * alpha
        v = ct.load(V, index=(bid_n, j, 0), shape=(1, TILE_N, TILE_D),
                    padding_mode=ct.PaddingMode.ZERO).reshape((TILE_N, TILE_D))
        acc = ct.mma(p.astype(V.dtype), v, acc)
        m_i = m_new
    out = acc / l_i
    # Zero out padded rows (row >= T) to avoid writing garbage
    row_valid = ct.where(row_idx < T, 1.0, 0.0)  # [TILE_M, 1]
    out = out * row_valid
    ct.store(Out, index=(bid_n, bid_m, 0), tile=out.reshape((1, TILE_M, TILE_D)).astype(Out.dtype))


@ct.kernel
def silu_mul_fwd_kernel(Gate, Up, Out, TILE: ConstInt):
    """SwiGLU forward: out = silu(gate) * up = gate * sigmoid(gate) * up."""
    bid = ct.bid(0)
    nt = ct.num_tiles(Gate, axis=1, shape=(1, TILE))
    for j in range(nt):
        g = ct.load(Gate, index=(bid, j), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
        u = ct.load(Up, index=(bid, j), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
        sig = 1.0 / (1.0 + ct.exp(-g))
        ct.store(Out, index=(bid, j), tile=(g * sig * u).astype(Out.dtype))


@ct.kernel
def silu_mul_bwd_kernel(DOut, Gate, Up, DGate, DUp, TILE: ConstInt):
    """SwiGLU backward.  d_gate = d_out * up * (sig + gate*sig*(1-sig)), d_up = d_out * silu(gate)."""
    bid = ct.bid(0)
    nt = ct.num_tiles(Gate, axis=1, shape=(1, TILE))
    for j in range(nt):
        do = ct.load(DOut, index=(bid, j), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
        g = ct.load(Gate, index=(bid, j), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
        u = ct.load(Up, index=(bid, j), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
        sig = 1.0 / (1.0 + ct.exp(-g))
        silu_g = g * sig
        dg = do * u * (sig + g * sig * (1.0 - sig))
        du = do * silu_g
        ct.store(DGate, index=(bid, j), tile=dg.astype(DGate.dtype))
        ct.store(DUp, index=(bid, j), tile=du.astype(DUp.dtype))


@ct.kernel
def embed_fwd_kernel(Ids, Table, Out, TILE_D: ConstInt):
    """Embedding lookup: out[i] = table[ids[i]].  One block per token."""
    bid = ct.bid(0)
    idx = ct.load(Ids, index=(bid,), shape=(1,))
    row = ct.gather(Table, (idx, ct.arange(TILE_D, dtype=ct.int32)))
    ct.store(Out, index=(bid,), shape=(TILE_D,), tile=row.astype(Out.dtype))


@ct.kernel
def embed_bwd_kernel(Ids, DOut, DTable, TILE_D: ConstInt):
    """Embedding backward: scatter dOut into dTable at ids positions."""
    bid = ct.bid(0)
    idx = ct.load(Ids, index=(bid,), shape=(1,))
    grad = ct.load(DOut, index=(bid,), shape=(TILE_D,), padding_mode=ct.PaddingMode.ZERO)
    ct.scatter(DTable, (idx, ct.arange(TILE_D, dtype=ct.int32)), grad)


# ── NumPy/CuPy reference ops (used when cuTile unavailable) ────────────────


def _np_rms_norm(x, w, eps):
    rstd = 1.0 / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x * rstd * w, rstd

def _np_rope(x, cos, sin):
    d = x.shape[-1]
    x0, x1 = x[..., :d // 2], x[..., d // 2:]
    return np.concatenate([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)

def _np_attention(q, k, v, scale):
    T = q.shape[-2]
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale
    # Causal mask: position i can only attend to j <= i
    causal = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
    scores = scores + causal
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.matmul(weights, v), weights

def _np_silu_mul(gate, up):
    sig = 1.0 / (1.0 + np.exp(-gate.clip(-88, 88)))
    return gate * sig * up


# ── Layer-level operations ──────────────────────────────────────────────────

def _tile_pow2(n, cap=1024):
    """Next power-of-2 ≤ cap."""
    return min(cap, 1 << max(0, (n - 1)).bit_length()) if n > 0 else 1


def _pad_to(n, tile):
    """Round n up to next multiple of tile.  Returns padded size."""
    return ((n + tile - 1) // tile) * tile


def matmul(a, b):
    """General matrix multiply: C = A @ B using cuTile, CuPy cuBLAS, or NumPy.

    cuTile path pads output to tile-aligned dims to prevent OOB writes.
    """
    if _USE_CUTILE_KERNELS:
        M, K = a.shape[-2], a.shape[-1]
        N = b.shape[-1]
        a2 = a.reshape(-1, K)
        Mf = a2.shape[0]
        TM = _tile_pow2(min(Mf, 128))
        TN = _tile_pow2(min(N, 128))
        TK = _tile_pow2(min(K, 128))
        # Pad output to tile-aligned dims to prevent OOB writes
        Mpad = _pad_to(Mf, TM)
        Npad = _pad_to(N, TN)
        c = cp.zeros((Mpad, Npad), dtype=cp.float32)
        grid = (Mpad // TM, Npad // TN)
        ct.launch(cp.cuda.get_current_stream(), grid, matmul_kernel,
                  (a2, b, c, TM, TN, TK))
        return c[:Mf, :N].reshape(*a.shape[:-1], N)
    # CuPy (GPU cuBLAS) or NumPy (CPU) fallback
    xp = cp
    return xp.matmul(a.astype(xp.float32), b.astype(xp.float32))


def rms_norm(x, w, eps, return_rstd=False):
    """RMSNorm forward. Returns (y, rstd) if return_rstd else y.

    cuTile path pads row dim to tile size to prevent OOB writes.
    """
    xp = cp
    N = x.shape[-1]
    x2 = x.reshape(-1, N).astype(xp.float32)
    M = x2.shape[0]
    if _USE_CUTILE_KERNELS:
        TN = _tile_pow2(N)
        Npad = _pad_to(N, TN)
        # Pad x2 and w if N isn't tile-aligned
        if Npad != N:
            x2_pad = cp.zeros((M, Npad), dtype=cp.float32)
            x2_pad[:, :N] = x2
            w_pad = cp.zeros(Npad, dtype=w.dtype)
            w_pad[:N] = w
        else:
            x2_pad = x2
            w_pad = w
        y_pad = cp.empty((M, Npad), dtype=cp.float32)
        rstd = cp.empty(M, dtype=cp.float32)
        eps_k = int(eps * 1e9)
        ct.launch(cp.cuda.get_current_stream(), (M,), rms_norm_fwd_kernel,
                  (x2_pad, w_pad, y_pad, rstd, eps_k, N, TN))
        y = y_pad[:, :N].reshape(x.shape)
    else:
        # CuPy GPU or NumPy CPU path
        w_f = w.astype(xp.float32)
        rstd = 1.0 / xp.sqrt(xp.mean(x2 ** 2, axis=-1, keepdims=True) + eps)
        y = (x2 * rstd * w_f).reshape(x.shape)
        rstd = rstd.reshape(M)
    return (y, rstd) if return_rstd else y


def apply_rope(x, cos, sin):
    """Apply RoPE.  x: [..., D], cos/sin: [..., D//2].

    cuTile path pads column dim to tile size to prevent OOB writes.
    """
    if _USE_CUTILE_KERNELS:
        shape = x.shape
        D = shape[-1]
        x2 = x.reshape(-1, D)
        Nrows = x2.shape[0]
        TD = _tile_pow2(D // 2)
        Dpad = _pad_to(D, TD * 2)  # full D must accommodate 2 × TD tiles
        if Dpad != D:
            x2_pad = cp.zeros((Nrows, Dpad), dtype=cp.float32)
            x2_pad[:, :D] = x2
            cos_pad = cp.zeros((Nrows, Dpad // 2), dtype=cp.float32)
            cos_pad[:, :D // 2] = cos.reshape(-1, D // 2)
            sin_pad = cp.zeros((Nrows, Dpad // 2), dtype=cp.float32)
            sin_pad[:, :D // 2] = sin.reshape(-1, D // 2)
            out = cp.empty((Nrows, Dpad), dtype=cp.float32)
        else:
            x2_pad = x2
            cos_pad = cos.reshape(-1, D // 2)
            sin_pad = sin.reshape(-1, D // 2)
            out = cp.empty_like(x2)
        ct.launch(cp.cuda.get_current_stream(), (Nrows,), rope_kernel,
                  (x2_pad, cos_pad, sin_pad, out, Dpad if Dpad != D else D, TD))
        return out[:, :D].reshape(shape) if Dpad != D else out.reshape(shape)
    # CuPy GPU or NumPy CPU fallback
    xp = cp
    d = x.shape[-1]
    x0, x1 = x[..., :d // 2], x[..., d // 2:]
    return xp.concatenate([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1).astype(x.dtype)


def silu_mul(gate, up):
    """SwiGLU: silu(gate) * up.

    cuTile path pads column dim to tile size to prevent OOB writes.
    """
    if _USE_CUTILE_KERNELS:
        shape = gate.shape
        N = shape[-1]
        g2 = gate.reshape(-1, N)
        u2 = up.reshape(-1, N)
        M = g2.shape[0]
        T = _tile_pow2(N)
        Npad = _pad_to(N, T)
        if Npad != N:
            g2_pad = cp.zeros((M, Npad), dtype=cp.float32)
            g2_pad[:, :N] = g2
            u2_pad = cp.zeros((M, Npad), dtype=cp.float32)
            u2_pad[:, :N] = u2
            out = cp.empty((M, Npad), dtype=cp.float32)
        else:
            g2_pad = g2
            u2_pad = u2
            out = cp.empty_like(g2)
        ct.launch(cp.cuda.get_current_stream(), (M,), silu_mul_fwd_kernel,
                  (g2_pad, u2_pad, out, T))
        return out[:, :N].reshape(shape) if Npad != N else out.reshape(shape)
    # CuPy GPU or NumPy CPU fallback
    xp = cp
    g = gate.astype(xp.float32)
    g_clipped = xp.clip(g, -88, 88)
    sig = 1.0 / (1.0 + xp.exp(-g_clipped))
    return (g * sig * up.astype(xp.float32)).astype(gate.dtype)


def precompute_rope(seq_len, head_dim, theta=1e6):
    """Precompute cos/sin for RoPE.  Returns (cos, sin) of shape [seq_len, head_dim//2]."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
    t = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(t, freqs)  # [seq_len, half]
    return np.cos(angles), np.sin(angles)


# ── Qwen3 Model ────────────────────────────────────────────────────────────

def _zeros(shape, xp=None):
    xp = xp or cp
    return xp.zeros(shape, dtype=xp.float32)

def _randn(shape, scale=0.02, xp=None):
    xp = xp or cp
    return (xp.random.randn(*shape) * scale).astype(xp.float32)


class Qwen3Model:
    """Qwen3 transformer with cuTile forward and backward passes."""

    def __init__(self, config: Qwen3Config):
        self.cfg = config
        self._init_weights()
        self._cache = {}  # activation cache for backward

    def _init_weights(self):
        """Initialize all parameters as CuPy arrays."""
        c = self.cfg
        xp = cp
        s = 0.02  # init scale
        self.params: dict[str, np.ndarray] = {}
        # Embedding
        self.params["embed.weight"] = _randn((c.vocab_size, c.hidden_size), s, xp)
        # Per-layer weights
        Qd = c.num_attention_heads * c.head_dim
        Kd = c.num_key_value_heads * c.head_dim
        for i in range(c.num_hidden_layers):
            p = f"layers.{i}"
            # Attention norms
            self.params[f"{p}.attn_norm.weight"] = xp.ones(c.hidden_size, dtype=xp.float32)
            # QKV projections (no bias)
            self.params[f"{p}.attn.q_proj.weight"] = _randn((c.hidden_size, Qd), s, xp)
            self.params[f"{p}.attn.k_proj.weight"] = _randn((c.hidden_size, Kd), s, xp)
            self.params[f"{p}.attn.v_proj.weight"] = _randn((c.hidden_size, Kd), s, xp)
            # QK-norm weights (per-head)
            self.params[f"{p}.attn.q_norm.weight"] = xp.ones(c.head_dim, dtype=xp.float32)
            self.params[f"{p}.attn.k_norm.weight"] = xp.ones(c.head_dim, dtype=xp.float32)
            # Output projection
            self.params[f"{p}.attn.o_proj.weight"] = _randn((Qd, c.hidden_size), s, xp)
            # FFN norm
            self.params[f"{p}.ffn_norm.weight"] = xp.ones(c.hidden_size, dtype=xp.float32)
            # SwiGLU FFN (no bias)
            self.params[f"{p}.ffn.gate_proj.weight"] = _randn(
                (c.hidden_size, c.intermediate_size), s, xp)
            self.params[f"{p}.ffn.up_proj.weight"] = _randn(
                (c.hidden_size, c.intermediate_size), s, xp)
            self.params[f"{p}.ffn.down_proj.weight"] = _randn(
                (c.intermediate_size, c.hidden_size), s, xp)
        # Final norm
        self.params["final_norm.weight"] = xp.ones(c.hidden_size, dtype=xp.float32)
        # LM head (tied with embedding if configured)
        if not c.tie_word_embeddings:
            self.params["lm_head.weight"] = _randn((c.hidden_size, c.vocab_size), s, xp)

    def _get_lm_head_weight(self):
        if self.cfg.tie_word_embeddings:
            return self.params["embed.weight"].T  # [H, V]
        return self.params["lm_head.weight"]

    def forward(self, token_ids, recompute_attn=False, no_cache=False):
        """Forward pass.

        Args:
            token_ids: [B, T] int array of token IDs.
            recompute_attn: If True, skip caching attention weights to save
                memory (~4.5 GB for 28-layer model at T=512, N=8). The backward
                pass will recompute them from cached Q/K. Default False for
                backward compatibility.
            no_cache: If True, skip ALL activation caching. Use for reference
                models or any forward-only pass where backward() won't be called.
                Saves ~13 GB for 28-layer model at B=8, T=562. Cannot call
                backward() after a no_cache forward.

        Returns:
            logits: [B, T, V] float array.
        """
        xp = cp
        c = self.cfg
        B, T = token_ids.shape
        _c = not no_cache  # shorthand: cache activations?
        cache = {"token_ids": token_ids, "layers": []} if _c else None

        # CuPy memory pool cleanup: release freed blocks every N layers
        # to prevent pool fragmentation from consuming all GPU memory.
        _pool = (cp.get_default_memory_pool()
                 if hasattr(cp, 'get_default_memory_pool') else None)
        _pool_interval = 4  # free pool every 4 layers

        # Embedding lookup
        hidden = self.params["embed.weight"][token_ids]  # [B, T, H]
        if _c:
            cache["embed_out"] = hidden.copy()

        # Precompute RoPE
        rope_cos, rope_sin = precompute_rope(T, c.head_dim, c.rope_theta)
        if xp is not np:
            rope_cos, rope_sin = xp.asarray(rope_cos), xp.asarray(rope_sin)

        Nh, Nkv, D = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        groups = Nh // Nkv

        for i in range(c.num_hidden_layers):
            p = f"layers.{i}"
            lc = {} if _c else None

            # ── Attention block ──
            residual = hidden
            if _c:
                lc["attn_input"] = hidden.copy()
            hidden = rms_norm(hidden, self.params[f"{p}.attn_norm.weight"],
                              c.rms_norm_eps, return_rstd=_c)
            if _c:
                hidden, rstd = hidden
                lc["attn_norm_rstd"] = rstd
                lc["attn_norm_out"] = hidden.copy()

            # QKV projections: [B, T, H] @ [H, Qd/Kd] -> [B, T, Qd/Kd]
            q = matmul(hidden, self.params[f"{p}.attn.q_proj.weight"])
            k = matmul(hidden, self.params[f"{p}.attn.k_proj.weight"])
            v = matmul(hidden, self.params[f"{p}.attn.v_proj.weight"])

            # Reshape for multi-head: [B, T, Nh/Nkv, D]
            q = q.reshape(B, T, Nh, D)
            k = k.reshape(B, T, Nkv, D)
            v = v.reshape(B, T, Nkv, D)

            # QK-Norm (per-head RMSNorm)
            if _c:
                lc["q_prenorm"] = q.copy()
                lc["k_prenorm"] = k.copy()
            q_flat = rms_norm(q.reshape(-1, D),
                self.params[f"{p}.attn.q_norm.weight"],
                c.rms_norm_eps, return_rstd=_c)
            k_flat = rms_norm(k.reshape(-1, D),
                self.params[f"{p}.attn.k_norm.weight"],
                c.rms_norm_eps, return_rstd=_c)
            if _c:
                q_flat, q_rstd = q_flat
                k_flat, k_rstd = k_flat
                lc["q_norm_rstd"] = q_rstd
                lc["k_norm_rstd"] = k_rstd
            q = q_flat.reshape(q.shape)
            k = k_flat.reshape(k.shape)

            # RoPE
            # cos/sin: [T, D//2] -> broadcast to [B, T, 1, D//2]
            cos_t = rope_cos[:T].reshape(1, T, 1, D // 2)
            sin_t = rope_sin[:T].reshape(1, T, 1, D // 2)
            q = apply_rope(q, xp.broadcast_to(cos_t, (B, T, Nh, D // 2)),
                           xp.broadcast_to(sin_t, (B, T, Nh, D // 2)))
            k = apply_rope(k, xp.broadcast_to(cos_t, (B, T, Nkv, D // 2)),
                           xp.broadcast_to(sin_t, (B, T, Nkv, D // 2)))

            # Transpose for attention: [B, Nh/Nkv, T, D]
            q = xp.transpose(q, (0, 2, 1, 3))
            k = xp.transpose(k, (0, 2, 1, 3))
            v = xp.transpose(v, (0, 2, 1, 3))

            # Expand KV heads for GQA: [B, Nkv, T, D] -> [B, Nh, T, D]
            if groups > 1:
                k = xp.repeat(k, groups, axis=1)
                v = xp.repeat(v, groups, axis=1)

            # Attention — use CuPy matmul (cuBLAS) for correctness
            scale = np.float32(1.0 / math.sqrt(D))
            scores = xp.matmul(q, xp.swapaxes(k, -2, -1)) * scale
            causal = xp.asarray(np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1))
            scores = (scores + causal).astype(xp.float32)
            scores = scores - scores.max(axis=-1, keepdims=True)
            weights = xp.exp(scores).astype(xp.float32)
            weights = weights / weights.sum(axis=-1, keepdims=True)
            attn_out = xp.matmul(weights, v).astype(xp.float32)
            if _c:
                if not recompute_attn:
                    lc["attn_weights"] = weights
                lc["q"], lc["k"], lc["v"] = q, k, v
                lc["attn_out"] = attn_out

            # Transpose back: [B, Nh, T, D] -> [B, T, Nh*D]
            attn_out = xp.transpose(attn_out, (0, 2, 1, 3)).reshape(B, T, Nh * D)

            # Output projection + residual
            hidden = matmul(attn_out, self.params[f"{p}.attn.o_proj.weight"]) + residual
            if _c:
                lc["attn_proj_input"] = attn_out

            # ── FFN block ──
            residual = hidden
            if _c:
                lc["ffn_input"] = hidden.copy()
            hidden = rms_norm(hidden, self.params[f"{p}.ffn_norm.weight"],
                              c.rms_norm_eps, return_rstd=_c)
            if _c:
                hidden, rstd = hidden
                lc["ffn_norm_rstd"] = rstd
                lc["ffn_norm_out"] = hidden.copy()

            gate = matmul(hidden, self.params[f"{p}.ffn.gate_proj.weight"])
            up = matmul(hidden, self.params[f"{p}.ffn.up_proj.weight"])
            if _c:
                lc["gate"], lc["up"] = gate.copy(), up.copy()
            ffn_act = silu_mul(gate, up)
            if _c:
                lc["ffn_act"] = ffn_act
            hidden = matmul(ffn_act, self.params[f"{p}.ffn.down_proj.weight"]) + residual

            if _c:
                cache["layers"].append(lc)

            # Periodically release CuPy memory pool to prevent fragmentation
            if _pool and (i + 1) % _pool_interval == 0:
                _pool.free_all_blocks()

        # Final norm + LM head
        if _c:
            cache["final_input"] = hidden.copy()
        hidden = rms_norm(hidden, self.params["final_norm.weight"],
                          c.rms_norm_eps, return_rstd=_c)
        if _c:
            hidden, rstd = hidden
            cache["final_rstd"] = rstd
            cache["final_norm_out"] = hidden.copy()

        logits = matmul(hidden, self._get_lm_head_weight())  # [B, T, V]
        if _c:
            self._cache = cache
        return logits

    def backward(self, grad_logits):
        """Backward pass through the full model.

        Args:
            grad_logits: [B, T, V] gradient w.r.t. logits.

        Returns:
            Dictionary of parameter gradients: {param_name: grad_array}.
        """
        xp = cp
        c = self.cfg
        cache = self._cache
        grads: dict[str, np.ndarray] = {}
        B, T = cache["token_ids"].shape
        Nh, Nkv, D = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        groups = Nh // Nkv

        # LM head backward: dH = dLogits @ W_lm^T,  dW_lm = H^T @ dLogits
        lm_w = self._get_lm_head_weight()  # [H, V]
        dh = matmul(grad_logits, lm_w.T)  # [B, T, H]
        fn_out = cache["final_norm_out"]
        # dW_lm = fn_out^T @ dLogits (sum over B*T)
        dw_lm = matmul(fn_out.reshape(-1, c.hidden_size).T,
                        grad_logits.reshape(-1, c.vocab_size))  # [H, V]
        if c.tie_word_embeddings:
            grads["embed.weight"] = dw_lm.T  # will be added to embed grad later
        else:
            grads["lm_head.weight"] = dw_lm

        # Final norm backward
        dh, dw_fn = self._rms_norm_bwd(
            dh, cache["final_input"], self.params["final_norm.weight"],
            cache["final_rstd"], c.rms_norm_eps)
        grads["final_norm.weight"] = dw_fn

        # Layers in reverse
        for i in reversed(range(c.num_hidden_layers)):
            p = f"layers.{i}"
            lc = cache["layers"][i]

            # ── FFN backward ──
            # Residual: dh splits into ffn path and skip
            dresidual = dh.copy()

            # FFN norm backward
            dffn = dh
            # down_proj backward: dAct = dh @ W_down^T, dW_down = act^T @ dh
            dact = matmul(dffn, self.params[f"{p}.ffn.down_proj.weight"].T)
            grads[f"{p}.ffn.down_proj.weight"] = matmul(
                lc["ffn_act"].reshape(-1, c.intermediate_size).T,
                dffn.reshape(-1, c.hidden_size))

            # SwiGLU backward
            dgate, dup = self._silu_mul_bwd(dact, lc["gate"], lc["up"])

            # gate/up proj backward
            norm_out = lc["ffn_norm_out"]
            grads[f"{p}.ffn.gate_proj.weight"] = matmul(
                norm_out.reshape(-1, c.hidden_size).T,
                dgate.reshape(-1, c.intermediate_size))
            grads[f"{p}.ffn.up_proj.weight"] = matmul(
                norm_out.reshape(-1, c.hidden_size).T,
                dup.reshape(-1, c.intermediate_size))
            dh_ffn = matmul(dgate, self.params[f"{p}.ffn.gate_proj.weight"].T) + \
                     matmul(dup, self.params[f"{p}.ffn.up_proj.weight"].T)

            # FFN norm backward
            dh_ffn, dw = self._rms_norm_bwd(
                dh_ffn, lc["ffn_input"], self.params[f"{p}.ffn_norm.weight"],
                lc["ffn_norm_rstd"], c.rms_norm_eps)
            grads[f"{p}.ffn_norm.weight"] = dw
            dh = dresidual + dh_ffn  # add skip connection gradient

            # ── Attention backward ──
            dresidual = dh.copy()

            # O_proj backward:  o_proj maps [B,T,Nh*D] -> [B,T,H]
            # dh is [B,T,H], attn_proj_input is [B,T,Nh*D]
            grads[f"{p}.attn.o_proj.weight"] = matmul(
                lc["attn_proj_input"].reshape(-1, Nh * D).T,
                dh.reshape(-1, c.hidden_size))
            # dAttn = dh @ o_proj.weight^T -> [B,T,Nh*D]
            d_attn = matmul(dh, self.params[f"{p}.attn.o_proj.weight"].T)
            d_attn = d_attn.reshape(B, T, Nh, D)

            # Attention backward (through softmax + QKV)
            # d_attn: [B, T, Nh, D] -> [B, Nh, T, D]
            d_attn = xp.transpose(d_attn, (0, 2, 1, 3))
            q, k, v = lc["q"], lc["k"], lc["v"]
            scale = 1.0 / math.sqrt(D)

            if "attn_weights" in lc:
                aw = lc["attn_weights"]  # [B, Nh, T, T] (already causal)
            else:
                # Recompute attention weights from Q, K (with causal mask)
                scores = xp.matmul(q, xp.swapaxes(k, -2, -1)) * scale
                causal = xp.triu(xp.full((T, T), -1e9, dtype=xp.float32), k=1)
                scores = scores + causal
                scores -= scores.max(axis=-1, keepdims=True)
                aw = xp.exp(scores)
                aw /= aw.sum(axis=-1, keepdims=True)

            # dV = P^T @ dO  [B, Nh, T, D]
            dv = xp.matmul(xp.swapaxes(aw, -2, -1), d_attn)
            # dP = dO @ V^T  [B, Nh, T, T]
            dp = xp.matmul(d_attn, xp.swapaxes(v, -2, -1))
            # Softmax backward: dS = P * (dP - sum(P*dP, axis=-1, keepdims=True))
            ds = aw * (dp - xp.sum(aw * dp, axis=-1, keepdims=True))
            ds = ds * scale
            # dQ = dS @ K, dK = dS^T @ Q
            dq = xp.matmul(ds, k)
            dk = xp.matmul(xp.swapaxes(ds, -2, -1), q)

            # GQA: reduce dK, dV from [B, Nh, T, D] -> [B, Nkv, T, D]
            if groups > 1:
                dk = dk.reshape(B, Nkv, groups, T, D).sum(axis=2)
                dv = dv.reshape(B, Nkv, groups, T, D).sum(axis=2)

            # Transpose back: [B, heads, T, D] -> [B, T, heads, D]
            dq = xp.transpose(dq, (0, 2, 1, 3))
            dk = xp.transpose(dk, (0, 2, 1, 3))
            dv = xp.transpose(dv, (0, 2, 1, 3))

            # RoPE backward: inverse rotation (swap sin sign)
            rope_cos, rope_sin = precompute_rope(T, D, c.rope_theta)
            if xp is not np:
                rope_cos, rope_sin = xp.asarray(rope_cos), xp.asarray(rope_sin)
            cos_t = rope_cos[:T].reshape(1, T, 1, D // 2)
            sin_t = rope_sin[:T].reshape(1, T, 1, D // 2)
            # Inverse RoPE: use -sin
            dq = apply_rope(dq, xp.broadcast_to(cos_t, dq.shape[:-1] + (D // 2,)),
                            xp.broadcast_to(-sin_t, dq.shape[:-1] + (D // 2,)))
            dk = apply_rope(dk, xp.broadcast_to(cos_t, dk.shape[:-1] + (D // 2,)),
                            xp.broadcast_to(-sin_t, dk.shape[:-1] + (D // 2,)))

            # QK-norm backward (proper RMSNorm backward per-head)
            dq_flat = dq.reshape(-1, D)
            q_pre = lc["q_prenorm"].reshape(-1, D)
            dq_flat, dw_qn = self._rms_norm_bwd(
                dq_flat, q_pre, self.params[f"{p}.attn.q_norm.weight"],
                lc["q_norm_rstd"], c.rms_norm_eps)
            grads[f"{p}.attn.q_norm.weight"] = dw_qn
            dq = dq_flat.reshape(B, T, Nh, D)

            dk_flat = dk.reshape(-1, D)
            k_pre = lc["k_prenorm"].reshape(-1, D)
            dk_flat, dw_kn = self._rms_norm_bwd(
                dk_flat, k_pre, self.params[f"{p}.attn.k_norm.weight"],
                lc["k_norm_rstd"], c.rms_norm_eps)
            grads[f"{p}.attn.k_norm.weight"] = dw_kn
            dk = dk_flat.reshape(B, T, Nkv, D)

            # Reshape to flat projections: [B, T, Nh*D] / [B, T, Nkv*D]
            dq = dq.reshape(B, T, Nh * D)
            dk = dk.reshape(B, T, Nkv * D)
            dv = dv.reshape(B, T, Nkv * D)

            # QKV projection backward
            norm_out = lc["attn_norm_out"]
            grads[f"{p}.attn.q_proj.weight"] = matmul(
                norm_out.reshape(-1, c.hidden_size).T, dq.reshape(-1, Nh * D))
            grads[f"{p}.attn.k_proj.weight"] = matmul(
                norm_out.reshape(-1, c.hidden_size).T, dk.reshape(-1, Nkv * D))
            grads[f"{p}.attn.v_proj.weight"] = matmul(
                norm_out.reshape(-1, c.hidden_size).T, dv.reshape(-1, Nkv * D))

            dh_attn = matmul(dq, self.params[f"{p}.attn.q_proj.weight"].T) + \
                      matmul(dk, self.params[f"{p}.attn.k_proj.weight"].T) + \
                      matmul(dv, self.params[f"{p}.attn.v_proj.weight"].T)

            # Attention norm backward
            dh_attn, dw = self._rms_norm_bwd(
                dh_attn, lc["attn_input"], self.params[f"{p}.attn_norm.weight"],
                lc["attn_norm_rstd"], c.rms_norm_eps)
            grads[f"{p}.attn_norm.weight"] = dw
            dh = dresidual + dh_attn

        # Embedding backward: scatter dh into embedding table (vectorized)
        dembed = _zeros((c.vocab_size, c.hidden_size), xp)
        ids = cache["token_ids"]
        ids_flat = ids.reshape(-1)  # [B*T]
        dh_flat = dh.reshape(-1, c.hidden_size)  # [B*T, H]
        xp.add.at(dembed, ids_flat, dh_flat)
        if c.tie_word_embeddings:
            dembed += grads["embed.weight"]  # add LM head grad
        grads["embed.weight"] = dembed

        # Free activation cache — backward has consumed all cached tensors.
        self._cache = {}

        return grads

    def _rms_norm_bwd(self, dy, x, w, rstd, eps):
        """RMSNorm backward. Returns (dx, dw)."""
        xp = cp
        N = x.shape[-1]
        x2 = x.reshape(-1, N)
        dy2 = dy.reshape(-1, N)
        M = x2.shape[0]
        rstd2 = rstd.reshape(M, 1)
        # c = mean(w * x * dy) per row
        c = (w * x2 * dy2).sum(axis=-1, keepdims=True) / N
        dx = rstd2 * (w * dy2 - x2 * rstd2 * rstd2 * c)
        dw = (rstd2 * x2 * dy2).sum(axis=0)
        return dx.reshape(dy.shape), dw

    def _silu_mul_bwd(self, dout, gate, up):
        """SwiGLU backward. Returns (dgate, dup)."""
        xp = cp
        sig = 1.0 / (1.0 + xp.exp(-gate.clip(-88, 88)))
        silu_g = gate * sig
        dgate = dout * up * (sig + gate * sig * (1.0 - sig))
        dup = dout * silu_g
        return dgate, dup

    def parameters(self) -> dict[str, np.ndarray]:
        """All model parameters."""
        return self.params

    def load_weights(self, path: str):
        """Load weights from safetensors file (native or HuggingFace format).

        Detects HuggingFace naming (keys starting with 'model.') and
        automatically maps names + transposes linear projections.
        Converts bfloat16 → float32.
        """
        xp = cp if cp is not np else np

        # HuggingFace name → our name mapping
        _HF_MAP = {
            "model.embed_tokens.weight": "embed.weight",
            "model.norm.weight": "final_norm.weight",
        }
        # Per-layer HF → internal key suffixes
        _HF_LAYER_MAP = {
            "input_layernorm.weight": "attn_norm.weight",
            "self_attn.q_proj.weight": "attn.q_proj.weight",
            "self_attn.k_proj.weight": "attn.k_proj.weight",
            "self_attn.v_proj.weight": "attn.v_proj.weight",
            "self_attn.o_proj.weight": "attn.o_proj.weight",
            "self_attn.q_norm.weight": "attn.q_norm.weight",
            "self_attn.k_norm.weight": "attn.k_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "mlp.gate_proj.weight": "ffn.gate_proj.weight",
            "mlp.up_proj.weight": "ffn.up_proj.weight",
            "mlp.down_proj.weight": "ffn.down_proj.weight",
        }
        # Keys whose weights need transposing (nn.Linear: [out,in] → our [in,out])
        _TRANSPOSE = {"q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"}

        def _map_hf_key(hf_key):
            """Map HuggingFace key to internal key."""
            if hf_key in _HF_MAP:
                return _HF_MAP[hf_key]
            # model.layers.{i}.suffix
            if hf_key.startswith("model.layers."):
                parts = hf_key.split(".", 3)  # ['model','layers','{i}','suffix']
                idx, suffix = parts[2], parts[3]
                if suffix in _HF_LAYER_MAP:
                    return f"layers.{idx}.{_HF_LAYER_MAP[suffix]}"
            # lm_head handled separately (tied weights)
            return hf_key

        def _needs_transpose(key):
            for t in _TRANSPOSE:
                if t in key:
                    return True
            return False

        def _assign(name, tensor):
            # Convert bf16/fp16 → fp32
            if hasattr(tensor, 'dtype') and tensor.dtype != np.float32:
                tensor = tensor.astype(np.float32)
            if name in self.params:
                if self.params[name].shape != tensor.shape:
                    raise ValueError(f"Shape mismatch for {name}: "
                                     f"expected {self.params[name].shape}, got {tensor.shape}")
                self.params[name] = xp.asarray(tensor) if xp is not np else tensor

        def _load_tensors(path):
            try:
                import ml_dtypes  # noqa: F401 — registers bfloat16 with numpy
            except ImportError:
                pass
            from safetensors import safe_open
            import glob as _glob
            # Accept a directory or a single file
            if os.path.isdir(path):
                files = sorted(_glob.glob(os.path.join(path, "*.safetensors")))
                if not files:
                    raise FileNotFoundError(f"No .safetensors files in {path}")
            else:
                files = [path]
            tensors = {}
            for fpath in files:
                with safe_open(fpath, framework="numpy") as f:
                    for k in f.keys():
                        tensors[k] = f.get_tensor(k)
            return tensors

        tensors = _load_tensors(path)
        is_hf = any(k.startswith("model.") for k in tensors)

        if is_hf:
            for hf_key, tensor in tensors.items():
                # Skip lm_head for tied-weight models
                if hf_key == "lm_head.weight" and self.cfg.tie_word_embeddings:
                    continue
                name = _map_hf_key(hf_key)
                if _needs_transpose(name) and tensor.ndim == 2:
                    tensor = tensor.T
                _assign(name, tensor)
        else:
            for k, v in tensors.items():
                _assign(k, v)

    def _alloc_kv_cache(self, B, max_len):
        """Pre-allocate KV cache buffers for all layers.

        Returns list of (K_buf, V_buf) per layer, each shape [B, Nkv, max_len, D].
        """
        xp = cp
        Nkv, D = self.cfg.num_key_value_heads, self.cfg.head_dim
        cache = []
        for _ in range(self.cfg.num_hidden_layers):
            k = xp.zeros((B, Nkv, max_len, D), dtype=xp.float32)
            v = xp.zeros((B, Nkv, max_len, D), dtype=xp.float32)
            cache.append((k, v))
        return cache

    def forward_inference(self, token_ids, kv_cache, seq_len, rope_cos, rope_sin):
        """Forward pass for inference with pre-allocated KV cache.

        Computes logits only for new tokens, writes K/V into pre-allocated
        cache buffers at positions [seq_len : seq_len + T_new].

        Args:
            token_ids: [B, T_new] int array of NEW token IDs only.
            kv_cache: list of (K_buf, V_buf) from _alloc_kv_cache().
                      Mutated in-place. K/V shape: [B, Nkv, max_len, D].
            seq_len: number of tokens already in the cache.
            rope_cos: pre-computed cos, shape [max_len, D//2].
            rope_sin: pre-computed sin, shape [max_len, D//2].

        Returns:
            logits: [B, T_new, V] float array.
        """
        xp = cp
        c = self.cfg
        B, T_new = token_ids.shape
        start_pos = seq_len
        T_total = start_pos + T_new

        # Embedding lookup — only for new tokens
        hidden = self.params["embed.weight"][token_ids]  # [B, T_new, H]

        # RoPE slice for new positions only
        rc = rope_cos[start_pos:T_total]  # [T_new, D//2]
        rs = rope_sin[start_pos:T_total]

        Nh, Nkv, D = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        groups = Nh // Nkv
        scale = np.float32(1.0 / math.sqrt(D))

        for i in range(c.num_hidden_layers):
            p = f"layers.{i}"

            # ── Attention block ──
            residual = hidden
            hidden, _ = rms_norm(hidden, self.params[f"{p}.attn_norm.weight"],
                                 c.rms_norm_eps, return_rstd=True)

            # QKV projections for new tokens only
            q = matmul(hidden, self.params[f"{p}.attn.q_proj.weight"])
            k_new = matmul(hidden, self.params[f"{p}.attn.k_proj.weight"])
            v_new = matmul(hidden, self.params[f"{p}.attn.v_proj.weight"])

            q = q.reshape(B, T_new, Nh, D)
            k_new = k_new.reshape(B, T_new, Nkv, D)
            v_new = v_new.reshape(B, T_new, Nkv, D)

            # QK-Norm (per-head RMSNorm)
            q_flat, _ = rms_norm(q.reshape(-1, D),
                self.params[f"{p}.attn.q_norm.weight"],
                c.rms_norm_eps, return_rstd=True)
            q = q_flat.reshape(q.shape)
            k_flat, _ = rms_norm(k_new.reshape(-1, D),
                self.params[f"{p}.attn.k_norm.weight"],
                c.rms_norm_eps, return_rstd=True)
            k_new = k_flat.reshape(k_new.shape)

            # RoPE on new tokens only
            cos_t = rc.reshape(1, T_new, 1, D // 2)
            sin_t = rs.reshape(1, T_new, 1, D // 2)
            q = apply_rope(q, xp.broadcast_to(cos_t, (B, T_new, Nh, D // 2)),
                           xp.broadcast_to(sin_t, (B, T_new, Nh, D // 2)))
            k_new = apply_rope(k_new,
                               xp.broadcast_to(cos_t, (B, T_new, Nkv, D // 2)),
                               xp.broadcast_to(sin_t, (B, T_new, Nkv, D // 2)))

            # Transpose: [B, T, heads, D] -> [B, heads, T, D]
            q = xp.transpose(q, (0, 2, 1, 3))          # [B, Nh, T_new, D]
            k_new = xp.transpose(k_new, (0, 2, 1, 3))  # [B, Nkv, T_new, D]
            v_new = xp.transpose(v_new, (0, 2, 1, 3))  # [B, Nkv, T_new, D]

            # Write new K/V into pre-allocated buffers (no copy/concat)
            k_buf, v_buf = kv_cache[i]
            k_buf[:, :, start_pos:T_total, :] = k_new
            v_buf[:, :, start_pos:T_total, :] = v_new

            # Attention over [0, T_total) using buffer slices
            k_full = k_buf[:, :, :T_total, :]  # [B, Nkv, T_total, D] — view, no copy
            v_full = v_buf[:, :, :T_total, :]

            # GQA: broadcast KV heads instead of repeat (avoid copy)
            if groups > 1:
                # [B, Nkv, T, D] -> [B, Nkv, 1, T, D] -> [B, Nkv, groups, T, D] -> [B, Nh, T, D]
                k_attn = xp.broadcast_to(
                    k_full[:, :, xp.newaxis, :, :],
                    (B, Nkv, groups, T_total, D)
                ).reshape(B, Nh, T_total, D)
                v_attn = xp.broadcast_to(
                    v_full[:, :, xp.newaxis, :, :],
                    (B, Nkv, groups, T_total, D)
                ).reshape(B, Nh, T_total, D)
            else:
                k_attn = k_full
                v_attn = v_full

            # Attention: Q [B, Nh, T_new, D] × K^T [B, Nh, D, T_total]
            scores = xp.matmul(q, xp.swapaxes(k_attn, -2, -1)) * scale

            # Causal mask
            if T_new > 1:
                row_pos = xp.arange(start_pos, T_total).reshape(T_new, 1)
                col_pos = xp.arange(T_total).reshape(1, T_total)
                causal = xp.where(col_pos > row_pos,
                                  xp.asarray(np.float32(-1e9)),
                                  xp.asarray(np.float32(0.0)))
                scores = scores + causal
            # T_new == 1: no mask needed (single decode token attends to all)

            scores = scores - scores.max(axis=-1, keepdims=True)
            weights = xp.exp(scores.astype(xp.float32))
            weights = weights / weights.sum(axis=-1, keepdims=True)
            attn_out = xp.matmul(weights, v_attn).astype(xp.float32)

            # Transpose back: [B, Nh, T_new, D] -> [B, T_new, Nh*D]
            attn_out = xp.transpose(attn_out, (0, 2, 1, 3)).reshape(B, T_new, Nh * D)
            hidden = matmul(attn_out, self.params[f"{p}.attn.o_proj.weight"]) + residual

            # ── FFN block ──
            residual = hidden
            hidden, _ = rms_norm(hidden, self.params[f"{p}.ffn_norm.weight"],
                                 c.rms_norm_eps, return_rstd=True)
            gate = matmul(hidden, self.params[f"{p}.ffn.gate_proj.weight"])
            up = matmul(hidden, self.params[f"{p}.ffn.up_proj.weight"])
            ffn_act = silu_mul(gate, up)
            hidden = matmul(ffn_act, self.params[f"{p}.ffn.down_proj.weight"]) + residual

        # Final norm + LM head
        hidden, _ = rms_norm(hidden, self.params["final_norm.weight"],
                             c.rms_norm_eps, return_rstd=True)
        logits = matmul(hidden, self._get_lm_head_weight())  # [B, T_new, V]
        return logits

    def generate(self, prompt_ids, max_new_tokens=256, temperature=0.0, top_k=0,
                 eos_token_id=None):
        """Autoregressive generation with pre-allocated KV cache.

        Prefills the prompt in one pass, then decodes one token at a time
        reusing cached key/value pairs.  Much faster than full-sequence forward.

        Args:
            prompt_ids: 1-D array of token IDs (the prompt).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_k: Top-k sampling (0 = disabled).
            eos_token_id: Stop generation when this token is produced.

        Returns:
            1-D numpy array of generated token IDs (prompt + completion).
        """
        xp = cp
        prompt_len = len(prompt_ids)
        max_len = prompt_len + max_new_tokens
        ids = list(int(x) for x in prompt_ids)

        # Pre-allocate KV cache and RoPE tables once
        kv_cache = self._alloc_kv_cache(1, max_len)
        rope_cos, rope_sin = precompute_rope(max_len, self.cfg.head_dim, self.cfg.rope_theta)
        if xp is not np:
            rope_cos, rope_sin = xp.asarray(rope_cos), xp.asarray(rope_sin)

        # Prefill: process entire prompt at once
        input_ids = xp.array([ids], dtype=xp.int32)  # [1, T_prompt]
        logits = self.forward_inference(input_ids, kv_cache, seq_len=0,
                                        rope_cos=rope_cos, rope_sin=rope_sin)
        next_logits = logits[0, -1]  # [V]
        seq_len = prompt_len

        for _ in range(max_new_tokens):
            if hasattr(next_logits, 'get'):
                nl = next_logits.get()
            else:
                nl = next_logits
            nl = nl.astype(np.float32)

            if temperature <= 0:
                token = int(np.argmax(nl))
            else:
                nl = nl / temperature
                if top_k > 0:
                    topk_idx = np.argpartition(nl, -top_k)[-top_k:]
                    mask = np.full_like(nl, -np.inf)
                    mask[topk_idx] = nl[topk_idx]
                    nl = mask
                nl -= nl.max()
                probs = np.exp(nl)
                probs /= probs.sum()
                token = int(np.random.choice(len(probs), p=probs))

            ids.append(token)
            if eos_token_id is not None and token == eos_token_id:
                break

            # Decode: single new token with KV cache
            new_tok = xp.array([[token]], dtype=xp.int32)
            logits = self.forward_inference(new_tok, kv_cache, seq_len=seq_len,
                                            rope_cos=rope_cos, rope_sin=rope_sin)
            next_logits = logits[0, 0]
            seq_len += 1

        return np.array(ids, dtype=np.int32)


# ── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Qwen3 smoke test — forward + backward")
    print("=" * 60)
    np.random.seed(42)
    cfg = Qwen3Config.tiny()
    model = Qwen3Model(cfg)
    B, T = 2, 8

    print(f"Config: {cfg.hidden_size}H, {cfg.num_hidden_layers}L, "
          f"{cfg.num_attention_heads}Nh, {cfg.num_key_value_heads}Nkv, "
          f"vocab={cfg.vocab_size}")
    print(f"Params: {sum(p.size for p in model.params.values()):,}")

    # Forward
    ids = np.random.randint(0, cfg.vocab_size, (B, T))
    logits = model.forward(ids)
    assert logits.shape == (B, T, cfg.vocab_size), f"Bad logits shape: {logits.shape}"
    print(f"[forward]  logits shape={logits.shape}, "
          f"mean={logits.mean():.4f}, std={logits.std():.4f}")

    # Backward with dummy grad (cross-entropy-like)
    targets = np.random.randint(0, cfg.vocab_size, (B, T))
    # Softmax + cross-entropy gradient: softmax(logits) - one_hot(targets)
    logits_shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits_shifted) / np.exp(logits_shifted).sum(axis=-1, keepdims=True)
    grad_logits = probs.copy()
    for b in range(B):
        for t in range(T):
            grad_logits[b, t, targets[b, t]] -= 1.0
    grad_logits /= (B * T)

    grads = model.backward(grad_logits)
    print(f"[backward] computed {len(grads)} parameter gradients")

    # Verify gradient shapes match parameter shapes
    for name, param in model.params.items():
        assert name in grads, f"Missing gradient for {name}"
        assert grads[name].shape == param.shape, \
            f"Shape mismatch for {name}: param={param.shape}, grad={grads[name].shape}"
    print("[shapes]   All gradient shapes match parameter shapes")

    # Numerical gradient check (float64 loss for precision)
    def _ce_loss(lgts, tgt):
        if hasattr(lgts, 'get'):
            lgts = lgts.get()
        if hasattr(tgt, 'get'):
            tgt = tgt.get()
        l = lgts.astype(np.float64)
        l -= l.max(-1, keepdims=True)
        lp = l - np.log(np.exp(l).sum(-1, keepdims=True))
        return -np.mean([lp[b, t, tgt[b, t]] for b in range(B) for t in range(T)])

    # Compute analytical gradients once from original forward
    model.forward(ids)
    grads = model.backward(grad_logits)

    # Check projection weights (large gradients, reliable numerical check)
    test_params = [
        ("layers.0.ffn.gate_proj.weight", (0, 0)),
        ("layers.0.ffn.down_proj.weight", (1, 0)),
        ("layers.0.attn.q_proj.weight", (0, 0)),
        ("layers.0.attn.o_proj.weight", (0, 0)),
        ("final_norm.weight", (0,)),
    ]
    eps_fd = 1e-3  # larger eps for float32 forward pass
    all_ok = True
    for pname, idx in test_params:
        orig = float(model.params[pname][idx])
        model.params[pname][idx] = orig + eps_fd
        loss_p = _ce_loss(model.forward(ids), targets)
        model.params[pname][idx] = orig - eps_fd
        loss_m = _ce_loss(model.forward(ids), targets)
        model.params[pname][idx] = orig
        num_g = (loss_p - loss_m) / (2 * eps_fd)
        ana_g = float(grads[pname][idx])
        rel = abs(num_g - ana_g) / (abs(num_g) + abs(ana_g) + 1e-12)
        ok = rel < 0.15  # float32 forward + multi-layer tolerance
        if not ok:
            all_ok = False
        print(f"[grad_check] {pname}{list(idx)}: "
              f"num={num_g:.6f} ana={ana_g:.6f} rel={rel:.4f} {'OK' if ok else 'FAIL'}")
    assert all_ok, "Gradient check failed"
    print("[grad_check] PASSED")

    if _USE_CUTILE_KERNELS:
        backend = "cuTile kernels"
    elif cp is not np:
        backend = "CuPy GPU"
    else:
        backend = "NumPy CPU"
    print("=" * 60)
    print(f"All tests passed ({backend}, TILERL_BACKEND={_BACKEND or 'auto'}).")
