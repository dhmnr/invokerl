"""Test forward pass using ONLY CuPy operations (no cuTile kernels).
If this produces correct output, the bug is in cuTile kernels."""
import cupy as cp
import numpy as np
import math

from qwen3 import Qwen3Config, Qwen3Model, precompute_rope


def rms_norm_cupy(x, w, eps):
    """Pure CuPy RMSNorm."""
    x = x.astype(cp.float32)
    ms = (x * x).mean(axis=-1, keepdims=True)
    rs = 1.0 / cp.sqrt(ms + eps)
    return (x * rs * w).astype(cp.float32)


def silu_mul_cupy(gate, up):
    """Pure CuPy SwiGLU."""
    sig = 1.0 / (1.0 + cp.exp(-gate.astype(cp.float32)))
    return (gate * sig * up).astype(cp.float32)


def apply_rope_cupy(x, cos, sin):
    """Pure CuPy RoPE (split-half)."""
    D = x.shape[-1]
    half = D // 2
    x0 = x[..., :half]
    x1 = x[..., half:]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return cp.concatenate([out0, out1], axis=-1).astype(x.dtype)


cfg = Qwen3Config.qwen3_0_6b()
model = Qwen3Model(cfg)
model.load_weights(
    "/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    "c1899de289a04d12100db370d81485cdf75e47ca/"
)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

c = cfg
D, Nh, Nkv = c.head_dim, c.num_attention_heads, c.num_key_value_heads
groups = Nh // Nkv

# Generate step by step
prompt = "<|im_start|>user\nWhat is 2+3? Give only the number.\n<|im_end|>\n<|im_start|>assistant\n"
ids = list(tokenizer.encode(prompt))
print("Prompt: %d tokens" % len(ids))

for step in range(50):
    T = len(ids)
    input_ids = cp.array([ids], dtype=cp.int32)
    B = 1

    # Embedding
    hidden = model.params["embed.weight"][input_ids]

    rope_cos_np, rope_sin_np = precompute_rope(T, D, c.rope_theta)
    rope_cos = cp.asarray(rope_cos_np)
    rope_sin = cp.asarray(rope_sin_np)

    for layer in range(c.num_hidden_layers):
        p = "layers.%d" % layer
        residual = hidden

        # Attention
        norm_h = rms_norm_cupy(hidden, model.params[p + ".attn_norm.weight"], c.rms_norm_eps)
        q = cp.matmul(norm_h, model.params[p + ".attn.q_proj.weight"]).reshape(B, T, Nh, D)
        k = cp.matmul(norm_h, model.params[p + ".attn.k_proj.weight"]).reshape(B, T, Nkv, D)
        v = cp.matmul(norm_h, model.params[p + ".attn.v_proj.weight"]).reshape(B, T, Nkv, D)

        # QK-norm
        q = rms_norm_cupy(q, model.params[p + ".attn.q_norm.weight"], c.rms_norm_eps)
        k = rms_norm_cupy(k, model.params[p + ".attn.k_norm.weight"], c.rms_norm_eps)

        # RoPE
        cos_t = rope_cos[:T].reshape(1, T, 1, D // 2)
        sin_t = rope_sin[:T].reshape(1, T, 1, D // 2)
        q = apply_rope_cupy(q, cp.broadcast_to(cos_t, (B, T, Nh, D // 2)),
                            cp.broadcast_to(sin_t, (B, T, Nh, D // 2)))
        k = apply_rope_cupy(k, cp.broadcast_to(cos_t, (B, T, Nkv, D // 2)),
                            cp.broadcast_to(sin_t, (B, T, Nkv, D // 2)))

        # GQA attention
        q2 = cp.transpose(q, (0, 2, 1, 3))
        k2 = cp.transpose(k, (0, 2, 1, 3))
        v2 = cp.transpose(v, (0, 2, 1, 3))
        if groups > 1:
            k2 = cp.repeat(k2, groups, axis=1)
            v2 = cp.repeat(v2, groups, axis=1)

        scale = np.float32(1.0 / math.sqrt(D))
        scores = cp.matmul(q2, cp.swapaxes(k2, -2, -1)) * scale
        causal = cp.asarray(np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1))
        scores = (scores + causal).astype(cp.float32)
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = cp.exp(scores).astype(cp.float32)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        attn_out = cp.matmul(weights, v2).astype(cp.float32)
        attn_out = cp.transpose(attn_out, (0, 2, 1, 3)).reshape(B, T, Nh * D)

        hidden = cp.matmul(attn_out, model.params[p + ".attn.o_proj.weight"]) + residual

        # FFN
        residual = hidden
        norm_h = rms_norm_cupy(hidden, model.params[p + ".ffn_norm.weight"], c.rms_norm_eps)
        gate = cp.matmul(norm_h, model.params[p + ".ffn.gate_proj.weight"])
        up = cp.matmul(norm_h, model.params[p + ".ffn.up_proj.weight"])
        act = silu_mul_cupy(gate, up)
        hidden = cp.matmul(act, model.params[p + ".ffn.down_proj.weight"]) + residual

    # Final norm + LM head
    hidden = rms_norm_cupy(hidden, model.params["final_norm.weight"], c.rms_norm_eps)
    logits = cp.matmul(hidden, model.params["embed.weight"].T)  # tied weights
    next_logits = logits[0, -1].get().astype(np.float64)

    if np.isnan(next_logits).any():
        print("Step %d: NaN! seq_len=%d" % (step, T))
        break

    token = int(np.argmax(next_logits))
    ids.append(token)
    word = tokenizer.decode([token])

    if token == 151645:  # eos
        print("Step %d: EOS" % step)
        break

    if step < 30:
        print("Step %2d (T=%3d): %s" % (step, T, repr(word)))

response = tokenizer.decode(ids[len(tokenizer.encode(prompt)):], skip_special_tokens=False)
print("\nFull response:\n%s" % response[:500])
