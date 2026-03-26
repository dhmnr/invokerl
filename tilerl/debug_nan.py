"""Debug NaN in forward pass at T=9."""
import cupy as cp
import numpy as np
from qwen3 import (Qwen3Config, Qwen3Model, rms_norm, matmul,
                    precompute_rope, apply_rope, silu_mul)
import math

cfg = Qwen3Config.qwen3_0_6b()
model = Qwen3Model(cfg)
model.load_weights(
    "/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    "c1899de289a04d12100db370d81485cdf75e47ca/"
)

T = 9
ids = cp.arange(1, T + 1, dtype=cp.int32).reshape(1, -1)
B = 1
c = cfg
D, Nh, Nkv = c.head_dim, c.num_attention_heads, c.num_key_value_heads
groups = Nh // Nkv

hidden = model.params["embed.weight"][ids]
rope_cos, rope_sin = precompute_rope(T, D, c.rope_theta)
rope_cos, rope_sin = cp.asarray(rope_cos), cp.asarray(rope_sin)

for layer in range(c.num_hidden_layers):
    p = "layers.%d" % layer
    residual = hidden

    # Attention block
    norm_h = rms_norm(hidden, model.params[p + ".attn_norm.weight"], c.rms_norm_eps)
    q = matmul(norm_h, model.params[p + ".attn.q_proj.weight"]).reshape(B, T, Nh, D)
    k = matmul(norm_h, model.params[p + ".attn.k_proj.weight"]).reshape(B, T, Nkv, D)
    v = matmul(norm_h, model.params[p + ".attn.v_proj.weight"]).reshape(B, T, Nkv, D)

    q_flat = rms_norm(q.reshape(-1, D), model.params[p + ".attn.q_norm.weight"], c.rms_norm_eps)
    k_flat = rms_norm(k.reshape(-1, D), model.params[p + ".attn.k_norm.weight"], c.rms_norm_eps)
    q = q_flat.reshape(B, T, Nh, D)
    k = k_flat.reshape(B, T, Nkv, D)

    cos_t = rope_cos[:T].reshape(1, T, 1, D // 2)
    sin_t = rope_sin[:T].reshape(1, T, 1, D // 2)
    q = apply_rope(q, cp.broadcast_to(cos_t, (B, T, Nh, D // 2)),
                   cp.broadcast_to(sin_t, (B, T, Nh, D // 2)))
    k = apply_rope(k, cp.broadcast_to(cos_t, (B, T, Nkv, D // 2)),
                   cp.broadcast_to(sin_t, (B, T, Nkv, D // 2)))

    if bool(cp.isnan(q).any()):
        print("Layer %d: NaN after RoPE" % layer)
        break

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

    if bool(cp.isnan(attn_out).any()):
        print("Layer %d: NaN in attention" % layer)
        break

    hidden = matmul(attn_out, model.params[p + ".attn.o_proj.weight"]) + residual

    if bool(cp.isnan(hidden).any()):
        print("Layer %d: NaN after O_proj+residual" % layer)
        break

    # FFN block
    residual = hidden
    norm_h = rms_norm(hidden, model.params[p + ".ffn_norm.weight"], c.rms_norm_eps)
    gate = matmul(norm_h, model.params[p + ".ffn.gate_proj.weight"])
    up = matmul(norm_h, model.params[p + ".ffn.up_proj.weight"])

    if bool(cp.isnan(gate).any()):
        print("Layer %d: NaN in gate_proj" % layer)
        break

    act = silu_mul(gate, up)

    if bool(cp.isnan(act).any()):
        print("Layer %d: NaN in silu_mul" % layer)
        print("  gate max=%.2f" % float(cp.abs(gate).max()))
        print("  up max=%.2f" % float(cp.abs(up).max()))
        print("  act max=%.2f" % float(cp.abs(act[~cp.isnan(act)]).max()))
        break

    hidden = matmul(act, model.params[p + ".ffn.down_proj.weight"]) + residual

    if bool(cp.isnan(hidden).any()):
        print("Layer %d: NaN after FFN" % layer)
        break

    max_val = float(cp.abs(hidden).max())
    if layer < 3 or layer >= 25 or layer % 5 == 0:
        print("Layer %2d: max=%.2f" % (layer, max_val))

if not bool(cp.isnan(hidden).any()):
    print("All %d layers OK, max=%.2f" % (c.num_hidden_layers, float(cp.abs(hidden).max())))
