"""Analytical FLOP and memory-traffic estimates per training phase.

Used for roofline analysis and MFU calculation. Numbers are rough — the
attention FLOP count is a per-sequence approximation and ignores causal
masking, and the KV cache memory model assumes all tokens are decoded
sequentially.
"""

from __future__ import annotations


def estimate_flops_per_step(
    hidden_size: int = 1024,
    num_layers: int = 28,
    num_attention_heads: int = 16,
    num_kv_heads: int = 8,
    intermediate_size: int = 3072,
    vocab_size: int = 151936,
    seq_len: int = 512,
    batch_size: int = 4,
    group_size: int = 4,
    accumulation_steps: int = 4,
) -> dict[str, dict]:
    """Estimate FLOPs and memory traffic for each training phase.

    Returns {phase_name: {flops, bytes, arithmetic_intensity, description}}.
    All values are per optimizer step (summed over micro-batches).
    """
    B_gen = batch_size * group_size
    B_train = B_gen * accumulation_steps
    S = seq_len
    H = hidden_size
    L = num_layers
    V = vocab_size
    I = intermediate_size
    d = H // num_attention_heads

    qkv_flops = 2 * S * H * (H + 2 * (num_kv_heads * d))
    attn_flops = 2 * S * S * H
    out_proj_flops = 2 * S * H * H
    ffn_flops = 2 * S * H * I * 3
    per_layer = qkv_flops + attn_flops + out_proj_flops + ffn_flops
    lm_head_flops = 2 * S * H * V

    fwd_flops = L * per_layer + lm_head_flops
    bwd_flops = 2 * fwd_flops

    params_per_layer = (
        H * (H + 2 * num_kv_heads * d)
        + H * H
        + H * I * 3
        + H * 4
    )
    total_params = L * params_per_layer + V * H
    param_bytes_bf16 = total_params * 2
    param_bytes_fp32 = total_params * 4

    results = {}

    gen_flops = B_gen * fwd_flops
    gen_kv_bytes = S * B_gen * L * 2 * num_kv_heads * d * 2
    gen_bytes = S * param_bytes_bf16 + gen_kv_bytes
    results["generation"] = {
        "flops": gen_flops,
        "bytes": gen_bytes,
        "description": f"{B_gen} sequences x {S} tokens (autoregressive, memory-bound)",
    }

    results["reward"] = {
        "flops": 0,
        "bytes": 0,
        "description": "CPU-only rule-based matching",
    }

    K = accumulation_steps
    ref_fwd_flops = B_train * fwd_flops
    ref_fwd_bytes = K * param_bytes_bf16 + B_train * S * H * 2
    results["ref_forward"] = {
        "flops": ref_fwd_flops,
        "bytes": ref_fwd_bytes,
        "description": f"{B_train} sequences ({K} micro-batches) forward (bf16, no grad)",
    }

    pol_fwd_flops = B_train * fwd_flops
    pol_fwd_bytes = K * param_bytes_bf16 + B_train * S * H * 2
    results["policy_forward"] = {
        "flops": pol_fwd_flops,
        "bytes": pol_fwd_bytes,
        "description": f"{B_train} sequences forward with grad (bf16 autocast)",
    }

    loss_flops = B_train * S * 20
    loss_bytes = B_train * S * 4 * 6
    results["loss_computation"] = {
        "flops": loss_flops,
        "bytes": loss_bytes,
        "description": "Clipped surrogate + KL penalty",
    }

    bwd_total_flops = B_train * bwd_flops
    bwd_bytes = K * (param_bytes_bf16 + param_bytes_fp32) + B_train * S * H * 4
    results["backward"] = {
        "flops": bwd_total_flops,
        "bytes": bwd_bytes,
        "description": f"{B_train} sequences backward (gradient checkpointing)",
    }

    opt_flops = total_params * 10
    opt_bytes = param_bytes_fp32 * 3
    results["optimizer_step"] = {
        "flops": opt_flops,
        "bytes": opt_bytes,
        "description": "AdamW update + grad clipping",
    }

    results["weight_sync"] = {
        "flops": 0,
        "bytes": param_bytes_bf16 * 2,
        "description": "safetensors save → disk → vLLM reload_weights",
    }

    for phase in results.values():
        if phase["bytes"] > 0:
            phase["arithmetic_intensity"] = phase["flops"] / phase["bytes"]
        else:
            phase["arithmetic_intensity"] = 0.0

    return results
