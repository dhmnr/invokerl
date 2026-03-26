"""Qwen3-0.6B inference using pure cuTile + CuPy.

Downloads model from HuggingFace, maps weights to our Qwen3Model, and runs
autoregressive text generation.
"""
from __future__ import annotations

import sys
import time

import os

import numpy as np

# Backend selection — mirrors qwen3.py pattern
_BACKEND = os.environ.get("TILERL_BACKEND", "").lower()

if _BACKEND == "numpy":
    import numpy as cp  # type: ignore[assignment]
else:
    try:
        import cupy as cp
    except ImportError:
        cp = np  # type: ignore[assignment]

from qwen3 import Qwen3Model, Qwen3Config


# HuggingFace → our weight name mapping
def _hf_to_ours(hf_name: str) -> str | None:
    """Map a HuggingFace Qwen3 weight name to our internal name."""
    n = hf_name
    # Embedding
    if n == "model.embed_tokens.weight":
        return "embed.weight"
    # Final norm
    if n == "model.norm.weight":
        return "final_norm.weight"
    # LM head (tied — skip if tie_word_embeddings)
    if n == "lm_head.weight":
        return None  # tied with embed.weight
    # Per-layer weights
    if n.startswith("model.layers."):
        parts = n.split(".")
        layer_idx = parts[2]
        rest = ".".join(parts[3:])
        p = f"layers.{layer_idx}"
        mapping = {
            "input_layernorm.weight": f"{p}.attn_norm.weight",
            "self_attn.q_proj.weight": f"{p}.attn.q_proj.weight",
            "self_attn.k_proj.weight": f"{p}.attn.k_proj.weight",
            "self_attn.v_proj.weight": f"{p}.attn.v_proj.weight",
            "self_attn.q_norm.weight": f"{p}.attn.q_norm.weight",
            "self_attn.k_norm.weight": f"{p}.attn.k_norm.weight",
            "self_attn.o_proj.weight": f"{p}.attn.o_proj.weight",
            "post_attention_layernorm.weight": f"{p}.ffn_norm.weight",
            "mlp.gate_proj.weight": f"{p}.ffn.gate_proj.weight",
            "mlp.up_proj.weight": f"{p}.ffn.up_proj.weight",
            "mlp.down_proj.weight": f"{p}.ffn.down_proj.weight",
        }
        return mapping.get(rest)
    return None


def load_hf_weights(model: Qwen3Model, repo_id: str):
    """Download and load weights from a HuggingFace repo."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import glob
    import os
    import ml_dtypes  # noqa: F401 — registers bfloat16 with numpy

    print(f"Downloading {repo_id}...")
    local_dir = snapshot_download(
        repo_id,
        allow_patterns=["*.safetensors", "tokenizer*", "*.json"],
    )
    print(f"  -> {local_dir}")

    # Load all safetensor shards
    shard_files = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
    print(f"  -> {len(shard_files)} safetensor file(s)")

    # Linear projection suffixes — these weights are stored [out, in] in HF
    # but [in, out] in our model, so they ALWAYS need transposing (even when square)
    _LINEAR_SUFFIXES = {
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
    }

    loaded, skipped, missing = 0, 0, 0
    for sf in shard_files:
        with safe_open(sf, framework="numpy") as f:
            for hf_name in f.keys():
                tensor = f.get_tensor(hf_name)
                if tensor.dtype != np.float32:
                    tensor = tensor.astype(np.float32)
                our_name = _hf_to_ours(hf_name)
                if our_name is None:
                    skipped += 1
                    continue
                if our_name not in model.params:
                    print(f"  WARN: mapped '{hf_name}' -> '{our_name}' but not in model")
                    missing += 1
                    continue
                # HF stores linear weights as [out, in], we store as [in, out]
                # Always transpose linear weights — shape check alone fails for
                # square matrices (e.g. k_proj [1024,1024] in Qwen3-0.6B)
                is_linear = any(our_name.endswith(s) for s in _LINEAR_SUFFIXES)
                if is_linear and tensor.ndim == 2:
                    tensor = tensor.T
                expected_shape = model.params[our_name].shape
                if tensor.shape != expected_shape:
                    print(f"  WARN: shape mismatch for {our_name}: "
                          f"expected {expected_shape}, got {tensor.shape}")
                    continue
                model.params[our_name] = cp.asarray(tensor) if cp is not np else tensor
                loaded += 1

    print(f"  Loaded {loaded} params, skipped {skipped}, missing {missing}")
    return local_dir


def greedy_generate(model: Qwen3Model, token_ids: list[int], max_new: int = 50,
                    temperature: float = 0.0) -> list[int]:
    """Autoregressive greedy/sampling generation."""
    xp = cp
    generated = list(token_ids)

    for _ in range(max_new):
        ids = xp.array([generated], dtype=xp.int32)  # [1, seq_len]
        logits = model.forward(ids)  # [1, seq_len, V]
        # Get logits for last position
        next_logits = logits[0, -1, :]  # [V]
        if hasattr(next_logits, 'get'):
            next_logits = next_logits.get()
        next_logits = next_logits.astype(np.float64)

        if temperature <= 0:
            # Greedy
            next_token = int(np.argmax(next_logits))
        else:
            # Sample
            probs = np.exp(next_logits - next_logits.max())
            probs /= probs.sum()
            next_token = int(np.random.choice(len(probs), p=probs))

        generated.append(next_token)

        # Stop on EOS (Qwen3: 151645 = <|endoftext|>, 151643 = <|im_end|>)
        if next_token in (151645, 151643):
            break

    return generated


if __name__ == "__main__":
    repo_id = "Qwen/Qwen3-0.6B"
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"

    print(f"=== Qwen3-0.6B Inference (cuTile + CuPy) ===\n")

    # Init model
    cfg = Qwen3Config.qwen3_0_6b()
    model = Qwen3Model(cfg)
    print(f"Model: {sum(p.size for p in model.params.values()):,} params")

    # Load weights
    local_dir = load_hf_weights(model, repo_id)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    print(f"Tokenizer: {tokenizer.__class__.__name__}, vocab={len(tokenizer)}\n")

    # Tokenize
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"Prompt: {prompt!r}")
    print(f"Tokens: {input_ids[:20]}{'...' if len(input_ids) > 20 else ''}\n")

    # Generate
    print("Generating...")
    t0 = time.time()
    output_ids = greedy_generate(model, input_ids, max_new=50)
    dt = time.time() - t0
    new_tokens = len(output_ids) - len(input_ids)
    print(f"Generated {new_tokens} tokens in {dt:.2f}s ({new_tokens/dt:.1f} tok/s)\n")

    # Decode
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"Output: {output_text}")
