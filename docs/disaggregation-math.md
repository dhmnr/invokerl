# Disaggregated RL Training: Generation vs Training Saturation

## The Question

How many generation GPUs does it take to saturate one training GPU in an
online RL post-training pipeline?

## Pipeline Structure

Online RL (GRPO, PPO, DAPO) has a serial dependency:

```
Generate completions → Score rewards → Train policy → Sync weights → Repeat
     (memory-bound)                    (compute-bound)
```

Generation dominates wall-clock time (~87% on RTX 5090, single GPU). In a
disaggregated setup, generation and training run on separate GPUs:

```
Gen GPU 0 ─┐
Gen GPU 1 ─┤
   ...     ├──→ [Rollout Buffer] ──→ [Train GPU] ──→ weight broadcast
Gen GPU N ─┘
```

N_gen generation GPUs feed 1 training GPU. The question is: what N_gen
saturates the training GPU?

## Variables

```
P = model parameters (0.6B, 1.7B, 4B, 7B, 70B)
B = batch size (prompts per micro-batch)
G = group size (completions per prompt)
S = sequence length (prompt + completion tokens)

Per GPU:
F  = peak BF16 FLOPS (H100: ~495 TFLOPS dense)
BW = peak HBM bandwidth (H100: 3.35 TB/s)
η  = MFU — fraction of peak FLOPS achieved during training (~0.85)
```

## Derivation

### Generation time (memory-bandwidth-bound)

Autoregressive decode reads the full model weights per token. For B×G
concurrent sequences decoded in parallel:

```
bytes_per_token = 2P                         (bf16 = 2 bytes per parameter)
tokens_to_generate = S                       (per sequence; all B×G decode in parallel)

T_gen = S × 2P / BW_eff
```

Where BW_eff is the effective memory bandwidth, which depends on batch size
and is always below peak BW due to kernel launch overhead, KV cache access
patterns, and scheduling gaps.

### Training time (compute-bound)

Forward + backward ≈ 6× model FLOPs per token per sequence
(2× forward, 4× backward):

```
total_flops = 6 × 2P × B × G × S = 12 × P × B × G × S

T_train = 12 × P × B × G × S / (F × η)
```

### The ratio

N_gen GPUs saturate 1 training GPU when:

```
N_gen / T_gen = 1 / T_train
N_gen = T_gen / T_train
```

Substituting:

```
N_gen = [S × 2P / BW_eff] / [12 × P × B × G × S / (F × η)]
      = [2P × F × η] / [12 × P × BW_eff × B × G]
      = F × η / (6 × BW_eff × B × G)
```

### Key insight: P and S cancel

The generation-to-training ratio is independent of model size and sequence
length. Both generation and training scale linearly with P and S, so their
ratio depends only on:

1. **Hardware**: F, BW (the GPU's compute-to-bandwidth ratio)
2. **Batch config**: B × G
3. **Effective bandwidth**: BW_eff (empirical, depends on model + GPU + framework)

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   N_gen = F × η / (6 × BW_eff × B × G)                 │
│                                                          │
│   Simplified (measurement-calibrated):                   │
│                                                          │
│   N_gen = C / (B × G)                                   │
│                                                          │
│   where C = T_gen/T_train × B×G  (measured at ref B×G)  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Effective Bandwidth

The linear model `BW_eff = BW × B×G / B_sat` is too pessimistic at small
batch sizes. Real frameworks (vLLM with PagedAttention, prefetching,
continuous batching) achieve better utilization than naive scaling predicts.

From our RTX 5090 measurements at B×G=4:
- Measured T_gen = 5695ms, T_train = 856ms
- BW_eff ≈ 2P × S / T_gen × S = 2(0.6e9)(512) / 5.695 ≈ 108 GB/s
- This is ~6% of peak 1.79 TB/s — low, but 2× better than linear model predicts

The measurement-calibrated constant C avoids modeling BW_eff explicitly:
- RTX 5090: C ≈ 27  (from 5695/856 × 4)
- H100 (estimated): C ≈ 67  (scaled by FLOPS/BW ratio)

BW_eff improves with model size (larger models amortize kernel launch
overhead over more work per layer) and batch size (wider matmuls saturate
memory controllers).

## Saturation Tables

### H100 → H100 (C ≈ 67)

| B | G | B×G | N_gen to saturate 1 training GPU |
|---|---|-----|----------------------------------|
| 1 | 4 | 4   | ~17                              |
| 1 | 8 | 8   | ~8                               |
| 4 | 4 | 16  | ~4                               |
| 4 | 8 | 32  | ~2                               |
| 8 | 8 | 64  | ~1 (crossover: gen ≈ train)      |
| 16| 8 | 128 | <1 (training becomes bottleneck) |

### RTX 5090 → RTX 5090 (C ≈ 27)

| B | G | B×G | N_gen to saturate 1 training GPU |
|---|---|-----|----------------------------------|
| 1 | 4 | 4   | ~7                               |
| 4 | 4 | 16  | ~2                               |
| 4 | 8 | 32  | ~1 (crossover)                   |
| 8 | 8 | 64  | <1 (training becomes bottleneck) |

### Crossover batch size (N_gen = 1)

The batch size where a single GPU can do both generation and training
without either being idle:

```
B×G_crossover = C

H100:      B×G ≈ 64   (e.g., B=8, G=8)
RTX 5090:  B×G ≈ 32   (e.g., B=4, G=8)
```

## Implications

### 1. Batch size is the primary optimization lever

Increasing B×G from 4 → 64 reduces the gen:train ratio from 17:1 to 1:1
on H100. This is a 17× improvement from a config change, not hardware.

### 2. Disaggregation makes sense at small batch sizes

If memory constraints force small B×G (large models, limited VRAM), then
multiple generation GPUs are needed. For Qwen3-0.6B at B×G=4, you need
~17 gen H100s per train H100 — economically terrible.

### 3. The cost-optimal strategy is maximizing batch size

Rather than scaling generation horizontally:
- Dedicate more VRAM to KV cache (increase gpu_memory_utilization)
- Use longer contexts with larger batch sizes
- Consider speculative decoding to reduce effective decode steps

### 4. Model size doesn't change the ratio

P cancels in the equation. A 70B model has the same gen:train ratio as a
0.6B model at the same B×G. (Caveat: BW_eff improves with model size
because larger models amortize per-layer overhead, so C is actually
slightly smaller for larger models — favoring them.)

### 5. Offline RL sidesteps the problem entirely

DPO, SimPO, and other offline methods pre-generate completions once,
then train without a generation loop. This eliminates the gen:train
ratio problem at the cost of on-policy freshness.

## Memory Constraints on Batch Size

The equation says "increase B×G" but batch size is bounded by KV cache memory.

### KV cache per sequence

```
KV_per_token = 2(K+V) × num_kv_heads × head_dim × 2(bf16 bytes)
KV_per_token_all_layers = KV_per_token × num_layers

Qwen3-0.6B: 2 × 8 × 128 × 2 × 28 layers = 112 KB/token
  Per sequence (S=512):  56 MB
  Per sequence (S=2048): 230 MB

Qwen3-4B: 2 × 8 × 128 × 2 × 36 layers = 144 KB/token
  Per sequence (S=512):  72 MB
  Per sequence (S=2048): 295 MB
```

### GPU memory budget

```
Total VRAM = M_vllm + M_policy + M_optimizer + M_ref + M_activations

M_vllm = M_model + M_kv_cache + overhead
       = 2P + B×G × S × KV_per_token_all_layers + ~1 GB

M_policy    = 2P                        (bf16 weights)
M_optimizer = 4P                        (fp32 AdamW m + v)
M_ref       = 2P                        (frozen bf16, optional)
M_act       ≈ 2 × B_micro × S × H      (per micro-batch, grad checkpoint)
```

vLLM's `gpu_memory_utilization` parameter splits VRAM between vLLM and
PyTorch training. This is the key constraint on batch size.

### Max B×G by configuration (Qwen3-0.6B, S=2048)

| Setup                        | VRAM for KV | Max B×G | At crossover? |
|------------------------------|-------------|---------|---------------|
| 1× 5090, util=0.3 (current) | 7.4 GB      | ~32     | ✓ (C≈27)     |
| 1× 5090, util=0.5           | 13.8 GB     | ~60     | ✓✓            |
| 2× 5090 disagg, gen=0.8     | 24.4 GB     | ~106    | ✓✓✓           |
| 1× H100, util=0.5           | 38.8 GB     | ~168    | ✓✓✓           |
| 2× H100 disagg, gen=0.8     | 62.8 GB     | ~273    | ✓✓✓           |

### Important: PagedAttention allocates on demand

vLLM uses PagedAttention — KV cache blocks are allocated as tokens are
generated, not reserved for the full `max_model_len`. For GSM8K, actual
sequences are ~300-600 tokens (not 2048), so effective KV per sequence
is ~55 MB for Qwen3-0.6B.

This means at `gpu_memory_utilization=0.3` (7.4 GB for KV cache), we can
fit ~134 concurrent sequences — far beyond any practical batch size.
**Memory is NOT the binding constraint for Qwen3-0.6B on GSM8K.**

### Key finding

On a single RTX 5090, B×G=32 (and even B×G=64) fits comfortably at
current `gpu_memory_utilization=0.3`. The crossover is at C ≈ 27.
**A single GPU can balance gen ≈ train just by increasing batch size —
no disaggregation needed for Qwen3-0.6B on GSM8K.**

Disaggregation becomes necessary for:
- Larger models (4B+) where KV cache is bigger per sequence
- Larger batch sizes needed for stable training (B×G > 32)
- Pipeline overlap (async gen + train for throughput)

### General max B×G formula

```
B×G_max = (VRAM × gpu_mem_util - M_model - overhead) / (KV_per_seq)

where KV_per_seq = 2 × num_layers × num_kv_heads × head_dim × 2 × S
```

### Full constrained equation

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   N_gen = C / min(B×G, B×G_max)                                 │
│                                                                  │
│   B×G_max = (VRAM_gen × gpu_util - M_model - overhead) / M_kv   │
│                                                                  │
│   Crossover (N_gen = 1) when B×G ≥ C AND B×G ≤ B×G_max         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Experimental Validation Plan

Sweep B×G = {4, 16, 32, 64} on a single GPU, measuring T_gen and T_train
independently. Predict the crossover point matches B×G ≈ C.

```bash
# Requires server with enough VRAM for large batch generation
for bg in 4 16 32 64; do
    python -m invokerl --config invokerl/configs/grpo_gsm8k.yaml \
        --max-steps 3 --eval-samples 0 \
        --batch-size $((bg / 4)) --group-size 4 \
        --verbose 2>&1 | grep "time="
done
```
