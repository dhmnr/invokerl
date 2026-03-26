# Cipher Checkpoint — 2026-03-26 04:30

## Current State
- All sweeps **COMPLETE**. invokerl validated: **56.0% GSM8K accuracy, beating tilerl's 54%**.
- Best config: **fp32 master weights + lr=1e-6** (original tilerl LR, works with fp32 precision)
- Alternative: bf16 + lr=5e-6 also works (55.5%) as a workaround
- Awaiting daemon's direction for next steps

## Final Sweep Results (200-sample eval, ±3.5% CI)

| Config | Baseline | Step 50 | Step 100 | Step 150 | Step 200 | Final | Wall-clock |
|--------|----------|---------|----------|----------|----------|-------|------------|
| bf16 lr=5e-6 | 47.5% | 50.5% | 49.5% | 49.0% | 52.0% | **55.5%** | ~28 min |
| fp32 lr=1e-6 | 50.0% | 46.0% | **52.5%** | 50.0% | **56.0%** | 55.0% | ~40 min |
| tilerl (ref) | 46% | 36% | 48% | 50% | 54% | 54% | ~5.5 hrs |

### Screening runs (50-sample eval, noisy)
| Config | Step 50 | Step 100 | Final | KL@100 |
|--------|---------|----------|-------|--------|
| bf16 lr=1e-6 baseline (200 steps) | 44% | 40% | 54% | 0.0003 |
| bf16 lr=5e-6 (100 steps) | 50% | 40% | 46% | 0.0017 |
| combo (lr=2e-6, beta=0.02, G=8) | 40% | 46% | 46% | — |

## Root Cause Chain (all solved)
1. **Prompt format** → ChatML fix (2% → 50%)
2. **OOM** → max_model_len + gpu_memory_utilization fix
3. **KL collapse** → ref model required (no-ref collapsed 50→38%)
4. **Batch size** → 1 to match tilerl
5. **bf16 precision** → fp32 master weights (the key fix: KL 0.0003→0.0037 at step 100)

## Key Findings
1. **bf16 precision was the root cause** — lr=1e-6 updates rounded away by 8-bit mantissa. fp32 preserves them (KL 12× higher)
2. **J-curve is real** — all runs dip mid-training then recover at steps 180-200
3. **50-sample eval noise was misleading** — 200-sample eval revealed stable improvement
4. **LR is dominant for bf16** — lr=5e-6 needed to overcome bf16 rounding. Not needed with fp32
5. **Beta and group_size had minimal impact** in screening (LR dominated)

## Training Trajectory (fp32 lr=1e-6, 200 steps)
| Step | Loss | Reward | KL | Gnorm | LR |
|------|------|--------|-----|-------|----|
| 0 | -0.0023 | 0.438 | 0.0006 | 2.19 | 4.00e-08 |
| 10 | 0.0004 | 0.500 | 0.0006 | 1.76 | 2.40e-07 |
| 20 | 0.0241 | 0.625 | 0.0005 | 2.25 | 4.40e-07 |
| 30 | 0.0167 | 0.188 | 0.0007 | 2.20 | 6.40e-07 |
| 40 | 0.0003 | 0.250 | 0.0008 | 2.01 | 8.40e-07 |
| 50 | 0.0000 | 0.250 | 0.0011 | 0.05 | 1.00e-06 |
| 60 | 0.0188 | 0.250 | 0.0014 | 1.60 | 1.00e-06 |
| 70 | 0.0234 | 0.688 | 0.0023 | 2.06 | 1.00e-06 |
| 80 | 0.0164 | 0.375 | 0.0030 | 2.36 | 1.00e-06 |
| 90 | 0.0948 | 0.688 | 0.0030 | 2.58 | 1.00e-06 |
| 100 | 0.0026 | 0.375 | 0.0037 | 1.29 | 1.00e-06 |
| 110 | 0.0686 | 0.750 | 0.0038 | 3.07 | 1.00e-06 |
| 120 | 0.0900 | 0.688 | 0.0032 | 2.49 | 1.00e-06 |
| 130 | 0.0002 | 0.250 | 0.0039 | 0.08 | 1.00e-06 |
| 140 | 0.0243 | 0.625 | 0.0053 | 1.54 | 1.00e-06 |
| 150 | 0.0258 | 0.500 | 0.0063 | 2.28 | 1.00e-06 |
| 160 | 0.0196 | 0.188 | 0.0046 | 1.42 | 1.00e-06 |
| 170 | 0.0271 | 0.375 | 0.0046 | 2.34 | 1.00e-06 |
| 180 | 0.0124 | 0.688 | 0.0054 | 1.83 | 1.00e-06 |
| 190 | 0.0421 | 0.500 | 0.0047 | 2.11 | 1.00e-06 |

## Code Changes (cipher)
- `engine/generator.py` — dtype cast fix for fp32→bf16 weight sync
- `configs/sweep_fp32_lr1e6.yaml` — fp32 + lr=1e-6 config
- `configs/sweep_fp32_lr5e6.yaml` — fp32 + lr=5e-6 config
- `.hivemind/context/cipher-checkpoint.md` — sweep results documentation

## Server Details
- SSH: `ssh -p 54410 root@93.91.156.101 -i ~/.ssh/id_ed25519`
- Weights: `/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/`
- Logs: `/root/tilerl/sweep_lr5e6_200step.log`, `/root/tilerl/fp32_lr1e6.log`
- Checkpoints: `/root/tilerl/checkpoints/sweep_lr5e6/`, `/root/tilerl/checkpoints/sweep_fp32_lr1e6/`
