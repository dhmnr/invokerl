"""Compare model.forward() vs manual forward at T=9 to find discrepancy."""
import cupy as cp
import numpy as np
from qwen3 import Qwen3Config, Qwen3Model

cfg = Qwen3Config.qwen3_0_6b()
model = Qwen3Model(cfg)
model.load_weights(
    "/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    "c1899de289a04d12100db370d81485cdf75e47ca/"
)

# Test at T=9
T = 9
ids = cp.arange(1, T + 1, dtype=cp.int32).reshape(1, -1)

print("Running model.forward()...")
logits = model.forward(ids)
has_nan = bool(cp.isnan(logits).any())
print("  NaN: %s, shape: %s" % (has_nan, logits.shape))

if has_nan:
    # Check the cache to find where NaN appeared
    cache = model._cache
    print("  Checking cache...")
    for key, val in cache.items():
        if key == "layers":
            for i, lc in enumerate(val):
                for k2, v2 in lc.items():
                    if isinstance(v2, cp.ndarray) and bool(cp.isnan(v2).any()):
                        print("    Layer %d, %s: NaN! shape=%s" % (i, k2, v2.shape))
        elif isinstance(val, cp.ndarray) and bool(cp.isnan(val).any()):
            print("    %s: NaN!" % key)

# Also test at T=5 (should work)
print("\nRunning model.forward() at T=5...")
ids5 = cp.arange(1, 6, dtype=cp.int32).reshape(1, -1)
logits5 = model.forward(ids5)
print("  NaN: %s" % bool(cp.isnan(logits5).any()))

# Test T=8
print("\nRunning model.forward() at T=8...")
ids8 = cp.arange(1, 9, dtype=cp.int32).reshape(1, -1)
logits8 = model.forward(ids8)
print("  NaN: %s" % bool(cp.isnan(logits8).any()))
