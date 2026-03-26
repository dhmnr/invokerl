# Code Conventions

## CRITICAL: No PyTorch
- Zero torch dependency. Pure cuTile + CuPy + NumPy only.
- No `import torch`, no `torch.autograd`, no `torch.optim`
- Arrays are `cp.ndarray` (CuPy), not torch tensors
- Gradients are computed manually in cuTile kernels, not via autograd
- Optimizer is in `optim.py` as cuTile kernels

## Imports Order
1. Standard library (math, dataclasses, etc.)
2. Third party (cupy, numpy)
3. cuTile (`import cuda.tile as ct`)
4. Local (`from qwen3 import Qwen3Model, Qwen3Config`)

## Naming
- Kernels: `snake_case` with descriptive suffix (e.g., `dpo_loss_fwd_kernel`, `grpo_advantage_kernel`)
- Forward kernels: `*_fwd_kernel`
- Backward kernels: `*_bwd_kernel`
- Config dataclasses: `PascalCase` + `Config` (e.g., `DPOConfig`)
- Trainer classes: `PascalCase` + `Trainer` (e.g., `DPOTrainer`)
- Type alias: `ConstInt = ct.Constant[int]`

## Kernel Patterns
- Use `ct.bid(0)` for primary block index
- Tile sizes must be power-of-2, defined as ConstInt parameters
- Always validate dimensions in wrapper functions before launch
- For float hyperparameters passed as ConstInt: use `int(val * 1000)` pattern,
  document precision limitation (3 decimal places)

## Backend Selection (qwen3.py)
- `TILERL_BACKEND` env var controls compute backend:
  - `numpy` — NumPy CPU (always correct, use on sm_120/Blackwell)
  - `cupy` — CuPy GPU via cuBLAS (use on A100/H100)
  - `cutile` — cuTile kernels (fastest, has OOB bugs on non-pow2 dims)
  - empty/unset — auto: CuPy if available, else NumPy. cuTile disabled.
- `_USE_CUTILE_KERNELS` flag gates cuTile kernel usage in 4 core ops:
  `matmul()`, `rms_norm()`, `apply_rope()`, `silu_mul()`
- When cuTile disabled, these ops use CuPy GPU or NumPy CPU equivalents

## Array Convention
- All arrays are CuPy: `cp.ndarray` (or NumPy if TILERL_BACKEND=numpy)
- Default dtype: `cp.float32` (or `cp.float16`/`cp.bfloat16` where appropriate)
- Parameter dicts: `dict[str, cp.ndarray]` with dot-separated names
  (e.g., `"layers.0.attention.q_proj.weight"`)

## Documentation
- Module docstring: algorithm name, paper citation, core formula
- Each kernel gets a docstring explaining inputs/outputs/math
- Inline comments for non-obvious computations

## Testing
- Each file has `if __name__ == "__main__":` smoke test
- Compare cuTile kernel outputs against NumPy/CuPy reference implementations
- Use `cp.testing.assert_array_almost_equal()` or `np.testing.assert_allclose()`

## Dependencies
- cuda-tile (cuTile Python — GPU kernels)
- cupy-cuda13x (GPU arrays, replaces torch tensors)
- numpy (validation, CPU fallback)
- safetensors (weight loading)
