"""Distributed training utilities — FSDP wrapping and batch broadcast.

Provides helpers for multi-GPU training with FSDP (Fully Sharded Data
Parallel). Used by the disaggregated pipeline when --fsdp is enabled.

Architecture:
    torchrun --nproc_per_node=M  (launches M training ranks)
    Rank 0: gen thread (vLLM) + pipeline queue + FSDP training
    Rank 1..M-1: FSDP training only (receive batches via broadcast)
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from invokerl.algorithms.base import RolloutBatch

logger = logging.getLogger(__name__)


# -- Process group helpers ---------------------------------------------------


def init_distributed(backend: str = "nccl", device_id: int | None = None) -> int:
    """Initialize torch.distributed. Returns rank.

    Expects torchrun environment variables (RANK, WORLD_SIZE, LOCAL_RANK).

    Args:
        device_id: CUDA device to bind this rank to. If None, uses LOCAL_RANK.
                   In disagg mode with --train-device-start, pass
                   train_device_start + LOCAL_RANK so NCCL uses the correct
                   GPU (not the gen GPU).
    """
    if dist.is_initialized():
        return dist.get_rank()

    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    if device_id is None:
        device_id = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(device_id)
    logger.info(
        "Distributed init: rank=%d world=%d device=cuda:%d",
        rank, dist.get_world_size(), device_id,
    )
    return rank


def get_rank() -> int:
    """Current rank, or 0 if not distributed."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """World size, or 1 if not distributed."""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_rank() -> bool:
    """True if rank 0 or not distributed."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all ranks."""
    if dist.is_initialized():
        dist.barrier()


# -- FSDP wrapping -----------------------------------------------------------


def wrap_model_fsdp(
    model: torch.nn.Module,
    device_id: int | torch.device | None = None,
    sharding_strategy: str = "FULL_SHARD",
    mixed_precision_dtype: torch.dtype | None = torch.bfloat16,
    use_orig_params: bool = True,
    cpu_offload: bool = False,
) -> torch.nn.Module:
    """Wrap a model in FSDP1 with sensible defaults for LLM training.

    Args:
        model: The model to wrap (e.g., PolicyModel.model).
        device_id: CUDA device for this rank.
        sharding_strategy: FSDP sharding strategy. One of:
            "FULL_SHARD" — shard params + grads + optimizer (ZeRO-3)
            "SHARD_GRAD_OP" — shard grads + optimizer only (ZeRO-2)
            "NO_SHARD" — DDP-like, no sharding
        mixed_precision_dtype: dtype for compute (bf16/fp16). None = fp32.
        use_orig_params: Keep original param names (needed for optimizer).
        cpu_offload: Offload params to CPU when not in use (saves GPU memory).

    Returns:
        FSDP-wrapped model.
    """
    from torch.distributed.fsdp import (
        CPUOffload,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    # Resolve sharding strategy.
    strategy = getattr(ShardingStrategy, sharding_strategy)

    # Mixed precision: compute in bf16/fp16, reduce in fp32.
    mp = None
    if mixed_precision_dtype is not None:
        mp = MixedPrecision(
            param_dtype=mixed_precision_dtype,
            reduce_dtype=torch.float32,
            buffer_dtype=mixed_precision_dtype,
        )

    # Auto-wrap policy: wrap each transformer layer individually.
    # This gives FSDP per-layer sharding granularity.
    auto_wrap_policy = None
    try:
        # Try to find the transformer layer class (Qwen2DecoderLayer, etc.)
        layer_cls = _find_transformer_layer_class(model)
        if layer_cls is not None:
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={layer_cls},
            )
            logger.info("FSDP auto-wrap policy: %s", layer_cls.__name__)
    except Exception as e:
        logger.warning("Could not detect transformer layer class: %s", e)

    offload = CPUOffload(offload_params=True) if cpu_offload else None

    wrapped = FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=mp,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device_id,
        use_orig_params=use_orig_params,
        cpu_offload=offload,
    )

    logger.info(
        "FSDP wrapped: strategy=%s, mp=%s, device=%s",
        sharding_strategy, mixed_precision_dtype, device_id,
    )
    return wrapped


def _find_transformer_layer_class(model: torch.nn.Module) -> type | None:
    """Detect the transformer decoder layer class for FSDP auto-wrapping.

    Looks for common HuggingFace layer class names (Qwen2DecoderLayer,
    LlamaDecoderLayer, etc.) in the model's module tree.
    """
    # Common layer class names from HuggingFace transformers.
    known_names = {
        "Qwen2DecoderLayer", "Qwen3DecoderLayer",
        "LlamaDecoderLayer", "MistralDecoderLayer",
        "Phi3DecoderLayer", "GemmaDecoderLayer",
    }
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name in known_names:
            return type(module)
    return None


def get_full_state_dict(
    model: torch.nn.Module,
    rank0_only: bool = True,
    offload_to_cpu: bool = True,
) -> dict[str, Tensor]:
    """Extract the full (unsharded) state dict from an FSDP model.

    Only rank 0 gets the full dict when rank0_only=True (other ranks
    get an empty dict). This is used for weight sync to vLLM.

    Args:
        model: FSDP-wrapped model.
        rank0_only: If True, only rank 0 returns the full state dict.
        offload_to_cpu: Move tensors to CPU (avoids GPU memory spike).

    Returns:
        Full state dict on rank 0, empty dict on other ranks.
    """
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        StateDictType,
    )

    cfg = FullStateDictConfig(
        offload_to_cpu=offload_to_cpu,
        rank0_only=rank0_only,
    )
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        return model.state_dict()


# -- RolloutBatch broadcast --------------------------------------------------

# Tensor fields in RolloutBatch that need broadcasting.
_BATCH_TENSOR_FIELDS = [
    "token_ids", "prompt_mask", "response_mask", "attention_mask",
    "rewards", "old_log_probs", "ref_log_probs",
]
_BATCH_OPTIONAL_TENSOR_FIELDS = ["token_rewards", "group_ids"]


def broadcast_batch(
    batch: RolloutBatch | None,
    src: int = 0,
    device: torch.device | str = "cuda",
) -> RolloutBatch:
    """Broadcast a RolloutBatch from src rank to all ranks.

    Rank `src` must have a valid batch. Other ranks pass None.
    Broadcasts tensor shapes as metadata first, then the tensors themselves.

    Args:
        batch: The batch on src rank, None on other ranks.
        device: Target device for tensors on non-src ranks.

    Returns:
        RolloutBatch on all ranks.
    """
    rank = get_rank()

    # Step 1: Broadcast scalar metadata (group_size, weight_version,
    # tensor shapes/dtypes, which optional fields are present).
    if rank == src:
        assert batch is not None, "src rank must provide a batch"
        metadata = _pack_batch_metadata(batch)
    else:
        metadata = None

    metadata = _broadcast_object(metadata, src=src)

    # Step 2: Broadcast tensors.
    tensors: dict[str, Tensor] = {}

    for name in _BATCH_TENSOR_FIELDS:
        shape, dtype = metadata["shapes"][name]
        if rank == src:
            t = getattr(batch, name).to(device).contiguous()
        else:
            t = torch.empty(shape, dtype=dtype, device=device)
        dist.broadcast(t, src=src)
        tensors[name] = t

    for name in _BATCH_OPTIONAL_TENSOR_FIELDS:
        if name not in metadata["shapes"]:
            tensors[name] = None
            continue
        shape, dtype = metadata["shapes"][name]
        if rank == src:
            t = getattr(batch, name).to(device).contiguous()
        else:
            t = torch.empty(shape, dtype=dtype, device=device)
        dist.broadcast(t, src=src)
        tensors[name] = t

    return RolloutBatch(
        token_ids=tensors["token_ids"],
        prompt_mask=tensors["prompt_mask"],
        response_mask=tensors["response_mask"],
        attention_mask=tensors["attention_mask"],
        rewards=tensors["rewards"],
        token_rewards=tensors.get("token_rewards"),
        old_log_probs=tensors["old_log_probs"],
        ref_log_probs=tensors["ref_log_probs"],
        group_ids=tensors.get("group_ids"),
        group_size=metadata["group_size"],
        weight_version=metadata["weight_version"],
        extras=metadata.get("extras", {}),
    )


def _pack_batch_metadata(batch: RolloutBatch) -> dict[str, Any]:
    """Extract shapes, dtypes, and scalar fields from a batch."""
    shapes: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}

    for name in _BATCH_TENSOR_FIELDS:
        t = getattr(batch, name)
        shapes[name] = (tuple(t.shape), t.dtype)

    for name in _BATCH_OPTIONAL_TENSOR_FIELDS:
        t = getattr(batch, name)
        if t is not None:
            shapes[name] = (tuple(t.shape), t.dtype)

    return {
        "shapes": shapes,
        "group_size": batch.group_size,
        "weight_version": batch.weight_version,
        "extras": {},  # Don't broadcast extras (value preds etc.) — recomputed per rank
    }


def _broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a Python object from src to all ranks using dist.broadcast_object_list."""
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]
