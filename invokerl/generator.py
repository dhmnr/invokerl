"""Generation engine: vLLM-backed completion generation + weight sync."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

# V1 multiprocessing mode: "0" = in-process (fast weight sync ~2.8ms),
# "1" = separate process (safetensors sync ~1.6s but isolated NCCL).
# Default to in-process for single-GPU/disagg-2GPU setups. When used with
# FSDP (torchrun), the caller should set VLLM_ENABLE_V1_MULTIPROCESSING=1
# because in-process mode conflicts with torchrun's NCCL process group.
# Must be set before importing vllm.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Controls generation behavior."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 1.0
    stop_strings: list[str] | None = None
    # vLLM-specific
    gpu_memory_utilization: float = 0.5  # leave room for training
    enforce_eager: bool = False  # disable CUDA graphs for debugging


@dataclass
class GenerationOutput:
    """Output from a generation call."""

    token_ids: Tensor  # [B, T] int64 — prompt + completion, padded
    prompt_lens: list[int]  # length of each prompt (before completion)
    completion_lens: list[int]  # length of each completion
    log_probs: Tensor  # [B, T] float — per-token log-probs (0 for prompt tokens)
    texts: list[str]  # decoded completion strings
    prompt_mask: Tensor  # [B, T] bool
    response_mask: Tensor  # [B, T] bool
    attention_mask: Tensor  # [B, T] bool


class BaseGenerator(ABC):
    """Abstract generation engine."""

    @abstractmethod
    def generate(self, prompts: list[str], config: GenerationConfig) -> GenerationOutput:
        """Generate completions for a batch of prompts."""
        ...

    @abstractmethod
    def update_weights(self, state_dict: dict[str, Tensor]) -> None:
        """Sync updated training weights into the generation engine."""
        ...

    @abstractmethod
    def compute_log_probs(self, token_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Compute per-token log-probs for given sequences (no generation)."""
        ...


class VLLMGenerator(BaseGenerator):
    """vLLM-backed generation with PagedAttention, log-prob extraction, and weight sync."""

    def __init__(
        self,
        model_name_or_path: str,
        gpu_memory_utilization: float = 0.5,
        enforce_eager: bool = False,
        dtype: str = "bfloat16",
        seed: int = 42,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
    ):
        from vllm import LLM

        self.tensor_parallel_size = tensor_parallel_size

        self.llm = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            dtype=dtype,
            seed=seed,
            max_model_len=max_model_len,
            enable_prefix_caching=True,  # reuse prompt KV cache for GRPO groups
            tensor_parallel_size=tensor_parallel_size,
        )
        # Store dtype for weight sync casting (fp32 master weights → bf16 vLLM)
        _dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self._model_dtype = _dtype_map.get(dtype, torch.bfloat16)
        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Weight sync: "direct" (GPU copy) or "safetensors" (fallback).
        # With TP > 1, direct GPU copy may not work (workers are separate
        # processes), so safetensors is the default for TP > 1.
        self._sync_strategy: str | None = None
        if tensor_parallel_size > 1:
            self._sync_strategy = "safetensors"
            logger.info(
                "TP=%d: using safetensors weight sync (workers are separate processes)",
                tensor_parallel_size,
            )

        logger.info(
            "VLLMGenerator initialized: model=%s, gpu_mem=%.1f%%, dtype=%s, tp=%d",
            model_name_or_path,
            gpu_memory_utilization * 100,
            dtype,
            tensor_parallel_size,
        )

    def generate(
        self,
        prompts: list[str],
        config: GenerationConfig,
    ) -> GenerationOutput:
        """Generate completions and return padded token_ids, log_probs, masks."""
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature if config.temperature > 1e-7 else 0.0,
            top_k=config.top_k if config.top_k > 0 else -1,
            top_p=config.top_p,
            logprobs=1,  # return log-prob for at least the sampled token
            stop=config.stop_strings or [],
        )

        outputs = self.llm.generate(prompts, params, use_tqdm=False)

        # Collect per-sequence data
        all_ids: list[list[int]] = []
        all_lps: list[list[float]] = []
        prompt_lens: list[int] = []
        completion_lens: list[int] = []
        texts: list[str] = []

        for out in outputs:
            p_ids = list(out.prompt_token_ids)
            comp = out.outputs[0]  # one completion per prompt
            c_ids = list(comp.token_ids)

            prompt_lens.append(len(p_ids))
            completion_lens.append(len(c_ids))
            texts.append(comp.text)
            all_ids.append(p_ids + c_ids)

            # Log-probs: 0 for prompt tokens, actual for generated tokens.
            # log_probs[t] = log P(token[t] | token[:t]).
            # Prompt tokens have no old-policy log-prob (weren't sampled).
            lps = [0.0] * len(p_ids)
            if comp.logprobs:
                for j, step_dict in enumerate(comp.logprobs):
                    if step_dict and j < len(c_ids):
                        tok = c_ids[j]
                        if tok in step_dict:
                            lps.append(step_dict[tok].logprob)
                        else:
                            # Sampled token not in top-k logprobs — take best
                            # available as approximation (rare with logprobs≥1)
                            lps.append(next(iter(step_dict.values())).logprob)
                    else:
                        lps.append(0.0)
            else:
                lps.extend([0.0] * len(c_ids))
            all_lps.append(lps)

        return self._pack_output(
            all_ids,
            all_lps,
            prompt_lens,
            completion_lens,
            texts,
        )

    def update_weights(self, state_dict: dict[str, Tensor]) -> None:
        """Sync training weights into vLLM. Direct GPU copy (~2ms) or safetensors fallback."""
        if self._sync_strategy is None:
            self._sync_strategy = self._detect_sync_strategy()

        if self._sync_strategy == "direct":
            self._sync_weights_direct(state_dict)
        else:
            self._sync_weights_safetensors(state_dict)

    def get_model_params(self) -> dict[str, Tensor]:
        """Return vLLM's model parameters for shared-weight mode."""
        model = self._get_model()
        return dict(model.named_parameters())

    # -- Weight sync strategies ------------------------------------------------

    def _detect_sync_strategy(self) -> str:
        """Detect fastest weight sync path: direct GPU copy or safetensors fallback."""
        try:
            model = self._get_model()
            if model is not None:
                logger.info("Weight sync: direct GPU parameter copy")
                return "direct"
        except (RuntimeError, AttributeError):
            pass

        backend = "tmpfs (/dev/shm)" if os.path.isdir("/dev/shm") else "disk"
        logger.info("Weight sync: safetensors via %s", backend)
        return "safetensors"

    def _sync_weights_direct(self, state_dict: dict[str, Tensor]) -> None:
        """Direct GPU→GPU copy into vLLM's in-process model. ~100ms for 0.6B.

        Handles vLLM's fused layers: qkv_proj (from q/k/v_proj) and
        gate_up_proj (from gate/up_proj). For fused layers, copies each
        component directly into the correct slice of the existing vLLM
        parameter buffer — no torch.cat, no GPU allocation, no CUDA graph
        pool conflict.
        """
        model = self._get_model()
        vllm_params = dict(model.named_parameters())
        synced = 0

        # Ensure any in-flight CUDA graph replays on the gen device complete
        # before we write to vLLM's parameter buffers.
        gen_device = next(iter(vllm_params.values())).device
        torch.cuda.synchronize(gen_device)

        for vname, vparam in vllm_params.items():
            synced += self._copy_policy_to_vllm(vname, vparam, state_dict)

        torch.cuda.synchronize(gen_device)
        logger.debug(
            "Weight sync: direct GPU copy complete (%d/%d params)", synced, len(vllm_params)
        )

    def _copy_policy_to_vllm(
        self,
        vllm_name: str,
        vparam: Tensor,
        state_dict: dict[str, Tensor],
    ) -> int:
        """Copy policy tensor(s) into a single vLLM parameter.

        For fused layers (qkv_proj, gate_up_proj), copies each component
        directly into the correct slice of the vLLM param buffer. This avoids
        torch.cat allocations that conflict with vLLM's CUDA graph memory pool.

        Returns 1 if the param was synced, 0 otherwise.
        """
        dtype = self._model_dtype
        device = vparam.device

        # Direct match (norms, o_proj, down_proj, embeddings, etc.)
        if vllm_name in state_dict:
            vparam.data.copy_(
                state_dict[vllm_name].to(dtype=dtype, device=device, non_blocking=True)
            )
            return 1

        # Fused QKV: copy q, k, v into slices of qkv_proj
        if ".qkv_proj." in vllm_name:
            q_name = vllm_name.replace(".qkv_proj.", ".q_proj.")
            k_name = vllm_name.replace(".qkv_proj.", ".k_proj.")
            v_name = vllm_name.replace(".qkv_proj.", ".v_proj.")
            if q_name in state_dict and k_name in state_dict and v_name in state_dict:
                q, k, v = state_dict[q_name], state_dict[k_name], state_dict[v_name]
                q_size, k_size = q.shape[0], k.shape[0]
                vparam.data[:q_size].copy_(q.to(dtype=dtype, device=device, non_blocking=True))
                vparam.data[q_size : q_size + k_size].copy_(
                    k.to(dtype=dtype, device=device, non_blocking=True)
                )
                vparam.data[q_size + k_size :].copy_(
                    v.to(dtype=dtype, device=device, non_blocking=True)
                )
                return 1
            return 0

        # Fused gate_up: copy gate, up into slices of gate_up_proj
        if ".gate_up_proj." in vllm_name:
            gate_name = vllm_name.replace(".gate_up_proj.", ".gate_proj.")
            up_name = vllm_name.replace(".gate_up_proj.", ".up_proj.")
            if gate_name in state_dict and up_name in state_dict:
                gate, up = state_dict[gate_name], state_dict[up_name]
                gate_size = gate.shape[0]
                vparam.data[:gate_size].copy_(
                    gate.to(dtype=dtype, device=device, non_blocking=True)
                )
                vparam.data[gate_size:].copy_(up.to(dtype=dtype, device=device, non_blocking=True))
                return 1
            return 0

        return 0

    def _sync_weights_safetensors(self, state_dict: dict[str, Tensor]) -> None:
        """Fallback: serialize to safetensors via tmpfs or disk."""
        import shutil
        import tempfile

        try:
            from safetensors.torch import save_file
        except ImportError:
            logger.warning("safetensors not installed, skipping weight sync")
            return

        shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
        tmp_dir = tempfile.mkdtemp(prefix="invokerl_ws_", dir=shm_dir)
        try:
            cpu_state = {
                k: v.to(dtype=self._model_dtype).cpu().contiguous() for k, v in state_dict.items()
            }
            save_file(cpu_state, os.path.join(tmp_dir, "model.safetensors"))

            self.llm.llm_engine.collective_rpc(
                "reload_weights",
                kwargs={
                    "weights_path": tmp_dir,
                    "is_checkpoint_format": True,
                },
            )
            backend = "tmpfs" if shm_dir else "disk"
            logger.debug("Weight sync: safetensors via %s", backend)
        except Exception as e:
            logger.warning("Weight sync failed: %s. Generation uses stale weights.", e)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def compute_log_probs(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Compute per-token log P(token[t] | token[:t]) for existing sequences."""
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=1,  # must generate ≥1 token for vLLM
            temperature=1.0,
            prompt_logprobs=1,  # return log-probs for each prompt token
        )

        # Build token ID lists (strip padding)
        B, T = token_ids.shape
        prompt_ids_list: list[list[int]] = []
        seq_lens: list[int] = []
        for i in range(B):
            sl = int(attention_mask[i].sum().item())
            prompt_ids_list.append(token_ids[i, :sl].tolist())
            seq_lens.append(sl)

        # Batch scoring through vLLM.
        # Pass token ID lists as positional arg — vLLM ≥0.18 accepts
        # Sequence[list[int]] as the `prompts` parameter.
        outputs = self.llm.generate(prompt_ids_list, params, use_tqdm=False)

        # Extract log-probs from prompt_logprobs
        result = torch.zeros(B, T, dtype=torch.float32)
        for i, out in enumerate(outputs):
            plps = out.prompt_logprobs
            if plps is None:
                continue
            ids = prompt_ids_list[i]
            # Position 0 is None (no predecessor). Start from 1.
            for t in range(1, min(len(plps), len(ids))):
                if plps[t] is not None:
                    tok = ids[t]
                    if tok in plps[t]:
                        result[i, t] = plps[t][tok].logprob

        return result

    # -- Internal helpers ------------------------------------------------------

    def _get_model(self):
        """Access vLLM's underlying nn.Module (path varies by vLLM version)."""
        engine = self.llm.llm_engine
        executor = engine.model_executor

        # vLLM 0.6.x+: driver_worker path (single GPU and TP leader)
        if hasattr(executor, "driver_worker"):
            worker = executor.driver_worker
            if hasattr(worker, "model_runner"):
                runner = worker.model_runner
                if hasattr(runner, "model"):
                    return runner.model

        # vLLM 0.5.x: direct model on executor
        if hasattr(executor, "model"):
            return executor.model

        # vLLM with worker wrapper
        if hasattr(executor, "workers") and len(executor.workers) > 0:
            worker = executor.workers[0]
            if hasattr(worker, "model_runner") and hasattr(worker.model_runner, "model"):
                return worker.model_runner.model

        raise RuntimeError(
            "Cannot access vLLM model for weight sync. "
            "Check vLLM version compatibility (tested with vLLM ≥0.6)."
        )

    def _pack_output(
        self,
        all_ids: list[list[int]],
        all_lps: list[list[float]],
        prompt_lens: list[int],
        completion_lens: list[int],
        texts: list[str],
    ) -> GenerationOutput:
        """Pad variable-length outputs into fixed-size tensors."""
        B = len(all_ids)
        max_len = max(len(ids) for ids in all_ids) if all_ids else 0
        pad_id = self.tokenizer.pad_token_id

        token_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        log_probs = torch.zeros(B, max_len, dtype=torch.float32)
        prompt_mask = torch.zeros(B, max_len, dtype=torch.bool)
        response_mask = torch.zeros(B, max_len, dtype=torch.bool)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)

        for i in range(B):
            seq_len = len(all_ids[i])
            token_ids[i, :seq_len] = torch.tensor(all_ids[i], dtype=torch.long)
            lp_len = min(len(all_lps[i]), max_len)
            log_probs[i, :lp_len] = torch.tensor(
                all_lps[i][:lp_len],
                dtype=torch.float32,
            )
            prompt_mask[i, : prompt_lens[i]] = True
            pl, cl = prompt_lens[i], completion_lens[i]
            response_mask[i, pl : pl + cl] = True
            attention_mask[i, :seq_len] = True

        return GenerationOutput(
            token_ids=token_ids,
            prompt_lens=prompt_lens,
            completion_lens=completion_lens,
            log_probs=log_probs,
            texts=texts,
            prompt_mask=prompt_mask,
            response_mask=response_mask,
            attention_mask=attention_mask,
        )
