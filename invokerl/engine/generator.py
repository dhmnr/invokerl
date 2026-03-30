"""Generation engine: vLLM-backed completion generation + weight sync."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Disable vLLM V1 multiprocessing so the model runs in-process.
# This enables direct GPU parameter copy for weight sync (~2.8ms vs ~1600ms).
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
    gpu_memory_utilization: float = 0.5   # leave room for training
    enforce_eager: bool = False           # disable CUDA graphs for debugging
    attention_backend: str | None = None  # "FLASHINFER", "FLASH_ATTN", etc. None=auto


@dataclass
class GenerationOutput:
    """Output from a generation call."""

    token_ids: Tensor           # [B, T] int64 — prompt + completion, padded
    prompt_lens: list[int]      # length of each prompt (before completion)
    completion_lens: list[int]  # length of each completion
    log_probs: Tensor           # [B, T] float — per-token log-probs (0 for prompt tokens)
    texts: list[str]            # decoded completion strings
    prompt_mask: Tensor         # [B, T] bool
    response_mask: Tensor       # [B, T] bool
    attention_mask: Tensor      # [B, T] bool


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
        attention_backend: str | None = None,
    ):
        from vllm import LLM

        # Auto-detect: use FlashInfer on Blackwell+ (sm_120) where vLLM's
        # bundled FlashAttention2 CUDA kernels have PTX compatibility issues.
        if attention_backend is None and torch.cuda.is_available():
            cc = torch.cuda.get_device_capability()
            if cc[0] >= 12:  # Blackwell (sm_120) or newer
                attention_backend = "FLASHINFER"
                logger.info("Auto-selected FLASHINFER backend for sm_%d%d", cc[0], cc[1])

        llm_kwargs: dict = dict(
            model=model_name_or_path,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            dtype=dtype,
            seed=seed,
            max_model_len=max_model_len,
            enable_prefix_caching=True,  # reuse prompt KV cache for GRPO groups
        )
        if attention_backend is not None:
            llm_kwargs["attention_backend"] = attention_backend

        self.llm = LLM(**llm_kwargs)
        # Store dtype for weight sync casting (fp32 master weights → bf16 vLLM)
        _dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self._model_dtype = _dtype_map.get(dtype, torch.bfloat16)
        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Weight sync: "direct" (GPU copy) or "safetensors" (fallback).
        self._sync_strategy: str | None = None

        logger.info(
            "VLLMGenerator initialized: model=%s, gpu_mem=%.1f%%, dtype=%s",
            model_name_or_path,
            gpu_memory_utilization * 100,
            dtype,
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

        outputs = self.llm.generate(prompts, params)

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
            all_ids, all_lps, prompt_lens, completion_lens, texts,
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
        """Direct GPU→GPU copy into vLLM's in-process model. ~2ms for 0.6B."""
        model = self._get_model()
        vllm_params = dict(model.named_parameters())

        for name, new_tensor in state_dict.items():
            if name in vllm_params:
                vllm_params[name].data.copy_(
                    new_tensor.to(
                        dtype=self._model_dtype,
                        device=vllm_params[name].device,
                        non_blocking=True,
                    )
                )

        torch.cuda.synchronize()
        logger.debug("Weight sync: direct GPU copy complete")

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
                k: v.to(dtype=self._model_dtype).cpu().contiguous()
                for k, v in state_dict.items()
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
            max_tokens=1,         # must generate ≥1 token for vLLM
            temperature=1.0,
            prompt_logprobs=1,    # return log-probs for each prompt token
        )

        # Build token ID lists (strip padding)
        B, T = token_ids.shape
        prompt_ids_list: list[list[int]] = []
        seq_lens: list[int] = []
        for i in range(B):
            sl = int(attention_mask[i].sum().item())
            prompt_ids_list.append(token_ids[i, :sl].tolist())
            seq_lens.append(sl)

        # Batch scoring through vLLM
        outputs = self.llm.generate(
            prompt_token_ids=prompt_ids_list,
            sampling_params=params,
        )

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
                all_lps[i][:lp_len], dtype=torch.float32,
            )
            prompt_mask[i, :prompt_lens[i]] = True
            pl, cl = prompt_lens[i], completion_lens[i]
            response_mask[i, pl:pl + cl] = True
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
