"""Generation engine interface and vLLM implementation.

The generator handles all completion generation. Algorithms never call this
directly — the trainer orchestrates it.

Key features:
- Batched generation with vLLM (PagedAttention, continuous batching)
- Returns per-token log-probs alongside completions (for old_log_probs)
- Weight update support (sync training weights → vLLM engine)
"""

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
    def generate(
        self,
        prompts: list[str],
        config: GenerationConfig,
    ) -> GenerationOutput:
        """Generate completions for a batch of prompts.

        Args:
            prompts: List of formatted prompt strings.
            config: Generation hyperparameters.

        Returns:
            GenerationOutput with token_ids, log_probs, and decoded texts.
        """
        ...

    @abstractmethod
    def update_weights(self, state_dict: dict[str, Tensor]) -> None:
        """Sync updated training weights into the generation engine.

        Called by the trainer after each optimizer step.

        Args:
            state_dict: Model state dict with updated parameter tensors.
        """
        ...

    @abstractmethod
    def compute_log_probs(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Compute per-token log-probs for given sequences without generating.

        Used for reference model log-prob computation when the reference
        is served by a separate engine instance.

        Args:
            token_ids: [B, T] token IDs.
            attention_mask: [B, T] attention mask.

        Returns:
            log_probs: [B, T] per-token log-probabilities.
        """
        ...


# ---------------------------------------------------------------------------
# vLLM implementation
# ---------------------------------------------------------------------------


class VLLMGenerator(BaseGenerator):
    """vLLM-backed generation engine.

    Uses vllm.LLM for in-process generation (no server overhead).
    Supports:
    - Batched generation with PagedAttention
    - Per-token log-prob extraction
    - Weight updates via direct parameter copy (single GPU)
    - Prefix caching for GRPO-style repeated prompts
    """

    def __init__(
        self,
        model_name_or_path: str,
        gpu_memory_utilization: float = 0.5,
        enforce_eager: bool = False,
        dtype: str = "bfloat16",
        seed: int = 42,
        max_model_len: int | None = None,
    ):
        """Initialize vLLM engine.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            gpu_memory_utilization: Fraction of GPU memory for vLLM (rest for training).
            enforce_eager: Disable CUDA graphs for debugging.
            dtype: Model dtype ("bfloat16", "float16", "float32").
            seed: Random seed for reproducibility.
            max_model_len: Maximum sequence length. None = use model config.
        """
        from vllm import LLM

        self.llm = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            dtype=dtype,
            seed=seed,
            max_model_len=max_model_len,
            enable_prefix_caching=True,  # reuse prompt KV cache for GRPO groups
        )
        # Store dtype for weight sync casting (fp32 master weights → bf16 vLLM)
        _dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self._model_dtype = _dtype_map.get(dtype, torch.bfloat16)
        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Weight sync strategy — detected lazily on first update_weights() call.
        # Possible values: "direct", "ipc", "safetensors"
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
        """Generate completions for a batch of prompts.

        For GRPO-style group generation, the trainer repeats each prompt
        group_size times. Prefix caching ensures the prompt KV cache is
        computed only once per unique prompt.

        Args:
            prompts: List of prompt strings (may contain duplicates for groups).
            config: Generation hyperparameters.

        Returns:
            GenerationOutput with padded token_ids, per-token log_probs, masks.
        """
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
        """Zero-copy weight sync: direct GPU→GPU parameter copy.

        On first call, probes the vLLM engine to find the fastest sync path:
        1. Direct parameter copy (in-process engine)  — ~2ms, GPU memcpy only
        2. CUDA IPC handles (subprocess engine)        — ~5ms, shared GPU memory
        3. safetensors via tmpfs /dev/shm (fallback)   — ~400ms, RAM round-trip
        4. safetensors via disk (last resort)           — ~1600ms

        Args:
            state_dict: HuggingFace model state dict from PolicyModel.
        """
        if self._sync_strategy is None:
            self._sync_strategy = self._detect_sync_strategy()

        if self._sync_strategy == "direct":
            self._sync_weights_direct(state_dict)
        elif self._sync_strategy == "ipc":
            self._sync_weights_ipc(state_dict)
        else:
            self._sync_weights_safetensors(state_dict)

    def get_model_params(self) -> dict[str, Tensor]:
        """Return vLLM's model parameters as a {name: tensor} dict.

        Used for shared-weight mode: PolicyModel's parameters are set to
        views into these tensors so both models share the same GPU memory.
        Only works with in-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0).
        """
        model = self._get_model()
        return dict(model.named_parameters())

    # -- Weight sync strategies ------------------------------------------------

    def _detect_sync_strategy(self) -> str:
        """Probe vLLM engine to find the fastest available sync method."""
        # Strategy 1: direct model access (same-process engine).
        # Works with vLLM GPUExecutor (single GPU, no multiprocessing).
        try:
            model = self._get_model()
            if model is not None:
                logger.info("Weight sync strategy: direct GPU parameter copy")
                return "direct"
        except (RuntimeError, AttributeError):
            pass

        # Strategy 2: CUDA IPC via collective_rpc.
        # Works with vLLM MultiprocessingGPUExecutor (v1 subprocess engine).
        # We register a custom method on the worker to accept IPC handles.
        try:
            if self._try_register_ipc_handler():
                logger.info("Weight sync strategy: CUDA IPC (cross-process)")
                return "ipc"
        except Exception as e:
            logger.debug("CUDA IPC setup failed: %s", e)

        # Strategy 3: safetensors serialization (tmpfs preferred over disk).
        backend = "tmpfs (/dev/shm)" if os.path.isdir("/dev/shm") else "disk"
        logger.info("Weight sync strategy: safetensors via %s", backend)
        return "safetensors"

    def _sync_weights_direct(self, state_dict: dict[str, Tensor]) -> None:
        """Direct GPU→GPU copy into vLLM's in-process model parameters.

        No serialization, no CPU, no disk. Just torch.Tensor.copy_() on the
        same CUDA device. ~2ms for a 0.6B model at 1.8 TB/s memory bandwidth.
        """
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

    def _try_register_ipc_handler(self) -> bool:
        """Register a CUDA IPC weight-loading method on vLLM workers.

        Adds _load_weights_ipc() to the Worker class so we can call it via
        collective_rpc. Returns True if registration succeeded.
        """
        engine = self.llm.llm_engine
        executor = engine.model_executor

        # Access the worker class to monkey-patch it.
        # For multiprocessing executors, workers are proxies — we need to
        # send the method definition via collective_rpc("execute_method") or
        # patch the class before workers are fully initialized.
        worker_cls = None
        if hasattr(executor, "driver_worker"):
            worker_cls = type(executor.driver_worker)
        elif hasattr(executor, "workers") and executor.workers:
            worker_cls = type(executor.workers[0])

        if worker_cls is None:
            return False

        # Check if already patched
        if hasattr(worker_cls, "_load_weights_ipc"):
            return True

        def _load_weights_ipc(worker_self, ipc_handles: dict) -> None:
            """Load model weights from CUDA IPC handles (zero-copy GPU access).

            Each handle maps to the same physical GPU memory allocated by the
            training process. We copy from shared memory into model params.
            """
            import torch as _torch

            runner = worker_self.model_runner
            model = runner.model
            param_dict = dict(model.named_parameters())

            for name, handle_info in ipc_handles.items():
                if name not in param_dict:
                    continue
                # Reconstruct storage from IPC handle — maps same GPU memory
                storage = _torch.cuda.UntypedStorage._new_shared_cuda(
                    handle_info["device"],
                    handle_info["handle"],
                    handle_info["nbytes"],
                )
                # Build tensor view over the shared storage
                src = _torch.empty(
                    handle_info["shape"],
                    dtype=handle_info["dtype"],
                    device=f"cuda:{handle_info['device']}",
                )
                src.set_(storage, storage_offset=0, size=handle_info["shape"])
                param_dict[name].data.copy_(src)

            _torch.cuda.synchronize()

        worker_cls._load_weights_ipc = _load_weights_ipc
        logger.debug("Registered _load_weights_ipc on %s", worker_cls.__name__)
        return True

    def _sync_weights_ipc(self, state_dict: dict[str, Tensor]) -> None:
        """Sync weights via CUDA IPC handles — zero-copy cross-process.

        Exports IPC handles (tiny metadata ~100 bytes each) for GPU tensors,
        sends them to the vLLM worker via collective_rpc. The worker
        reconstructs tensors pointing to the same physical GPU memory and
        copies into its model parameters. ~5ms total.
        """
        # Ensure tensors are on GPU in the right dtype, contiguous
        gpu_state = {}
        for name, tensor in state_dict.items():
            gpu_state[name] = tensor.to(
                dtype=self._model_dtype, device="cuda"
            ).contiguous()

        torch.cuda.synchronize()  # ensure writes are visible

        # Export IPC handles — small metadata, not the actual tensor data
        ipc_handles = {}
        for name, tensor in gpu_state.items():
            storage = tensor.untyped_storage()
            handle = storage._share_cuda_()
            ipc_handles[name] = {
                "device": handle[0],
                "handle": handle[1],
                "nbytes": handle[2],
                "shape": tuple(tensor.shape),
                "dtype": tensor.dtype,
            }

        # Send handles to worker — worker reconstructs tensors from same GPU memory
        self.llm.llm_engine.collective_rpc(
            "_load_weights_ipc",
            kwargs={"ipc_handles": ipc_handles},
        )

        # Keep gpu_state alive until worker has copied (prevent deallocation)
        del gpu_state
        logger.debug("Weight sync: CUDA IPC complete")

    def _sync_weights_safetensors(self, state_dict: dict[str, Tensor]) -> None:
        """Fallback: serialize to safetensors via tmpfs or disk.

        Uses /dev/shm (RAM-backed tmpfs) when available to avoid disk I/O.
        ~400ms via tmpfs, ~1600ms via disk.
        """
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
        """Compute per-token log-probs for existing sequences.

        Uses vLLM's prompt_logprobs feature: feed the full sequence as a
        "prompt" and extract the conditional log-probabilities.

        Convention: log_probs[t] = log P(token[t] | token[:t]).
        Position 0 is always 0 (no conditioning context).

        Args:
            token_ids: [B, T] token IDs (prompt + completion, padded).
            attention_mask: [B, T] True for non-padding positions.

        Returns:
            log_probs: [B, T] per-token log-probabilities.
        """
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
        """Access vLLM's underlying nn.Module.

        The internal path varies across vLLM versions. This tries the
        common locations and raises a clear error if none work.
        """
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
