"""Policy model wrapper — HuggingFace CausalLM with log-prob computation."""

from __future__ import annotations

import contextlib
import logging

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class PolicyModel:
    """Wraps a HuggingFace CausalLM for policy training.

    forward() returns per-token log-probs with gradients.
    Supports optional fp32 master weights with bf16 autocast.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        master_weights_fp32: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.master_weights_fp32 = master_weights_fp32

        load_dtype = torch.float32 if master_weights_fp32 else dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=load_dtype,
            device_map=device,
            attn_implementation="sdpa",
        )
        self._autocast_dtype = dtype if master_weights_fp32 else None
        self.model.train()
        self.model.gradient_checkpointing_enable()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info(
            "PolicyModel: %s%s",
            load_dtype,
            f" (autocast {dtype})" if master_weights_fp32 else "",
        )

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
        ce_chunk_size: int = 512,
    ) -> Tensor:
        """Compute per-token log-probs [B, T]. Position 0 is always 0.

        Uses chunked cross-entropy to avoid materializing the full
        [B*T, vocab_size] logits tensor. For Qwen3 (vocab=151936), this
        reduces peak memory from ~10GB to ~1.2GB at B×G=56.

        Args:
            ce_chunk_size: Number of tokens per cross-entropy chunk.
                           Lower = less memory, slightly more overhead.
        """
        token_ids = token_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        ctx = (
            torch.autocast("cuda", dtype=self._autocast_dtype)
            if self._autocast_dtype
            else contextlib.nullcontext()
        )
        with ctx:
            logits = self.model(
                input_ids=token_ids, attention_mask=attention_mask
            ).logits[:, :-1, :]  # [B, T-1, V]
            targets = token_ids[:, 1:]  # [B, T-1]

            log_probs = self._chunked_cross_entropy(
                logits, targets, ce_chunk_size,
            )

        # Pad position 0 with zeros
        pad = torch.zeros(
            token_ids.shape[0], 1, device=log_probs.device, dtype=log_probs.dtype
        )
        return torch.cat([pad, log_probs], dim=1)

    @staticmethod
    def _chunked_cross_entropy(
        logits: Tensor,
        targets: Tensor,
        chunk_size: int,
    ) -> Tensor:
        """Compute cross-entropy in chunks along the token dimension.

        Instead of materializing (B*T, V) in one shot, processes chunks
        of `chunk_size` tokens. Peak memory: B * chunk_size * V * 4 bytes
        instead of B * T * V * 4 bytes.

        Args:
            logits: [B, T, V] model output logits.
            targets: [B, T] target token IDs.
            chunk_size: Tokens per chunk.

        Returns:
            log_probs: [B, T] per-token log-probabilities (negated CE).
        """
        B, T, V = logits.shape

        # Small enough to do in one shot — skip chunking overhead.
        if B * T * V * 4 < 2 * (1024 ** 3):  # < 2 GB
            return -F.cross_entropy(
                logits.reshape(-1, V),
                targets.reshape(-1),
                reduction="none",
            ).reshape(B, T)

        log_probs = torch.empty(B, T, device=logits.device, dtype=logits.dtype)
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            log_probs[:, start:end] = -F.cross_entropy(
                logits[:, start:end, :].reshape(-1, V),
                targets[:, start:end].reshape(-1),
                reduction="none",
            ).reshape(B, end - start)

        return log_probs

    @torch.no_grad()
    def forward_no_grad(self, token_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """forward() without gradients. For reference model."""
        return self.forward(token_ids, attention_mask)

    def get_state_dict(self) -> dict[str, Tensor]:
        return self.model.state_dict()

    def parameters(self):
        return self.model.parameters()

    # -- Shared weights with vLLM ---------------------------------------------

    def share_vllm_weights(self, vllm_params: dict[str, Tensor]) -> int:
        """Point policy params at vLLM's GPU tensors (zero-copy shared memory).

        Handles vLLM's fused layers (qkv_proj, gate_up_proj).
        Must be called BEFORE creating the optimizer.
        Returns number of parameters shared.
        """
        if self.master_weights_fp32:
            logger.warning("Cannot share weights in fp32 mode — dtype mismatch")
            return 0

        # Can't share across devices (disaggregated mode).
        if vllm_params:
            vllm_device = next(iter(vllm_params.values())).device
            policy_device = next(self.model.parameters()).device
            if vllm_device != policy_device:
                logger.info(
                    "Cross-device: weight sharing disabled (vllm=%s, policy=%s)",
                    vllm_device, policy_device,
                )
                return 0

        policy_params = dict(self.model.named_parameters())
        shared = 0

        for name, param in policy_params.items():
            target = self._resolve_vllm_param(name, param, vllm_params, policy_params)
            if target is not None:
                param.data = target
                shared += 1

        self._weights_shared = True
        saved_mb = sum(p.nelement() * p.element_size() for p in policy_params.values()) / 1e6
        logger.info("Shared %d/%d params with vLLM (%.1f MB)", shared, len(policy_params), saved_mb)
        return shared

    def _resolve_vllm_param(
        self,
        name: str,
        param: Tensor,
        vllm_params: dict[str, Tensor],
        policy_params: dict[str, Tensor],
    ) -> Tensor | None:
        """Map a policy param to its vLLM equivalent, handling fused layers."""
        # Direct match
        if name in vllm_params:
            if vllm_params[name].shape == param.shape:
                return vllm_params[name].data
            return None

        # Fused QKV: q/k/v_proj -> qkv_proj (row-sliced views)
        for proj in ("q_proj", "k_proj", "v_proj"):
            if f".{proj}." not in name:
                continue
            fused = name.replace(f".{proj}.", ".qkv_proj.")
            if fused not in vllm_params:
                continue

            q_name = name.replace(f".{proj}.", ".q_proj.")
            k_name = name.replace(f".{proj}.", ".k_proj.")
            q_dim = policy_params[q_name].shape[0]
            k_dim = policy_params[k_name].shape[0]

            offsets = {"q_proj": (0, q_dim), "k_proj": (q_dim, q_dim + k_dim),
                       "v_proj": (q_dim + k_dim, q_dim + k_dim + param.shape[0])}
            start, end = offsets[proj]
            return vllm_params[fused].data[start:end]

        # Fused gate_up: gate_proj, up_proj -> gate_up_proj
        for proj in ("gate_proj", "up_proj"):
            if f".{proj}." not in name:
                continue
            fused = name.replace(f".{proj}.", ".gate_up_proj.")
            if fused not in vllm_params:
                continue

            gate_dim = policy_params[name.replace(f".{proj}.", ".gate_proj.")].shape[0]
            offsets = {"gate_proj": (0, gate_dim),
                       "up_proj": (gate_dim, gate_dim + param.shape[0])}
            start, end = offsets[proj]
            return vllm_params[fused].data[start:end]

        return None

    def verify_shared_weights(self, vllm_params: dict[str, Tensor]) -> bool:
        """Check data_ptr() equality after optimizer step."""
        policy_params = dict(self.model.named_parameters())
        mismatches = 0

        for name, param in policy_params.items():
            target = self._resolve_vllm_param(name, param, vllm_params, policy_params)
            if target is not None and param.data_ptr() != target.data_ptr():
                logger.warning("data_ptr mismatch: %s", name)
                mismatches += 1

        if mismatches:
            logger.error("%d params lost shared memory", mismatches)
        return mismatches == 0
