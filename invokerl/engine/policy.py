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

    def forward(self, token_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Compute per-token log-probs [B, T]. Position 0 is always 0."""
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

            log_probs = -F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            ).reshape(targets.shape)

        # Pad position 0 with zeros
        pad = torch.zeros(
            token_ids.shape[0], 1, device=log_probs.device, dtype=log_probs.dtype
        )
        return torch.cat([pad, log_probs], dim=1)

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
