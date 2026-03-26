"""Policy model wrapper for training.

Wraps a HuggingFace CausalLM model for:
- Forward pass: token_ids → per-token log-probs (with gradients)
- Backward: autograd handles it
- Weight extraction for syncing to generation engine

Mixed precision: fp32 master weights (for precise optimizer updates) with
bf16 forward pass (for speed and memory). This matches standard RL training
practice and preserves small gradient updates that bf16 would round away.

This is NOT hackable — researchers never touch this.
"""

from __future__ import annotations

import contextlib
import logging

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class PolicyModel:
    """PyTorch model wrapper for policy training.

    Handles log-prob computation with autograd. The trainer calls
    forward() to get log_probs, then loss.backward() for gradients.

    Uses fp32 master weights with bf16 autocast for forward/backward.
    This ensures optimizer updates (at lr=1e-6) don't get rounded away
    by bf16's limited mantissa precision.
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

        if master_weights_fp32:
            # Load in fp32 for precise optimizer updates.
            # Forward pass uses autocast to bf16 for speed.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=torch.float32,
                device_map=device,
                attn_implementation="sdpa",
            )
            self._autocast_dtype = dtype
            logger.info(
                "PolicyModel: fp32 master weights, %s autocast forward",
                dtype,
            )
        else:
            # Pure bf16 — smaller memory, lower precision updates.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=dtype,
                device_map=device,
                attn_implementation="sdpa",
            )
            self._autocast_dtype = None
            logger.info("PolicyModel: pure %s (no master weights)", dtype)

        self.model.train()
        self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Compute per-token log-probs for the given sequences.

        Args:
            token_ids: [B, T] token IDs.
            attention_mask: [B, T] attention mask.

        Returns:
            log_probs: [B, T] per-token log-probs. The value at position t
                       is log p(token_ids[t] | token_ids[:t]). Position 0
                       is always 0 (no prediction for the first token).
        """
        token_ids = token_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Autocast: fp32 weights are cast to bf16 for the forward pass,
        # giving bf16 speed with fp32 optimizer precision.
        ctx = (
            torch.autocast("cuda", dtype=self._autocast_dtype)
            if self._autocast_dtype is not None
            else contextlib.nullcontext()
        )
        with ctx:
            outputs = self.model(
                input_ids=token_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits[:, :-1, :]  # [B, T-1, V]
            targets = token_ids[:, 1:]          # [B, T-1]

            # Fused cross_entropy avoids materializing [B, T, V] log_softmax
            log_probs = -F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            ).reshape(targets.shape)  # [B, T-1]

        # Pad to match original sequence length (position 0 = 0)
        pad = torch.zeros(token_ids.shape[0], 1, device=log_probs.device,
                          dtype=log_probs.dtype)
        return torch.cat([pad, log_probs], dim=1)  # [B, T]

    @torch.no_grad()
    def forward_no_grad(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Same as forward() but without gradients. For reference model."""
        return self.forward(token_ids, attention_mask)

    def get_state_dict(self) -> dict[str, Tensor]:
        """Return model state dict for weight sync to generation engine."""
        return self.model.state_dict()

    def parameters(self):
        """Trainable parameters for optimizer."""
        return self.model.parameters()
