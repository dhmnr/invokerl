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

    def share_vllm_weights(self, vllm_params: dict[str, Tensor]) -> int:
        """Replace policy parameter data with views into vLLM's tensors.

        After this call, both models share the same GPU memory. Optimizer
        updates to policy params are immediately visible to vLLM — no
        weight sync needed.

        Handles vLLM's fused layers:
        - qkv_proj.weight → q_proj, k_proj, v_proj (contiguous views)
        - gate_up_proj.weight → gate_proj, up_proj (contiguous views)

        Must be called BEFORE creating the optimizer so moment buffers
        are initialized on the shared storage.

        Args:
            vllm_params: dict from VLLMGenerator.get_model_params().

        Returns:
            Number of parameters successfully shared.
        """
        if self.master_weights_fp32:
            logger.warning(
                "Cannot share weights in fp32 master mode — vLLM uses bf16. "
                "Falling back to copy-based sync."
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
        logger.info(
            "Shared %d/%d policy params with vLLM (%.1f MB saved)",
            shared, len(policy_params),
            sum(p.nelement() * p.element_size() for p in policy_params.values()) / 1e6,
        )
        return shared

    def _resolve_vllm_param(
        self,
        name: str,
        param: Tensor,
        vllm_params: dict[str, Tensor],
        policy_params: dict[str, Tensor],
    ) -> Tensor | None:
        """Map a single policy param to its vLLM equivalent (or a view of fused param).

        Returns the tensor to assign to param.data, or None if no match.
        """
        # Direct match (norms, embeddings, o_proj, down_proj, lm_head)
        if name in vllm_params:
            vllm_t = vllm_params[name]
            if vllm_t.shape == param.shape:
                return vllm_t.data
            # Shape mismatch — don't share (could be tied weights)
            logger.debug("Shape mismatch for %s: policy %s vs vllm %s", name, param.shape, vllm_t.shape)
            return None

        # Fused QKV: q_proj, k_proj, v_proj → qkv_proj
        for proj, offset_fn in [
            ("q_proj", lambda p, pp: (0, p.shape[0])),
            ("k_proj", lambda p, pp: (
                pp[name.replace("k_proj", "q_proj")].shape[0],
                pp[name.replace("k_proj", "q_proj")].shape[0] + p.shape[0],
            )),
            ("v_proj", lambda p, pp: (
                pp[name.replace("v_proj", "q_proj")].shape[0] + pp[name.replace("v_proj", "k_proj")].shape[0],
                pp[name.replace("v_proj", "q_proj")].shape[0] + pp[name.replace("v_proj", "k_proj")].shape[0] + p.shape[0],
            )),
        ]:
            if f".{proj}." in name:
                fused_name = name.replace(f".{proj}.", ".qkv_proj.")
                if fused_name in vllm_params:
                    start, end = offset_fn(param, policy_params)
                    return vllm_params[fused_name].data[start:end]

        # Fused gate_up: gate_proj, up_proj → gate_up_proj
        if ".gate_proj." in name:
            fused_name = name.replace(".gate_proj.", ".gate_up_proj.")
            if fused_name in vllm_params:
                return vllm_params[fused_name].data[:param.shape[0]]

        if ".up_proj." in name:
            fused_name = name.replace(".up_proj.", ".gate_up_proj.")
            if fused_name in vllm_params:
                gate_name = name.replace(".up_proj.", ".gate_proj.")
                gate_dim = policy_params[gate_name].shape[0]
                return vllm_params[fused_name].data[gate_dim:gate_dim + param.shape[0]]

        logger.debug("No vLLM match for policy param: %s", name)
        return None

    def verify_shared_weights(self, vllm_params: dict[str, Tensor]) -> bool:
        """Verify that policy and vLLM params share the same GPU memory.

        Checks data_ptr() equality (same physical memory), not just value
        equality. Call after the first optimizer step to catch silent copies.

        Returns True if all shared params have matching data_ptr().
        """
        policy_params = dict(self.model.named_parameters())
        mismatches = 0

        for name, param in policy_params.items():
            target = self._resolve_vllm_param(name, param, vllm_params, policy_params)
            if target is None:
                continue
            if param.data_ptr() != target.data_ptr():
                logger.warning("data_ptr mismatch: %s (policy=%x, vllm=%x)",
                               name, param.data_ptr(), target.data_ptr())
                mismatches += 1

        if mismatches:
            logger.error("%d params lost shared memory — falling back to copy sync", mismatches)
            return False

        logger.debug("Shared weight verification passed: all data_ptr() match")
        return True
