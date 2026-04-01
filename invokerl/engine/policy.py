"""Policy model wrapper — HuggingFace CausalLM with log-prob computation."""

from __future__ import annotations

import contextlib
import logging

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as ckpt
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
        ce_chunk_size: int = 64,
    ) -> Tensor:
        """Compute per-token log-probs [B, T]. Position 0 is always 0.

        Uses chunked lm_head + cross-entropy to avoid materializing the full
        [B*T, vocab_size] logits tensor. Instead of projecting all tokens to
        vocab at once (~10GB for B×G=56 at 512tok), we:
          1. Run the base transformer to get hidden states [B, T, H]
          2. Loop over sequence chunks: project chunk to vocab via lm_head,
             compute CE loss, discard logits — peak memory is B*chunk*V*4

        Args:
            ce_chunk_size: Number of tokens per lm_head+CE chunk.
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
            # Get hidden states from base transformer (no lm_head projection).
            # self.model.model is the base transformer (e.g., Qwen2Model)
            # without the final vocab projection.
            hidden_states = self.model.model(
                input_ids=token_ids, attention_mask=attention_mask,
            ).last_hidden_state[:, :-1, :]  # [B, T-1, H]

            targets = token_ids[:, 1:]  # [B, T-1]

            log_probs = self._chunked_lm_head_ce(
                hidden_states, targets, ce_chunk_size,
            )

        # Pad position 0 with zeros
        pad = torch.zeros(
            token_ids.shape[0], 1, device=log_probs.device, dtype=log_probs.dtype
        )
        return torch.cat([pad, log_probs], dim=1)

    def _chunked_lm_head_ce(
        self,
        hidden_states: Tensor,
        targets: Tensor,
        chunk_size: int,
    ) -> Tensor:
        """Fused chunked lm_head projection + cross-entropy.

        Projects hidden states to vocab and computes CE loss in chunks along
        the sequence dimension. Each chunk's logits are discarded after computing
        the loss, so peak memory is B * chunk_size * V instead of B * T * V.

        Args:
            hidden_states: [B, T, H] final hidden states from the transformer.
            targets: [B, T] target token IDs.
            chunk_size: Tokens per chunk.

        Returns:
            log_probs: [B, T] per-token log-probabilities (negated CE).
        """
        B, T, H = hidden_states.shape
        lm_head = self.model.lm_head  # Linear(H, V)
        V = lm_head.out_features

        # Small enough to do in one shot — skip chunking overhead.
        if B * T * V * 4 < 2 * (1024 ** 3):  # < 2 GB
            logits = lm_head(hidden_states)
            return -F.cross_entropy(
                logits.reshape(-1, V),
                targets.reshape(-1),
                reduction="none",
            ).reshape(B, T)

        log_probs = torch.empty(B, T, device=hidden_states.device, dtype=hidden_states.dtype)
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            # Recompute logits during backward instead of saving them.
            # Without this, autograd saves all chunk logits simultaneously,
            # defeating the chunked memory savings (~8 GB for B×G=56).
            log_probs[:, start:end] = ckpt(
                self._chunk_ce_fn,
                lm_head.weight,
                hidden_states[:, start:end, :].contiguous(),
                targets[:, start:end].contiguous(),
                use_reentrant=False,
            )

        return log_probs

    @staticmethod
    def _chunk_ce_fn(
        lm_head_weight: Tensor,
        hidden_chunk: Tensor,
        target_chunk: Tensor,
    ) -> Tensor:
        """Compute lm_head projection + CE for a single chunk.

        Separated as a static method so torch.utils.checkpoint can
        recompute the logits during backward (instead of saving them).
        """
        B, cs, H = hidden_chunk.shape
        V = lm_head_weight.shape[0]
        chunk_logits = F.linear(hidden_chunk, lm_head_weight)  # [B, cs, V]
        return -F.cross_entropy(
            chunk_logits.reshape(-1, V),
            target_chunk.reshape(-1),
            reduction="none",
        ).reshape(B, cs)

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
