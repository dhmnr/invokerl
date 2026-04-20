"""Shared constants: phase colors, ordering, hardware specs."""

from __future__ import annotations

from dataclasses import dataclass


PHASE_COLORS = {
    "generation": "#4285F4",
    "reward": "#FBBC04",
    "ref_forward": "#34A853",
    "policy_forward": "#FF6D01",
    "loss_computation": "#EA4335",
    "backward": "#9334E6",
    "optimizer_step": "#00BCD4",
    "weight_sync": "#795548",
    "warmup": "#9E9E9E",
}

PHASE_ORDER = [
    "generation", "reward", "ref_forward", "policy_forward",
    "loss_computation", "backward", "optimizer_step", "weight_sync",
]


@dataclass
class GPUSpecs:
    """Hardware specs for roofline model."""

    name: str = "RTX 5090"
    peak_bf16_tflops: float = 209.5
    peak_fp32_tflops: float = 104.8
    peak_fp16_tflops: float = 209.5
    mem_bandwidth_tb_s: float = 1.79
    vram_gb: float = 32.0

    @property
    def mem_bandwidth_gb_s(self) -> float:
        return self.mem_bandwidth_tb_s * 1000

    @property
    def ridge_point_bf16(self) -> float:
        return self.peak_bf16_tflops * 1000 / self.mem_bandwidth_gb_s

    @property
    def ridge_point_fp32(self) -> float:
        return self.peak_fp32_tflops * 1000 / self.mem_bandwidth_gb_s


RTX_5090 = GPUSpecs()
