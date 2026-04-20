"""Opt-in PyTorch profiler context manager with Perfetto export.

Usage:
    from invokerl.profiling import profile

    with profile() as p:
        trainer.step()

    p.summary()                    # wall vs CUDA busy vs unaccounted
    p.export_perfetto("trace.json")  # open at ui.perfetto.dev
"""

from __future__ import annotations

import os
import time

import torch


class Profile:
    """Holds the result of a profiled block.

    Attributes (valid after exiting the context):
        wall_s: wall-clock duration of the block, in seconds
        cpu_self_us: total CPU self time across profiler events (microseconds)
        cuda_busy_us: merged CUDA busy time (microseconds), accounts for overlap
        unaccounted_us: wall_us - cuda_busy_us
    """

    def __init__(self, record_shapes: bool = True, with_stack: bool = True,
                 with_flops: bool = True):
        self._record_shapes = record_shapes
        self._with_stack = with_stack
        self._with_flops = with_flops
        self._prof = None
        self.wall_s: float = 0.0
        self.cpu_self_us: float = 0.0
        self.cuda_busy_us: float = 0.0
        self.unaccounted_us: float = 0.0

    def __enter__(self) -> "Profile":
        self._prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=self._record_shapes,
            with_stack=self._with_stack,
            with_flops=self._with_flops,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._prof.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.wall_s = time.perf_counter() - self._t0
        self._prof.stop()
        self._analyze()
        return False

    def _analyze(self):
        """Aggregate wall / CPU / CUDA time from profiler events."""
        cpu_total_us = 0
        cuda_events = []

        for evt in self._prof.key_averages():
            if evt.self_cpu_time_total > 0:
                cpu_total_us += evt.self_cpu_time_total

        for evt in self._prof.events():
            if evt.device_type == torch.autograd.DeviceType.CUDA:
                dur = evt.time_range.end - evt.time_range.start
                if dur > 0:
                    cuda_events.append((evt.time_range.start, evt.time_range.end))

        # Merge overlapping CUDA intervals to get true GPU busy time
        cuda_busy_us = 0
        if cuda_events:
            cuda_events.sort()
            merged_start, merged_end = cuda_events[0]
            for s, e in cuda_events[1:]:
                if s <= merged_end:
                    merged_end = max(merged_end, e)
                else:
                    cuda_busy_us += merged_end - merged_start
                    merged_start, merged_end = s, e
            cuda_busy_us += merged_end - merged_start

        self.cpu_self_us = cpu_total_us
        self.cuda_busy_us = cuda_busy_us
        self.unaccounted_us = self.wall_s * 1e6 - cuda_busy_us

    def summary(self) -> None:
        """Print a wall-clock vs CUDA-busy vs unaccounted breakdown."""
        wall_us = self.wall_s * 1e6
        print(f"  Wall clock:          {self.wall_s:>10.3f}s")
        print(f"  CPU self time:       {self.cpu_self_us / 1e6:>10.3f}s  "
              f"({self.cpu_self_us / wall_us * 100:>5.1f}% of wall)")
        print(f"  CUDA busy time:      {self.cuda_busy_us / 1e6:>10.3f}s  "
              f"({self.cuda_busy_us / wall_us * 100:>5.1f}% of wall)")
        print(f"  Unaccounted:         {self.unaccounted_us / 1e6:>10.3f}s  "
              f"({self.unaccounted_us / wall_us * 100:>5.1f}% of wall)")

    def export_perfetto(self, path: str) -> None:
        """Write a Chrome/Perfetto trace. Open at ui.perfetto.dev."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._prof.export_chrome_trace(path)
        print(f"  Trace saved to: {path}")
        print(f"  Open at: https://ui.perfetto.dev")


def profile(**kwargs) -> Profile:
    """Create a Profile context manager. See Profile for kwargs."""
    return Profile(**kwargs)
