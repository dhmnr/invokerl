"""Opt-in profiling — one file, ~150 lines.

Usage:

    from invokerl import profile

    with profile() as p:
        trainer.step()

    p.summary()                  # prints wall / CPU / CUDA / unaccounted + per-phase
    p.export_trace("trace.json") # open in Chrome tracing or Perfetto

The per-phase numbers come from NVTX markers emitted by Trainer.step().
No NVTX installed? The markers are no-ops and per-phase lines simply don't print.

Running under Nsight Systems? Markers show up automatically:

    nsys profile --trace=cuda,nvtx python my_script.py
"""

from __future__ import annotations

import os
import time

import torch


# --- optional NVTX ---------------------------------------------------------
try:
    import nvtx as _nvtx
    _HAS_NVTX = True
except ImportError:
    _HAS_NVTX = False


class _Annotate:
    """Combined phase marker.

    Emits simultaneously into:
      1. NVTX (for `nsys profile`), when nvtx is installed
      2. torch.profiler (for in-process `profile()`), always
    """

    def __init__(self, name: str, color: str = "blue"):
        self._name = name
        self._color = color
        self._record = torch.profiler.record_function(name)
        self._nvtx_ctx = _nvtx.annotate(name, color=color) if _HAS_NVTX else None

    def __enter__(self):
        self._record.__enter__()
        if self._nvtx_ctx is not None:
            self._nvtx_ctx.__enter__()
        return self

    def __exit__(self, *exc):
        if self._nvtx_ctx is not None:
            self._nvtx_ctx.__exit__(*exc)
        self._record.__exit__(*exc)
        return False


def annotate(name: str, color: str = "blue"):
    """Phase marker — emits both NVTX (for nsys) and torch.profiler events.

    Used inside Trainer.step() so:
      - `nsys profile python script.py` gets kernel-level phase markers
      - `with profile(): trainer.step()` gets per-phase wall time
    Neither requires extra setup.
    """
    return _Annotate(name, color=color)


# --- Profile context manager -----------------------------------------------

class Profile:
    """Result of a profiled block.

    After exit, attributes are valid:
        wall_s, cpu_self_us, cuda_busy_us, unaccounted_us
        phase_times_us: {phase_name: total_microseconds}
    """

    def __init__(self, with_stack: bool = True):
        self._with_stack = with_stack
        self._prof = None
        self.wall_s: float = 0.0
        self.cpu_self_us: float = 0.0
        self.cuda_busy_us: float = 0.0
        self.unaccounted_us: float = 0.0
        self.phase_times_us: dict[str, float] = {}

    def __enter__(self) -> "Profile":
        self._prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=self._with_stack,
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
        """Compute wall/CPU/CUDA breakdown + per-phase times from NVTX markers."""
        cpu_total = 0
        for evt in self._prof.key_averages():
            if evt.self_cpu_time_total > 0:
                cpu_total += evt.self_cpu_time_total
        self.cpu_self_us = cpu_total

        # Merge overlapping CUDA intervals for true GPU busy time
        intervals = []
        for evt in self._prof.events():
            if evt.device_type == torch.autograd.DeviceType.CUDA:
                dur = evt.time_range.end - evt.time_range.start
                if dur > 0:
                    intervals.append((evt.time_range.start, evt.time_range.end))

        busy = 0
        if intervals:
            intervals.sort()
            ms, me = intervals[0]
            for s, e in intervals[1:]:
                if s <= me:
                    me = max(me, e)
                else:
                    busy += me - ms
                    ms, me = s, e
            busy += me - ms
        self.cuda_busy_us = busy
        self.unaccounted_us = self.wall_s * 1e6 - busy

        # Per-phase times from torch.profiler record_function events.
        # record_function emits both CPU-side and CUDA-side user_annotation events;
        # we take CPU-side wall time only (that's the phase's wall duration).
        phase_totals: dict[str, float] = {}
        for evt in self._prof.events():
            if (getattr(evt, "is_user_annotation", False)
                    and evt.device_type == torch.autograd.DeviceType.CPU):
                dur = evt.time_range.end - evt.time_range.start
                if dur > 0:
                    phase_totals[evt.name] = phase_totals.get(evt.name, 0.0) + dur
        self.phase_times_us = phase_totals

    def summary(self) -> None:
        """Print a wall/CPU/CUDA/unaccounted breakdown and per-phase times."""
        wall_us = self.wall_s * 1e6
        print(f"  Wall clock:     {self.wall_s:>8.3f}s")
        print(f"  CPU self time:  {self.cpu_self_us / 1e6:>8.3f}s  "
              f"({self.cpu_self_us / wall_us * 100:>5.1f}%)")
        print(f"  CUDA busy:      {self.cuda_busy_us / 1e6:>8.3f}s  "
              f"({self.cuda_busy_us / wall_us * 100:>5.1f}%)")
        print(f"  Unaccounted:    {self.unaccounted_us / 1e6:>8.3f}s  "
              f"({self.unaccounted_us / wall_us * 100:>5.1f}%)")

        if self.phase_times_us:
            print("\n  Per-phase:")
            total = sum(self.phase_times_us.values())
            for name, t in sorted(self.phase_times_us.items(), key=lambda x: -x[1]):
                pct = t / wall_us * 100 if wall_us > 0 else 0
                print(f"    {name:<20s} {t / 1e6:>8.3f}s  ({pct:>5.1f}%)")

    def export_trace(self, path: str) -> None:
        """Write a Chrome tracing JSON. Open at ui.perfetto.dev or chrome://tracing."""
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._prof.export_chrome_trace(path)
        print(f"  Trace saved to: {path}")


def profile(**kwargs) -> Profile:
    """Create a Profile context manager. See `Profile` for kwargs."""
    return Profile(**kwargs)
