"""Profiling utilities for invokerl.

Opt-in profiling. Use the context manager for ad-hoc tracing:

    from invokerl.profiling import profile

    with profile() as p:
        trainer.step()
    p.summary()
    p.export_perfetto("trace.json")

For CLI-driven profiling of N training steps (phase breakdown + roofline),
see `python -m invokerl.profiling --config <path> --num-steps 3`.
"""

from invokerl.profiling.trace import profile

__all__ = ["profile"]
