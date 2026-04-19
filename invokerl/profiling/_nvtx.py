"""Optional NVTX import with graceful fallback."""

from __future__ import annotations

try:
    import nvtx
    _has_nvtx = True
except ImportError:
    _has_nvtx = False

    class _FakeAnnotate:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class nvtx:  # type: ignore[no-redef]
        annotate = _FakeAnnotate
