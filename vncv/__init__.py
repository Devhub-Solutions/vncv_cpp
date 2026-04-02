"""vncv package entrypoint.

PyPI distribution `vncv_cpp` is intended to run with compiled C++ bindings
for the current platform (Linux/macOS/Windows).
"""

import platform


def _runtime_platform() -> str:
    return f"{platform.system().lower()}-{platform.machine().lower()}"


try:
    from ._cpp_wrapper import extract_text, main  # noqa: F401
    _BACKEND = "cpp"
except ImportError as exc:  # pragma: no cover - import-time safety
    raise ImportError(
        "vncv_cpp requires the compiled C++ backend for your platform. "
        f"Detected runtime: {_runtime_platform()}. "
        "Please install the binary wheel built for this OS/architecture from PyPI "
        "(do not use source-only installation for production runtime)."
    ) from exc

__all__ = ["extract_text", "main", "_BACKEND", "_runtime_platform"]
