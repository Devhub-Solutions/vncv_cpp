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
    runtime = _runtime_platform()
    suggestion = "pip install vncv_cpp"
    if "windows" in runtime:
        suggestion = "pip install vncv_cpp --only-binary=:all: --platform win_amd64"
    elif "linux" in runtime:
        suggestion = "pip install vncv_cpp --only-binary=:all: --platform manylinux2014_x86_64"
    
    raise ImportError(
        f"vncv_cpp binary backend not found for {runtime}.\n"
        f"Please ensure you have installed the correct binary wheel:\n"
        f"  {suggestion}\n"
        "Note: Source-only installations are not supported for production runtime."
    ) from exc

__all__ = ["extract_text", "main", "_BACKEND", "_runtime_platform"]
