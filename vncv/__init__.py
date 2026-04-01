"""
vncv – Vietnamese Computer Vision OCR package.

When the C++ extension (_vncv_core) is available it is used as the primary
inference backend (faster, no heavy Python dependencies at runtime).
If the extension is not present (e.g. when running from source without a
build step) the pure-Python fallback in ocr.py is used automatically.
"""

try:
    from ._cpp_wrapper import extract_text, main  # C++ backend  # noqa: F401
    _BACKEND = "cpp"
except ImportError:
    from .ocr import extract_text, main           # Python fallback  # noqa: F401
    _BACKEND = "python"

__all__ = ["extract_text", "main", "_BACKEND"]
