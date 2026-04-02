"""Compatibility module that always routes OCR calls to the C++ backend.

`vncv_cpp` is distributed as platform-specific binary wheels. This module keeps
legacy imports working (`from vncv.ocr import extract_text`) while ensuring
runtime execution uses the pybind11 C++ engine.
"""

from ._cpp_wrapper import extract_text, main

__all__ = ["extract_text", "main"]
