"""
Thin Python wrapper around the _vncv_core C++ extension.

Provides the same public API as ocr.py so callers are unaffected when
the C++ backend is active:

    extract_text(filepath, *, save_annotated=False, annotated_path=None,
                 ner=False, lang='vi', return_dict=False)
    main()
"""

from __future__ import annotations

import json
import os
import warnings
import urllib.request
import zipfile
from argparse import ArgumentParser
from importlib import resources
from pathlib import Path

# C++ extension – must be importable at this point (guarded in __init__.py)
from . import _vncv_core  # type: ignore[attr-defined]

__all__ = ["extract_text", "main"]


# ─────────────────────────────────────────────────────────────────────────────
# Weights helpers (Python-side – same as ocr.py)
# ─────────────────────────────────────────────────────────────────────────────

def _get_weights_dir() -> Path:
    package = __package__ or (__name__.rsplit(".", 1)[0] if "." in __name__ else "vncv")
    try:
        return Path(resources.files(package).joinpath("weights"))
    except Exception:
        return Path(__file__).resolve().parent / "weights"


WEIGHTS_DIR = _get_weights_dir()


def download_weights() -> None:
    weights_dir = WEIGHTS_DIR
    weights_dir.mkdir(parents=True, exist_ok=True)

    essential_files = ["model_encoder.onnx", "model_decoder.onnx", "vocab.json"]
    if all((weights_dir / f).exists() for f in essential_files):
        return

    url = "https://github.com/Devhub-Solutions/VNCV/releases/download/vocab/vocab.zip"
    zip_path = weights_dir / "vocab.zip"
    try:
        print("[VNCV System] Downloading missing models from GitHub...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"[VNCV System] Extracting models to {weights_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(weights_dir)
        if zip_path.exists():
            os.remove(zip_path)
        print("[VNCV System] Models installed successfully.")
    except Exception as exc:
        raise RuntimeError(
            f"Could not download models from {url}. "
            f"Please install them manually in {weights_dir}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Module-level engine cache (re-used across calls)
# ─────────────────────────────────────────────────────────────────────────────

_engine: "_vncv_core.OcrEngine | None" = None


def _get_engine() -> "_vncv_core.OcrEngine":
    global _engine
    download_weights()
    if _engine is None:
        _engine = _vncv_core.OcrEngine(str(WEIGHTS_DIR))
    return _engine


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_text(
    filepath: str,
    save_annotated: bool = False,
    annotated_path: "str | os.PathLike | None" = None,
    ner: bool = False,
    lang: str = "vi",
    return_dict: bool = False,
):
    """
    Run OCR on an image and return detected text lines.

    Parameters
    ----------
    filepath : str
        Path to the input image.
    save_annotated : bool
        If True, saves an annotated image with bounding boxes and text labels.
    annotated_path : str | os.PathLike | None
        Optional custom path for the annotated image.
    ner : bool
        If True, attempts to export an NER dataset (Python helper required).
    lang : str
        'vi' for Vietnamese (VietOCR) or 'en' for English (CTC).
    return_dict : bool
        If True returns a list of dicts with 'text', 'confidence', 'box'.
    """
    engine = _get_engine()

    ann_path = str(annotated_path) if annotated_path is not None else ""

    results = engine.extract(
        str(filepath),
        lang=lang,
        save_annotated=save_annotated,
        annotated_path=ann_path,
        return_dict=return_dict,
    )

    if ner:
        try:
            from generate_ner_dataset import process_single_document, save_jsonl
        except ImportError:
            warnings.warn(
                "NER export requested but generate_ner_dataset module is missing.",
                RuntimeWarning,
            )
        else:
            texts = [r["text"] if return_dict else r for r in results]
            doc_id = Path(filepath).stem
            doc = process_single_document(texts, doc_id=doc_id)
            if doc:
                save_jsonl([doc], "dataset.jsonl")
                print("[NER] Saved → dataset.jsonl")

    return results


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("filepath", type=str, help="image file path")
    parser.add_argument("--ner", action="store_true",
                        help="export NER dataset after OCR")
    parser.add_argument("--save-annotated", action="store_true",
                        help="save annotated image with bounding boxes")
    parser.add_argument("--annotated-path", type=str, default=None,
                        help="custom output path for annotated image")
    parser.add_argument("--lang", type=str, default="vi", choices=["vi", "en"],
                        help="language for OCR (vi or en)")
    parser.add_argument("--json", action="store_true",
                        help="output result as JSON instead of Python list")
    args = parser.parse_args()

    results = extract_text(
        args.filepath,
        save_annotated=args.save_annotated,
        annotated_path=args.annotated_path,
        ner=args.ner,
        lang=args.lang,
        return_dict=args.json,
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(results)
