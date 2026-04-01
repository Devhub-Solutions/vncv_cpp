"""
Shared pytest fixtures for VNCV test suite.
"""

import json
import os
import tempfile

import cv2
import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(REPO_ROOT, "vncv", "weights")
DETECTION_ONNX = os.path.join(WEIGHTS_DIR, "detection.onnx")
CLASSIFICATION_ONNX = os.path.join(WEIGHTS_DIR, "classification.onnx")
RECOGNITION_EN_ONNX = os.path.join(WEIGHTS_DIR, "recognition.onnx")
ENCODER_ONNX = os.path.join(WEIGHTS_DIR, "model_encoder.onnx")
DECODER_ONNX = os.path.join(WEIGHTS_DIR, "model_decoder.onnx")
VOCAB_JSON = os.path.join(WEIGHTS_DIR, "vocab.json")
TEST_IMAGE_PATH = os.path.join(
    REPO_ROOT, "vietocr-onnx-package", "test_images", "test_image.png"
)


# ---------------------------------------------------------------------------
# Pytest markers – skip integration tests when model files are missing
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_vietocr_weights: skip if VietOCR encoder/decoder/vocab not present",
    )
    config.addinivalue_line(
        "markers",
        "requires_detection_weights: skip if detection.onnx not present",
    )
    config.addinivalue_line(
        "markers",
        "requires_classification_weights: skip if classification.onnx not present",
    )
    config.addinivalue_line(
        "markers",
        "requires_recognition_en_weights: skip if recognition.onnx not present",
    )
    config.addinivalue_line(
        "markers",
        "requires_test_image: skip if test_image.png not present",
    )


def pytest_runtest_setup(item):
    for marker in item.iter_markers():
        if marker.name == "requires_vietocr_weights":
            missing = [
                p
                for p in [ENCODER_ONNX, DECODER_ONNX, VOCAB_JSON]
                if not os.path.exists(p)
            ]
            if missing:
                pytest.skip(f"VietOCR weights not found: {missing}")
        elif marker.name == "requires_detection_weights":
            if not os.path.exists(DETECTION_ONNX):
                pytest.skip(f"detection.onnx not found at {DETECTION_ONNX}")
        elif marker.name == "requires_classification_weights":
            if not os.path.exists(CLASSIFICATION_ONNX):
                pytest.skip(f"classification.onnx not found at {CLASSIFICATION_ONNX}")
        elif marker.name == "requires_recognition_en_weights":
            if not os.path.exists(RECOGNITION_EN_ONNX):
                pytest.skip(f"recognition.onnx not found at {RECOGNITION_EN_ONNX}")
        elif marker.name == "requires_test_image":
            if not os.path.exists(TEST_IMAGE_PATH):
                pytest.skip(f"test_image.png not found at {TEST_IMAGE_PATH}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def vocab_json_path():
    """Create a minimal vocab.json for unit tests."""
    vocab_data = {
        "chars": list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?"),
        "total_vocab_size": 4 + len(
            list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?")
        ),
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(vocab_data, f, ensure_ascii=False)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture(scope="session")
def small_bgr_image():
    """A small 100×200 BGR image with random content."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (100, 200, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def square_bgr_image():
    """A 64×64 BGR image for fast inference tests."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def text_line_pil():
    """A small PIL RGB image that resembles a text line (white background)."""
    img = Image.new("RGB", (200, 32), color=(255, 255, 255))
    return img


@pytest.fixture(scope="session")
def detection_instance():
    """Singleton Detection object (expensive to create)."""
    if not os.path.exists(DETECTION_ONNX):
        pytest.skip(f"detection.onnx not found")
    from vncv.ocr import Detection
    return Detection(DETECTION_ONNX)


@pytest.fixture(scope="session")
def classification_instance():
    """Singleton Classification object."""
    if not os.path.exists(CLASSIFICATION_ONNX):
        pytest.skip(f"classification.onnx not found")
    from vncv.ocr import Classification
    return Classification(CLASSIFICATION_ONNX)


@pytest.fixture(scope="session")
def recognition_en_instance():
    """Singleton EnglishRecognition object."""
    if not os.path.exists(RECOGNITION_EN_ONNX):
        pytest.skip(f"recognition.onnx not found")
    from vncv.ocr import EnglishRecognition
    return EnglishRecognition(RECOGNITION_EN_ONNX)
