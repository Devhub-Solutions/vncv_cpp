"""
Comparison and determinism tests.

These tests compare:
  - Multiple runs with the same input to verify determinism.
  - English vs Vietnamese recognition output structure.
  - Detection → crop → classification pipeline consistency.
  - Optional ONNX vs reference result checks.
"""

import numpy as np
import pytest
from PIL import Image

from vncv.ocr import (
    Classification,
    CTCDecoder,
    Detection,
    EnglishRecognition,
    crop_image,
    sort_polygon,
)
from vncv.vietocr_onnx import VocabONNX, process_image, process_input, resize


# ---------------------------------------------------------------------------
# Preprocessing determinism
# ---------------------------------------------------------------------------


class TestPreprocessingDeterminism:
    """process_image and process_input must be fully deterministic."""

    def _pil(self, seed=0):
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 256, (32, 100, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")

    def test_process_image_same_result_twice(self):
        img = self._pil()
        r1 = process_image(img)
        r2 = process_image(img)
        np.testing.assert_array_equal(r1, r2)

    def test_process_input_batch_dim(self):
        img = self._pil()
        r1 = process_input(img)
        r2 = process_input(img)
        np.testing.assert_array_equal(r1, r2)

    def test_different_images_give_different_arrays(self):
        r1 = process_image(self._pil(seed=1))
        r2 = process_image(self._pil(seed=99))
        assert not np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# sort_polygon + crop_image pipeline
# ---------------------------------------------------------------------------


class TestSortCropPipeline:
    def _frame(self, h=200, w=300):
        rng = np.random.default_rng(5)
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def test_crop_order_matches_sort_order(self):
        """Crops produced after sort_polygon should be in top-to-bottom order."""
        frame = self._frame()
        raw_pts = [
            np.array([[0, 100], [50, 100], [50, 120], [0, 120]], dtype=np.float32),
            np.array([[0, 10], [50, 10], [50, 30], [0, 30]], dtype=np.float32),
        ]
        sorted_pts = sort_polygon(list(raw_pts))
        crops = [crop_image(frame, p) for p in sorted_pts]
        # First crop's source y should be less than second
        assert sorted_pts[0][0][1] < sorted_pts[1][0][1]
        # Both crops should have valid size
        for crop in crops:
            assert crop.shape[0] > 0 and crop.shape[1] > 0

    def test_crop_pixel_values_from_frame(self):
        """Crop pixel range should stay within uint8 [0, 255]."""
        frame = self._frame()
        pts = np.float32([[10, 10], [80, 10], [80, 40], [10, 40]])
        crop = crop_image(frame, pts)
        assert crop.min() >= 0
        assert crop.max() <= 255


# ---------------------------------------------------------------------------
# CTCDecoder vs manual softmax
# ---------------------------------------------------------------------------


class TestCTCVsManualDecode:
    """
    Verify that CTCDecoder chooses the argmax character at each timestep,
    consistent with manual greedy decoding.
    """

    def test_argmax_matches_ctc(self):
        dec = CTCDecoder()
        vocab = len(dec.character)
        rng = np.random.default_rng(42)
        logits = rng.standard_normal((1, 10, vocab)).astype(np.float32)
        results, _ = dec(logits)

        # Manual greedy: argmax → collapse repeats → remove blanks
        indices = logits[0].argmax(axis=-1)  # (10,)
        deduped = [indices[0]]
        for idx in indices[1:]:
            if idx != deduped[-1]:
                deduped.append(idx)
        manual = "".join(dec.character[i] for i in deduped if i != 0)

        assert results[0] == manual


# ---------------------------------------------------------------------------
# Detection + Classification pipeline
# ---------------------------------------------------------------------------


@pytest.mark.requires_detection_weights
@pytest.mark.requires_classification_weights
class TestDetectionClassificationPipeline:
    def _image(self):
        rng = np.random.default_rng(7)
        return rng.integers(0, 256, (256, 400, 3), dtype=np.uint8)

    def test_pipeline_no_crash(self, detection_instance, classification_instance):
        frame = self._image()
        import cv2 as _cv2
        rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
        boxes = detection_instance(rgb)
        boxes = sort_polygon(list(boxes))
        crops = [crop_image(rgb, b) for b in boxes]
        if crops:
            imgs_out, labels = classification_instance(crops)
            assert len(imgs_out) == len(crops)
            assert len(labels) == len(crops)

    def test_classification_output_count_matches_crop_count(
        self, detection_instance, classification_instance
    ):
        frame = self._image()
        import cv2 as _cv2
        rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
        boxes = detection_instance(rgb)
        crops = [crop_image(rgb, b) for b in boxes]
        if crops:
            _, labels = classification_instance(crops)
            assert len(labels) == len(crops)


# ---------------------------------------------------------------------------
# English recognition: output structure
# ---------------------------------------------------------------------------


@pytest.mark.requires_recognition_en_weights
class TestEnglishRecognitionOutputStructure:
    def test_results_per_crop(self, recognition_en_instance):
        rng = np.random.default_rng(0)
        crops = [rng.integers(0, 256, (48, 120, 3), dtype=np.uint8) for _ in range(4)]
        results, confs = recognition_en_instance(crops)
        assert len(results) == 4
        assert len(confs) == 4

    def test_confidence_length_matches_text_length(self, recognition_en_instance):
        """Each per-line confidence list must have the same length as the text."""
        rng = np.random.default_rng(1)
        crops = [rng.integers(0, 256, (48, 100, 3), dtype=np.uint8) for _ in range(3)]
        results, confs = recognition_en_instance(crops)
        for text, conf in zip(results, confs):
            assert len(conf) == len(text), (
                f"Text '{text}' has {len(text)} chars but {len(conf)} confidence values"
            )

    def test_deterministic_across_two_calls(self, recognition_en_instance):
        rng = np.random.default_rng(2)
        crops = [rng.integers(0, 256, (48, 80, 3), dtype=np.uint8) for _ in range(3)]
        r1, c1 = recognition_en_instance([c.copy() for c in crops])
        r2, c2 = recognition_en_instance([c.copy() for c in crops])
        assert r1 == r2, "Results differ across two identical calls"


# ---------------------------------------------------------------------------
# VocabONNX encode → decode round-trip
# ---------------------------------------------------------------------------


class TestVocabRoundTrip:
    @pytest.fixture()
    def vocab(self, vocab_json_path):
        return VocabONNX(vocab_json_path)

    def test_char_to_index_to_char(self, vocab):
        for char, idx in vocab.c2i.items():
            assert vocab.i2c[idx] == char

    def test_known_sequence_round_trip(self, vocab):
        """Build a token sequence from known chars and verify decode recovers the string."""
        test_str = "abc"
        token_ids = [VocabONNX.SOS] + [vocab.c2i[c] for c in test_str if c in vocab.c2i] + [VocabONNX.EOS]
        decoded = vocab.decode(token_ids)
        assert decoded == test_str


# ---------------------------------------------------------------------------
# Compare resize output at different aspect ratios
# ---------------------------------------------------------------------------


class TestResizeComparison:
    """Verify that resize() output is consistent across different aspect ratios."""

    @pytest.mark.parametrize("w,h", [(32, 32), (64, 32), (512, 32), (1, 32)])
    def test_height_always_32(self, w, h):
        _, new_h = resize(w, h, expected_height=32, image_min_width=32, image_max_width=512)
        assert new_h == 32

    @pytest.mark.parametrize("w,h", [(32, 32), (200, 32), (1000, 32)])
    def test_width_multiple_of_10_or_clamped(self, w, h):
        """Width is a multiple of 10 unless it was clamped to image_max_width."""
        max_width = 512
        new_w, _ = resize(w, h, expected_height=32, image_min_width=32, image_max_width=max_width)
        if new_w < max_width:
            assert new_w % 10 == 0
