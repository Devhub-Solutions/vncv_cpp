"""
Tests for EnglishRecognition (CTC-based ONNX recognition).
"""

import numpy as np
import pytest

from vncv.ocr import EnglishRecognition


class TestEnglishRecognitionResize:
    INPUT_H, INPUT_W, INPUT_C = 48, 320, 3

    @pytest.fixture(autouse=True)
    def rec(self, recognition_en_instance):
        self.rec = recognition_en_instance

    def test_output_shape_channels_first(self):
        img = np.zeros((self.INPUT_H, 100, 3), dtype=np.uint8)
        result = self.rec.resize(img, max_wh_ratio=100 / self.INPUT_H)
        assert result.ndim == 3
        assert result.shape[0] == self.INPUT_C
        assert result.shape[1] == self.INPUT_H

    def test_output_dtype_float32(self):
        img = np.zeros((self.INPUT_H, 100, 3), dtype=np.uint8)
        result = self.rec.resize(img, max_wh_ratio=5.0)
        assert result.dtype == np.float32

    def test_normalized_range(self):
        """After (x/255 - 0.5) / 0.5 the range should be in [-1, 1]."""
        img = np.zeros((self.INPUT_H, 50, 3), dtype=np.uint8)
        result = self.rec.resize(img, max_wh_ratio=3.0)
        assert result.min() >= -1.0 - 1e-5
        assert result.max() <= 1.0 + 1e-5

    def test_padding_zeros_for_narrow_image(self):
        """Narrow images should be right-padded with zeros."""
        narrow = np.ones((self.INPUT_H, 10, 3), dtype=np.uint8) * 128
        result = self.rec.resize(narrow, max_wh_ratio=5.0)
        resized_w = int(np.ceil(self.INPUT_H * (10 / self.INPUT_H)))
        if resized_w < result.shape[2]:
            np.testing.assert_array_equal(
                result[:, :, resized_w:], 0.0
            )

    def test_wide_image_capped_at_max_width(self):
        """The output width should not exceed max_wh_ratio * input_h."""
        wide = np.zeros((self.INPUT_H, 5000, 3), dtype=np.uint8)
        max_wh = 10.0
        result = self.rec.resize(wide, max_wh_ratio=max_wh)
        assert result.shape[2] <= int(self.INPUT_H * max_wh) + 1


@pytest.mark.requires_recognition_en_weights
class TestEnglishRecognitionInference:
    @pytest.fixture(autouse=True)
    def rec(self, recognition_en_instance):
        self.rec = recognition_en_instance

    def _crops(self, n=3):
        rng = np.random.default_rng(123)
        return [rng.integers(0, 256, (48, 120, 3), dtype=np.uint8) for _ in range(n)]

    def test_returns_results_and_confidences(self):
        crops = self._crops(3)
        results, confs = self.rec(crops)
        assert len(results) == 3
        assert len(confs) == 3

    def test_results_are_strings(self):
        results, _ = self.rec(self._crops(2))
        for r in results:
            assert isinstance(r, str)

    def test_confidence_list_per_character(self):
        """Confidence for each text line is a list of per-character probabilities."""
        results, confs = self.rec(self._crops(2))
        for r, c in zip(results, confs):
            assert len(c) == len(r), "Confidence count != character count"

    def test_deterministic(self):
        crops = self._crops(2)
        r1, c1 = self.rec([img.copy() for img in crops])
        r2, c2 = self.rec([img.copy() for img in crops])
        assert r1 == r2

    def test_empty_list(self):
        results, confs = self.rec([])
        assert results == []
        assert confs == []

    def test_single_crop(self):
        results, confs = self.rec(self._crops(1))
        assert len(results) == 1
        assert len(confs) == 1

    def test_batch_larger_than_batch_size(self):
        """Batches larger than the internal batch size (6) should still work."""
        crops = self._crops(9)
        results, confs = self.rec(crops)
        assert len(results) == 9
