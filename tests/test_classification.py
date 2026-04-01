"""
Tests for the Classification class.
"""

import cv2
import numpy as np
import pytest

from vncv.ocr import Classification


class TestClassificationResize:
    """Test the static resize pre-processing method."""

    TARGET_C, TARGET_H, TARGET_W = 3, 48, 192

    @pytest.fixture(autouse=True)
    def cls(self, classification_instance):
        self.cls = classification_instance

    def test_output_shape(self):
        img = np.zeros((32, 128, 3), dtype=np.uint8)
        result = Classification.resize(img)
        assert result.shape == (self.TARGET_C, self.TARGET_H, self.TARGET_W)

    def test_output_dtype_float32(self):
        img = np.zeros((32, 64, 3), dtype=np.uint8)
        result = Classification.resize(img)
        assert result.dtype == np.float32

    def test_normalised_range(self):
        """After (x - 0.5) / 0.5 normalization, values should be in [-1, 1]."""
        img = np.zeros((48, 192, 3), dtype=np.uint8)
        result = Classification.resize(img)
        assert result.min() >= -1.0 - 1e-5
        assert result.max() <= 1.0 + 1e-5

    def test_wide_image_clamped_to_max_width(self):
        """Images wider than target width should be resized to target width."""
        wide = np.zeros((48, 5000, 3), dtype=np.uint8)
        result = Classification.resize(wide)
        assert result.shape == (self.TARGET_C, self.TARGET_H, self.TARGET_W)

    def test_narrow_image_padded_with_zeros(self):
        """Narrow images should be right-padded with zeros (not normalised fill)."""
        narrow = np.ones((48, 10, 3), dtype=np.uint8) * 255
        result = Classification.resize(narrow)
        resized_w = max(1, int(np.ceil(self.TARGET_H * (10 / 48))))
        if resized_w < self.TARGET_W:
            np.testing.assert_array_equal(
                result[:, :, resized_w:], 0.0
            )


@pytest.mark.requires_classification_weights
class TestClassificationInference:
    @pytest.fixture(autouse=True)
    def cls(self, classification_instance):
        self.cls = classification_instance

    def _random_crops(self, n=3, h=32, w=80):
        rng = np.random.default_rng(42)
        return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]

    def test_returns_images_and_labels(self):
        crops = self._random_crops(3)
        images_out, labels = self.cls(crops)
        assert len(images_out) == 3
        assert len(labels) == 3

    def test_labels_are_valid(self):
        """Each label should be one of '0' or '180'."""
        crops = self._random_crops(2)
        _, labels = self.cls(crops)
        for label, score in labels:
            assert label in ("0", "180"), f"Unexpected label '{label}'"

    def test_scores_in_range(self):
        crops = self._random_crops(2)
        _, labels = self.cls(crops)
        for _, score in labels:
            assert 0.0 <= float(score) <= 1.0

    def test_180_rotation_applied(self):
        """If a crop is classified as '180' with high confidence, it should be rotated."""
        rng = np.random.default_rng(99)
        crop = rng.integers(0, 256, (32, 80, 3), dtype=np.uint8)
        original = crop.copy()
        images_out, labels = self.cls([crop])
        label, score = labels[0]
        if label == "180" and score > self.cls.threshold:
            # Image should have been rotated 90° (cv2.rotate code 1 = ROTATE_90_COUNTERCLOCKWISE)
            assert images_out[0].shape != (32, 80, 3) or not np.array_equal(
                images_out[0], original
            ), "Expected rotation but image was unchanged"

    def test_deterministic(self):
        crops = self._random_crops(2)
        _, labels1 = self.cls([c.copy() for c in crops])
        _, labels2 = self.cls([c.copy() for c in crops])
        for (l1, s1), (l2, s2) in zip(labels1, labels2):
            assert l1 == l2
            assert abs(s1 - s2) < 1e-5

    def test_single_image(self):
        crop = self._random_crops(1)
        images_out, labels = self.cls(crop)
        assert len(labels) == 1

    def test_empty_list(self):
        """Passing an empty list should return two empty lists without error."""
        images_out, labels = self.cls([])
        assert images_out == []
        assert labels == []
