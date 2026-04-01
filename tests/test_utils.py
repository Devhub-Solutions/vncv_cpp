"""
Unit tests for utility functions: sort_polygon and crop_image.
"""

import numpy as np
import pytest

from vncv.ocr import crop_image, sort_polygon


# ---------------------------------------------------------------------------
# sort_polygon
# ---------------------------------------------------------------------------


class TestSortPolygon:
    def _poly(self, x0, y0, w=20, h=10):
        """Helper: create a simple axis-aligned polygon as numpy array."""
        return np.array(
            [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]],
            dtype=np.int32,
        )

    def test_top_to_bottom_order(self):
        """Polygons should be ordered top-to-bottom (ascending y)."""
        p_bottom = self._poly(0, 50)
        p_middle = self._poly(0, 25)
        p_top = self._poly(0, 0)
        result = sort_polygon([p_bottom, p_middle, p_top])
        y_values = [p[0][1] for p in result]
        assert y_values == sorted(y_values), "Polygons not sorted by ascending y"

    def test_same_y_left_to_right(self):
        """Polygons at the same vertical position should be ordered left-to-right."""
        p_right = self._poly(50, 10)
        p_left = self._poly(0, 10)
        result = sort_polygon([p_right, p_left])
        assert result[0][0][0] < result[1][0][0], "Left polygon should come first"

    def test_single_polygon(self):
        """A single polygon should be returned unchanged."""
        p = self._poly(5, 5)
        result = sort_polygon([p])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], p)

    def test_empty_list(self):
        """Empty input should return an empty list."""
        result = sort_polygon([])
        assert result == []

    def test_returns_list(self):
        """sort_polygon should always return a list."""
        result = sort_polygon([self._poly(0, 0)])
        assert isinstance(result, list)

    def test_mixed_y_positions(self):
        """Mixed layout: top-left, bottom-right, top-right → sorted correctly."""
        polygons = [
            self._poly(100, 80),
            self._poly(0, 0),
            self._poly(100, 0),
        ]
        result = sort_polygon(polygons)
        y0_vals = [p[0][1] for p in result]
        # Two polygons at y=0, one at y=80
        assert y0_vals[2] == 80
        assert y0_vals[0] == 0
        assert y0_vals[1] == 0
        # Among y=0 polygons, left-to-right
        assert result[0][0][0] <= result[1][0][0]

    def test_tolerance_within_10px(self):
        """Polygons whose y-values differ by less than 10 are considered same row."""
        p_left = self._poly(0, 0)
        p_right = self._poly(50, 8)  # diff = 8 < 10 → same row
        result = sort_polygon([p_right, p_left])
        # After bubble sort, p_left (x=0) should precede p_right (x=50)
        assert result[0][0][0] == 0


# ---------------------------------------------------------------------------
# crop_image
# ---------------------------------------------------------------------------


class TestCropImage:
    def _frame(self):
        rng = np.random.default_rng(7)
        return rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)

    def test_axis_aligned_rectangle(self):
        """A horizontal rectangle should be cropped to its bounding size."""
        frame = self._frame()
        pts = np.float32([[10, 10], [90, 10], [90, 30], [10, 30]])
        cropped = crop_image(frame, pts)
        assert cropped.shape[2] == 3, "Should be BGR image"
        assert cropped.shape[0] > 0 and cropped.shape[1] > 0

    def test_requires_exactly_four_points(self):
        """crop_image must raise AssertionError if not exactly 4 points."""
        frame = self._frame()
        with pytest.raises(AssertionError):
            crop_image(frame, np.float32([[0, 0], [10, 0], [10, 10]]))

    def test_tall_region_is_rotated(self):
        """If height / width >= 1.5 the image should be rotated 90°."""
        frame = self._frame()
        # A thin vertical strip: width=10, height=50
        pts = np.float32([[10, 10], [20, 10], [20, 60], [10, 60]])
        cropped = crop_image(frame, pts)
        # After rotation the previously-tall dimension becomes the width
        h, w = cropped.shape[:2]
        assert w >= h, "Tall image should be rotated so width >= height"

    def test_wide_region_not_rotated(self):
        """A wide region (width >> height) should not be rotated."""
        frame = self._frame()
        pts = np.float32([[10, 10], [100, 10], [100, 20], [10, 20]])
        cropped = crop_image(frame, pts)
        h, w = cropped.shape[:2]
        assert w >= h, "Wide image should remain landscape"

    def test_output_dtype_uint8(self):
        """Output should be a uint8 array."""
        frame = self._frame()
        pts = np.float32([[0, 0], [50, 0], [50, 20], [0, 20]])
        cropped = crop_image(frame, pts)
        assert cropped.dtype == np.uint8
