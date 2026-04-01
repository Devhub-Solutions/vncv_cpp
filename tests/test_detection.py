"""
Tests for the Detection class – both static helpers and full ONNX inference.
"""

import numpy as np
import pytest

from vncv.ocr import Detection


# ---------------------------------------------------------------------------
# Static / pure-Python helpers (no ONNX required)
# ---------------------------------------------------------------------------


class TestClockwiseOrder:
    def test_already_ordered(self):
        pts = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32)
        result = Detection.clockwise_order(pts)
        np.testing.assert_array_equal(result[0], [0, 0])   # top-left
        np.testing.assert_array_equal(result[2], [10, 5])  # bottom-right

    def test_shuffled_points(self):
        """Regardless of input order, poly[0] should be min-sum corner."""
        pts = np.array([[10, 5], [0, 5], [10, 0], [0, 0]], dtype=np.float32)
        result = Detection.clockwise_order(pts)
        np.testing.assert_array_equal(result[0], [0, 0])
        np.testing.assert_array_equal(result[2], [10, 5])

    def test_output_shape(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        result = Detection.clockwise_order(pts)
        assert result.shape == (4, 2)

    def test_float32_output(self):
        pts = np.array([[0, 0], [5, 0], [5, 3], [0, 3]], dtype=np.float32)
        result = Detection.clockwise_order(pts)
        assert result.dtype == np.float32


class TestClip:
    def test_clips_to_bounds(self):
        pts = np.array([[300, 200], [-5, 600]], dtype=np.float32)
        result = Detection.clip(pts, h=200, w=300)
        assert result[0, 0] <= 299
        assert result[0, 1] <= 199
        assert result[1, 0] >= 0
        assert result[1, 1] <= 199

    def test_in_bounds_unchanged(self):
        pts = np.array([[50, 50]], dtype=np.float32)
        result = Detection.clip(pts.copy(), h=200, w=300)
        np.testing.assert_array_equal(result, [[50, 50]])

    def test_negative_clipped_to_zero(self):
        pts = np.array([[-10, -20]], dtype=np.float32)
        result = Detection.clip(pts, h=100, w=100)
        assert result[0, 0] == 0
        assert result[0, 1] == 0


class TestBoxScore:
    def test_high_score_region(self):
        bitmap = np.ones((100, 100), dtype=np.float32) * 0.9
        contour = np.array([[[10, 10]], [[50, 10]], [[50, 30]], [[10, 30]]])
        score = Detection.box_score(bitmap, contour)
        assert abs(score - 0.9) < 0.01

    def test_zero_region(self):
        bitmap = np.zeros((100, 100), dtype=np.float32)
        contour = np.array([[[5, 5]], [[20, 5]], [[20, 20]], [[5, 20]]])
        score = Detection.box_score(bitmap, contour)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_score_in_range(self):
        rng = np.random.default_rng(1)
        bitmap = rng.random((50, 50)).astype(np.float32)
        contour = np.array([[[5, 5]], [[40, 5]], [[40, 40]], [[5, 40]]])
        score = Detection.box_score(bitmap, contour)
        assert 0.0 <= score <= 1.0


class TestGetMinBoxes:
    def test_returns_four_corners_and_min_side(self):
        contour = np.array([[[0, 0]], [[10, 0]], [[10, 5]], [[0, 5]]])
        box, min_side = Detection.get_min_boxes(contour)
        assert len(box) == 4
        assert min_side > 0

    def test_min_side_value(self):
        """min_side should equal the shorter dimension of the bounding box."""
        contour = np.array([[[0, 0]], [[20, 0]], [[20, 10]], [[0, 10]]])
        _, min_side = Detection.get_min_boxes(contour)
        assert min_side == pytest.approx(10.0, abs=1.0)


class TestDetectionResize:
    @pytest.fixture(autouse=True)
    def det(self, detection_instance):
        self.det = detection_instance

    def test_output_divisible_by_32(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = self.det.resize(img)
        h, w = resized.shape[:2]
        assert h % 32 == 0
        assert w % 32 == 0

    def test_large_image_downscaled(self):
        """Images larger than max_size should be downscaled."""
        img = np.zeros((2000, 3000, 3), dtype=np.uint8)
        resized = self.det.resize(img)
        assert max(resized.shape[:2]) <= self.det.max_size + 32

    def test_small_image_min_32(self):
        """Output dimensions should be at least 32."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        resized = self.det.resize(img)
        assert resized.shape[0] >= 32
        assert resized.shape[1] >= 32


class TestDetectionZeroPad:
    def test_pads_small_image(self):
        from vncv.ocr import Detection
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        padded = Detection.zero_pad(img)
        assert padded.shape == (32, 32, 3)

    def test_does_not_shrink_larger_image(self):
        from vncv.ocr import Detection
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        padded = Detection.zero_pad(img)
        assert padded.shape == (64, 64, 3)


class TestFilterPolygon:
    @pytest.fixture(autouse=True)
    def det(self, detection_instance):
        self.det = detection_instance

    def test_removes_tiny_box(self):
        """Boxes where w<=3 or h<=3 should be discarded."""
        tiny = [np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.int32)]
        result = self.det.filter_polygon(tiny, (100, 100))
        assert len(result) == 0

    def test_keeps_normal_box(self):
        normal = [np.array([[0, 0], [20, 0], [20, 10], [0, 10]], dtype=np.int32)]
        result = self.det.filter_polygon(normal, (100, 100))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Full ONNX inference (requires weights)
# ---------------------------------------------------------------------------


@pytest.mark.requires_detection_weights
class TestDetectionInference:
    @pytest.fixture(autouse=True)
    def det(self, detection_instance):
        self.det = detection_instance

    def test_returns_array_or_empty(self, small_bgr_image):
        """Detection on a random image should return an array-like (may be empty)."""
        result = self.det(small_bgr_image)
        assert hasattr(result, "__len__")

    def test_tiny_image_no_crash(self):
        """Tiny images (< 64 total pixels) should not raise."""
        tiny = np.zeros((5, 5, 3), dtype=np.uint8)
        self.det(tiny)  # should not raise

    def test_each_box_has_four_corners(self, small_bgr_image):
        """Every detected box must have exactly 4 corners."""
        result = self.det(small_bgr_image)
        for box in result:
            assert box.shape == (4, 2), f"Box shape {box.shape} != (4, 2)"

    def test_deterministic(self, small_bgr_image):
        """Two calls with the same image must return the same boxes."""
        r1 = self.det(small_bgr_image)
        r2 = self.det(small_bgr_image)
        assert len(r1) == len(r2)
        for b1, b2 in zip(r1, r2):
            np.testing.assert_array_equal(b1, b2)
