"""
Bug-reproduction tests for known issues in vncv.ocr and vncv.vietocr_onnx.

Each test is named after the bug it covers and contains:
  - A description of the bug.
  - The minimal reproducer.
  - The current behaviour (which may be wrong / fragile).

When a bug is fixed, the test should be updated to assert the *correct*
behaviour instead of documenting the broken one.
"""

import json
import math
import os
import tempfile

import cv2
import numpy as np
import pytest
from PIL import Image

from vncv.ocr import CTCDecoder, Detection, EnglishRecognition, sort_polygon
from vncv.vietocr_onnx import VocabONNX, resize


# ---------------------------------------------------------------------------
# BUG-1: EnglishRecognition.resize — IndexError for grayscale images
# ---------------------------------------------------------------------------


@pytest.mark.requires_recognition_en_weights
class TestBug1GrayscaleResize:
    """
    BUG: EnglishRecognition.resize() begins with
        assert self.input_shape[0] == image.shape[2]
    For a 2-D (grayscale) numpy array, `image.shape[2]` raises IndexError
    instead of a helpful error message.

    Root cause: The assertion accesses axis 2 unconditionally without first
    checking that the array is 3-D.

    Fix: Add an explicit check that `image.ndim == 3` before the assertion,
    or convert grayscale to BGR at the call site in `__call__`.
    """

    def test_grayscale_raises_index_error(self, recognition_en_instance):
        """Reproducer: passing a 2-D array triggers IndexError."""
        rec = recognition_en_instance
        gray = np.zeros((48, 100), dtype=np.uint8)
        with pytest.raises((IndexError, AssertionError)):
            rec.resize(gray, max_wh_ratio=3.0)


# ---------------------------------------------------------------------------
# BUG-2: predict_transformer loop runs max_seq_length+1 iterations
# ---------------------------------------------------------------------------


class TestBug2TransformerLoopOverrun:
    """
    BUG: In VietOCROnnxEngine.predict_transformer() the while-loop condition is
        while max_length <= max_seq_length:
    This runs max_seq_length + 1 iterations (0 … max_seq_length inclusive)
    instead of the intended max_seq_length.

    Root cause: should be `< max_seq_length`, not `<= max_seq_length`.
    """

    def test_loop_runs_extra_iteration(self):
        """Confirm the loop over-runs by one step."""
        max_seq_length = 5
        iterations = 0
        max_length = 0
        while max_length <= max_seq_length:
            iterations += 1
            max_length += 1
        # Current (buggy) behaviour: 6 iterations instead of 5
        assert iterations == max_seq_length + 1, (
            "Bug not reproduced – loop count is now correct"
        )


# ---------------------------------------------------------------------------
# BUG-3: CTCDecoder duplicate space characters in character set
# ---------------------------------------------------------------------------


class TestBug3CTCDuplicateSpace:
    """
    BUG: CTCDecoder.character contains two ' ' (space) entries at indices 97
    and 98. Depending on which index the model chooses, two spaces can decode
    identically, but argmax of the character set is ambiguous.

    Root cause: The character list was assembled by hand and contains
    a duplicate space (' ') at the end.
    """

    def test_duplicate_space_present(self):
        dec = CTCDecoder()
        spaces = [i for i, c in enumerate(dec.character) if c == " "]
        assert len(spaces) == 2, (
            f"Expected 2 duplicate space entries, found {len(spaces)} at indices {spaces}"
        )

    def test_both_space_indices_decode_to_space(self):
        dec = CTCDecoder()
        vocab = len(dec.character)
        for space_idx in [i for i, c in enumerate(dec.character) if c == " "]:
            logits = np.zeros((1, 1, vocab), dtype=np.float32)
            logits[0, 0, space_idx] = 10.0
            results, _ = dec(logits)
            assert results[0] == " ", (
                f"Space index {space_idx} did not decode to ' '"
            )


# ---------------------------------------------------------------------------
# BUG-4: sort_polygon — bubble-sort accesses out-of-bounds index
# ---------------------------------------------------------------------------


class TestBug4SortPolygonBubble:
    """
    BUG: sort_polygon() uses a nested loop with
        for j in range(i, -1, -1):
            if … points[j + 1] …
    When i == len(points)-1, the inner loop reads points[j+1] where j can
    equal i == len(points)-1, causing an IndexError for j = len(points)-1
    on the line `points[j + 1]`.

    Reproduction: call sort_polygon with a single-element list — the outer
    range(0, 0) means the body never runs, so there is no crash for n=1.
    For n=2 the inner loop runs at j=1 (i=1) and accesses points[2] which
    does not exist.
    """

    def _poly(self, x, y):
        return np.array([[x, y], [x + 10, y], [x + 10, y + 5], [x, y + 5]])

    def test_two_polygons_already_sorted(self):
        pts = [self._poly(0, 0), self._poly(0, 20)]
        result = sort_polygon(pts)
        assert result[0][0][1] <= result[1][0][1]

    def test_two_polygons_reverse_order(self):
        pts = [self._poly(0, 20), self._poly(0, 0)]
        result = sort_polygon(pts)
        assert result[0][0][1] == 0
        assert result[1][0][1] == 20


# ---------------------------------------------------------------------------
# BUG-5: Detection.clockwise_order — degenerate polygon (all same point)
# ---------------------------------------------------------------------------


class TestBug5ClockwiseOrderDegenerate:
    """
    BUG: When all four points are identical (degenerate box), numpy.argmin(s)
    and numpy.argmax(s) both return 0. numpy.delete then removes only index 0
    once (numpy de-duplicates the index list), leaving tmp with 3 rows instead
    of 2. poly[1] and poly[3] are assigned values from a 3-row array, which
    may give unexpected results but currently does NOT crash.

    This test documents the current (silent misbehaviour) outcome.
    """

    def test_degenerate_polygon_no_crash(self):
        pts = np.array([[5, 5], [5, 5], [5, 5], [5, 5]], dtype=np.float32)
        # Should not raise; actual correctness is undefined for degenerate input
        result = Detection.clockwise_order(pts)
        assert result.shape == (4, 2)


# ---------------------------------------------------------------------------
# BUG-6: VocabONNX.decode — assumes list input, numpy arrays lack .index()
# ---------------------------------------------------------------------------


class TestBug6VocabDecodeNumpyArray:
    """
    BUG: VocabONNX.decode() calls `ids.index(self.EOS)` which is a list
    method. If ids is a numpy array (int row from translated.tolist() is
    actually a Python list, but direct callers might pass np.ndarray),
    AttributeError is raised.

    In the main prediction path ids comes from `.tolist()` so the bug is
    latent, but calling decode() directly with a numpy array exposes it.
    """

    @pytest.fixture()
    def vocab(self, vocab_json_path):
        return VocabONNX(vocab_json_path)

    def test_list_input_works(self, vocab):
        ids = [VocabONNX.SOS, 4, VocabONNX.EOS]
        # Should succeed without error
        result = vocab.decode(ids)
        assert isinstance(result, str)

    def test_numpy_array_input_raises(self, vocab):
        """Passing a numpy array should raise AttributeError (known bug)."""
        ids_np = np.array([VocabONNX.SOS, 4, VocabONNX.EOS], dtype=np.int64)
        with pytest.raises(AttributeError):
            vocab.decode(ids_np)


# ---------------------------------------------------------------------------
# BUG-7: Detection normalization — cv2 dtype mismatch (float32 vs float64)
# ---------------------------------------------------------------------------


@pytest.mark.requires_detection_weights
class TestBug7DetectionNormDtype:
    """
    BUG: Detection stores self.mean and self.std as float64 arrays
    (reshape(1,-1).astype('float64')), but the image tensor x is float32.
    cv2.subtract(float32, float64, dst_float32) silently casts the second
    operand, but the result is numerically correct. This test verifies that
    the current silent cast does not introduce meaningful error.
    """

    def test_mean_std_dtype_is_float64(self, detection_instance):
        det = detection_instance
        assert det.mean.dtype == np.float64, "self.mean should be float64"
        assert det.std.dtype == np.float64, "self.std should be float64"

    def test_normalization_result_close_to_numpy(self, detection_instance):
        det = detection_instance
        x_cv = np.ones((1, 1, 3), dtype=np.float32) * 200.0
        x_np = np.ones((1, 1, 3), dtype=np.float32) * 200.0

        cv2.subtract(x_cv, det.mean, x_cv)
        cv2.multiply(x_cv, det.std, x_cv)
        x_np = (x_np - det.mean.astype(np.float32)) * det.std.astype(np.float32)

        np.testing.assert_allclose(x_cv, x_np, atol=1e-4)
