"""
Unit tests for VocabONNX and preprocessing helpers in vietocr_onnx.py.
"""

import json
import math
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from vncv.vietocr_onnx import VocabONNX, process_image, process_input, resize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHARS = list("abcABC123 !")


def _make_vocab_file(chars=None):
    if chars is None:
        chars = CHARS
    data = {"chars": chars, "total_vocab_size": 4 + len(chars)}
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(data, tmp, ensure_ascii=False)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# VocabONNX
# ---------------------------------------------------------------------------


class TestVocabONNX:
    @pytest.fixture(autouse=True)
    def vocab(self):
        path = _make_vocab_file()
        self.vocab = VocabONNX(path)
        yield
        os.unlink(path)

    def test_special_token_constants(self):
        assert VocabONNX.PAD == 0
        assert VocabONNX.SOS == 1
        assert VocabONNX.EOS == 2
        assert VocabONNX.MASK == 3

    def test_vocab_size(self):
        assert len(self.vocab) == 4 + len(CHARS)

    def test_char_to_index_offset(self):
        """User characters start at index 4 (after PAD/SOS/EOS/MASK)."""
        assert self.vocab.c2i[CHARS[0]] == 4
        assert self.vocab.c2i[CHARS[-1]] == 4 + len(CHARS) - 1

    def test_index_to_char_round_trip(self):
        for i, c in enumerate(CHARS):
            assert self.vocab.i2c[i + 4] == c

    def test_decode_sos_eos(self):
        """SOS should be stripped and decoding stops at EOS."""
        ids = [VocabONNX.SOS, 4, 5, VocabONNX.EOS, 6]
        result = self.vocab.decode(ids)
        assert result == CHARS[0] + CHARS[1]

    def test_decode_no_eos(self):
        """If there is no EOS, decode all characters after SOS."""
        ids = [VocabONNX.SOS, 4, 5, 6]
        result = self.vocab.decode(ids)
        assert result == CHARS[0] + CHARS[1] + CHARS[2]

    def test_decode_no_sos(self):
        """Sequence without SOS: start from index 0."""
        ids = [4, 5, VocabONNX.EOS]
        result = self.vocab.decode(ids)
        assert result == CHARS[0] + CHARS[1]

    def test_decode_empty(self):
        """Decoding an empty list should give an empty string."""
        assert self.vocab.decode([]) == ""

    def test_batch_decode(self):
        ids_a = [VocabONNX.SOS, 4, VocabONNX.EOS]
        ids_b = [VocabONNX.SOS, 5, VocabONNX.EOS]
        results = self.vocab.batch_decode([ids_a, ids_b])
        assert results[0] == CHARS[0]
        assert results[1] == CHARS[1]

    def test_unknown_index_returns_empty_string(self):
        """An index not in i2c should silently produce an empty string."""
        ids = [VocabONNX.SOS, 9999, VocabONNX.EOS]
        result = self.vocab.decode(ids)
        assert result == ""

    def test_i2c_special_tokens(self):
        assert self.vocab.i2c[0] == "<pad>"
        assert self.vocab.i2c[1] == "<sos>"
        assert self.vocab.i2c[2] == "<eos>"
        assert self.vocab.i2c[3] == "<mask>"


# ---------------------------------------------------------------------------
# resize()
# ---------------------------------------------------------------------------


class TestResizeHelper:
    def test_aspect_ratio_preserved(self):
        """Output width should be proportional to input aspect ratio."""
        w, h = 128, 32
        new_w, new_h = resize(w, h, expected_height=32, image_min_width=32, image_max_width=512)
        expected = math.ceil((32 * w / h) / 10) * 10
        assert new_w == expected
        assert new_h == 32

    def test_respects_min_width(self):
        """Very narrow images should be padded to image_min_width."""
        _, _ = resize(1, 100, 32, 32, 512)
        new_w, _ = resize(1, 100, expected_height=32, image_min_width=64, image_max_width=512)
        assert new_w >= 64

    def test_respects_max_width(self):
        """Very wide images should be clamped to image_max_width."""
        new_w, _ = resize(10000, 32, expected_height=32, image_min_width=32, image_max_width=512)
        assert new_w <= 512

    def test_round_to_10(self):
        """Output width should always be a multiple of 10 unless clamped by min/max."""
        for w in range(1, 300, 7):
            new_w, _ = resize(w, 32, 32, image_min_width=30, image_max_width=512)
            if new_w < 512 and new_w > 30:
                assert new_w % 10 == 0, f"Width {new_w} is not a multiple of 10 (input w={w})"

    def test_square_image(self):
        new_w, new_h = resize(32, 32, 32, 32, 512)
        assert new_h == 32
        assert new_w >= 32


# ---------------------------------------------------------------------------
# process_image()
# ---------------------------------------------------------------------------


class TestProcessImage:
    def _pil(self, w=100, h=32, color=(128, 64, 32)):
        return Image.new("RGB", (w, h), color=color)

    def test_output_shape(self):
        img = self._pil(100, 32)
        arr = process_image(img)
        assert arr.ndim == 3
        c, h, w = arr.shape
        assert c == 3
        assert h == 32

    def test_dtype_float32(self):
        arr = process_image(self._pil())
        assert arr.dtype == np.float32

    def test_range_zero_to_one(self):
        arr = process_image(self._pil())
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_converts_rgba_to_rgb(self):
        """RGBA images must be handled (converted to RGB inside process_image)."""
        rgba = Image.new("RGBA", (80, 32), color=(255, 0, 0, 128))
        arr = process_image(rgba)
        assert arr.shape[0] == 3

    def test_width_clamped_to_max(self):
        wide = self._pil(w=5000, h=32)
        arr = process_image(wide, image_max_width=512)
        assert arr.shape[2] <= 512

    def test_width_padded_to_min(self):
        narrow = self._pil(w=1, h=32)
        arr = process_image(narrow, image_min_width=32)
        assert arr.shape[2] >= 32


# ---------------------------------------------------------------------------
# process_input()
# ---------------------------------------------------------------------------


class TestProcessInput:
    def test_batch_dimension_added(self):
        img = Image.new("RGB", (100, 32))
        arr = process_input(img)
        assert arr.ndim == 4
        assert arr.shape[0] == 1

    def test_shape_1_c_h_w(self):
        img = Image.new("RGB", (80, 32))
        arr = process_input(img)
        b, c, h, w = arr.shape
        assert b == 1
        assert c == 3
        assert h == 32
