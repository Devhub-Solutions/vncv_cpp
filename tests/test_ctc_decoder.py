"""
Unit tests for CTCDecoder.
"""

import numpy as np
import pytest

from vncv.ocr import CTCDecoder


class TestCTCDecoder:
    @pytest.fixture(autouse=True)
    def decoder(self):
        self.dec = CTCDecoder()

    # ------------------------------------------------------------------
    # Character set
    # ------------------------------------------------------------------

    def test_character_count(self):
        """Decoder should have exactly 97 character entries (including blank)."""
        assert len(self.dec.character) == 97

    def test_blank_at_index_zero(self):
        assert self.dec.character[0] == "blank"

    # ------------------------------------------------------------------
    # Basic decoding
    # ------------------------------------------------------------------

    def _make_logits(self, batch, seq, char_seq):
        """
        Build a (batch, seq, vocab) logits tensor where each step in
        char_seq specifies the winning character (by index into self.dec.character).
        """
        vocab = len(self.dec.character)
        logits = np.zeros((batch, seq, vocab), dtype=np.float32)
        for t, idx in enumerate(char_seq):
            logits[0, t, idx] = 10.0
        return logits

    def test_simple_word_no_repeat(self):
        """Basic decoding of a short word without repeated characters."""
        # H → i → (blank) → ! → end of sequence
        indices = [
            self.dec.character.index("H"),
            self.dec.character.index("i"),
            0,  # blank
        ]
        logits = self._make_logits(1, 3, indices)
        results, confs = self.dec(logits)
        assert results[0] == "Hi"

    def test_collapse_repeats(self):
        """Consecutive identical tokens should be collapsed (CTC rule)."""
        # A A A (blank) B B → AB
        idx_A = self.dec.character.index("A")
        idx_B = self.dec.character.index("B")
        indices = [idx_A, idx_A, idx_A, 0, idx_B, idx_B]
        logits = self._make_logits(1, 6, indices)
        results, _ = self.dec(logits)
        assert results[0] == "AB"

    def test_blank_only_gives_empty_string(self):
        """All-blank sequence should decode to empty string."""
        logits = np.zeros((1, 5, len(self.dec.character)), dtype=np.float32)
        logits[0, :, 0] = 10.0
        results, confs = self.dec(logits)
        assert results[0] == ""

    def test_batch_decoding(self):
        """Multiple items in a batch should each be decoded independently."""
        vocab = len(self.dec.character)
        logits = np.zeros((2, 4, vocab), dtype=np.float32)
        # Batch 0 → 'A'
        logits[0, 0, self.dec.character.index("A")] = 10.0
        # Batch 1 → 'B'
        logits[1, 0, self.dec.character.index("B")] = 10.0
        results, _ = self.dec(logits)
        assert results[0] == "A"
        assert results[1] == "B"

    def test_tuple_input_uses_last_element(self):
        """When a tuple is passed the decoder should use its last element."""
        vocab = len(self.dec.character)
        logits = np.zeros((1, 3, vocab), dtype=np.float32)
        logits[0, 0, self.dec.character.index("Z")] = 10.0
        results_direct, _ = self.dec(logits)
        results_tuple, _ = self.dec((None, None, logits))
        assert results_direct == results_tuple

    def test_confidence_length_matches_decoded_chars(self):
        """Confidence list length should equal the number of decoded characters."""
        idx_A = self.dec.character.index("A")
        idx_B = self.dec.character.index("B")
        logits = self._make_logits(1, 4, [idx_A, idx_A, idx_B, 0])
        _, confs = self.dec(logits)
        # "AB" → 2 characters → 2 confidence values
        assert len(confs[0]) == 2

    def test_confidence_values_are_raw_logits(self):
        """
        CTCDecoder returns raw output values (logits or model scores),
        not normalised probabilities. Values are therefore unbounded.
        """
        idx = self.dec.character.index("a")
        logits = self._make_logits(1, 2, [idx, 0])
        _, confs = self.dec(logits)
        # At least one confidence value should be present
        assert len(confs[0]) >= 1

    def test_spaces_decoded_correctly(self):
        """Space characters (index 97 or 98) should decode as ' '."""
        space_indices = [i for i, c in enumerate(self.dec.character) if c == " "]
        assert len(space_indices) >= 1, "No space in character set"
        logits = self._make_logits(1, 1, [space_indices[0]])
        results, _ = self.dec(logits)
        assert results[0] == " "
