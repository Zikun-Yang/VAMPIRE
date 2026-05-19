"""Unit tests for pure utility functions in vampire._anno."""

import numpy as np
import pytest

from vampire._anno import (
    canonicalize_motif_str,
    rc,
    calculate_phase_difference,
    calculate_edit_distance_between_motifs,
    split_ops,
)


class TestCanonicalizeMotifStr:
    """Tests for canonicalize_motif_str (Booth algorithm)."""

    @pytest.mark.parametrize(
        "motif,expected",
        [
            ("A", "A"),
            ("AT", "AT"),
            ("TA", "AT"),
            ("AAA", "AAA"),
            ("TGGAT", "ATTGG"),
            ("ATGATG", "ATGATG"),
            ("TGATGA", "ATGATG"),
            ("CCAGCCAG", "AGCCAGCC"),
        ],
    )
    def test_canonicalize(self, motif, expected):
        assert canonicalize_motif_str(motif) == expected

    def test_empty(self):
        assert canonicalize_motif_str("") == ""


class TestRC:
    """Tests for reverse complement."""

    @pytest.mark.parametrize(
        "seq,expected",
        [
            ("A", "T"),
            ("AT", "AT"),
            ("GC", "GC"),
            ("TGGAT", "ATCCA"),
            ("ATCCA", "TGGAT"),
            ("NNN", "NNN"),
            ("ACGT", "ACGT"),
        ],
    )
    def test_rc(self, seq, expected):
        assert rc(seq) == expected


class TestCalculatePhaseDifference:
    """Tests for calculate_phase_difference."""

    @pytest.mark.parametrize(
        "m1,m2,expected",
        [
            ("ATGTTT", "TGTTTA", 5),
            ("ATGAGG", "GAGGAT", 4),
            ("TGGAT", "ATTGG", 2),
            ("ATTC", "ATTCG", 0),
        ],
    )
    def test_phase(self, m1, m2, expected):
        result = calculate_phase_difference(m1, m2)
        assert result == expected


class TestEditDistanceBetweenMotifs:
    """Tests for calculate_edit_distance_between_motifs."""

    def test_identical(self):
        m = np.array([0, 1, 2, 3])  # A, T, G, C
        assert calculate_edit_distance_between_motifs(m, m) == 0

    def test_one_mismatch(self):
        m1 = np.array([0, 1, 2, 3])  # ATGC
        m2 = np.array([0, 1, 2, 0])  # ATGA
        assert calculate_edit_distance_between_motifs(m1, m2) == 1

    def test_rotation_invariant_distance(self):
        # TGGAT vs ATTGG are rotations, edit distance should be 0
        from vampire._utils import encode_seq_to_array

        m1 = encode_seq_to_array("TGGAT")
        m2 = encode_seq_to_array("ATTGG")
        assert calculate_edit_distance_between_motifs(m1, m2) == 0


class TestSplitOps:
    """Tests for split_ops (CIGAR block splitting)."""

    def test_perfect_match(self):
        ops = ["=", "=", "=", "=", "="]
        result = split_ops(ops, 5, 2)
        assert len(result) == 1
        start, end, distance, score, cigar = result[0]
        assert start == 0
        assert end == 4
        assert distance == 0
        assert cigar == "5=/"

    def test_with_mismatch(self):
        ops = ["=", "=", "X", "=", "="]
        result = split_ops(ops, 5, 2)
        assert len(result) == 1
        assert result[0][2] == 1  # distance

    def test_exceeds_max_distance(self):
        ops = ["X", "X", "X", "X", "X"]
        result = split_ops(ops, 5, 2)
        assert len(result) == 0  # all blocks exceed max_distance

    def test_multi_blocks(self):
        ops = ["=", "=", "=", "=", "=", "=", "=", "=", "=", "="]
        result = split_ops(ops, 5, 0)
        assert len(result) == 2
        assert result[0][0] == 0
        assert result[0][1] == 4
        assert result[1][0] == 5
        assert result[1][1] == 9
