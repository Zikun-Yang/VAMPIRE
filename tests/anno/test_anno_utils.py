"""Unit tests for pure utility functions in vampire._anno."""

import numpy as np
import polars as pl
import pytest

from vampire._anno import (
    canonicalize_motif_str,
    rc,
    calculate_phase_difference,
    calculate_edit_distance_between_motifs,
    split_ops,
    make_raw,
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


class TestMakeRaw:
    """Tests for make_raw."""

    def _make_anno_df(self, rows):
        return pl.DataFrame(
            rows,
            schema={
                "chrom": pl.Utf8,
                "length": pl.Int64,
                "start": pl.Int64,
                "end": pl.Int64,
                "motif": pl.Int64,
                "orientation": pl.Utf8,
                "sequence": pl.Utf8,
                "score": pl.Int64,
                "cigar": pl.Utf8,
            },
        )

    def _make_motif_df(self, rows):
        return pl.DataFrame(
            rows,
            schema={"id": pl.Int64, "motif": pl.Utf8, "copyNumber": pl.Float64, "label": pl.Utf8},
        )

    def test_canonicalizes_and_deduplicates_rotations(self):
        motif_df = self._make_motif_df(
            [{"id": 0, "motif": "ATTGG", "copyNumber": 0.0, "label": "A"}]
        )
        anno_df = self._make_anno_df([
            {"chrom": "s1", "length": 10, "start": 0, "end": 4, "motif": 0, "orientation": "+", "sequence": "ATTGG", "score": 10, "cigar": "5=/"},
            {"chrom": "s1", "length": 10, "start": 5, "end": 9, "motif": 0, "orientation": "+", "sequence": "GGATT", "score": 10, "cigar": "5=/"},
        ])
        anno_out, concise_out, motif_out, dist_out = make_raw(
            anno_df, pl.DataFrame(), motif_df, pl.DataFrame(), 2, 7, 7, 7
        )

        assert motif_out.shape[0] == 1
        assert motif_out.item(0, "motif") == "ATTGG"
        assert motif_out.item(0, "copyNumber") == 2.0
        assert motif_out.item(0, "label") == "A"
        assert anno_out["cigar"].to_list() == ["5=/", "5=/"]
        assert anno_out["copyNumber"].to_list() == [1.0, 1.0]

    def test_reverse_complement_is_normalized(self):
        motif_df = self._make_motif_df(
            [{"id": 0, "motif": "ATTGG", "copyNumber": 0.0, "label": "B"}]
        )
        # rc("CCAAT") == "ATTGG"
        anno_df = self._make_anno_df([
            {"chrom": "s2", "length": 5, "start": 0, "end": 4, "motif": 0, "orientation": "-", "sequence": "CCAAT", "score": 10, "cigar": "5=/"},
        ])
        anno_out, concise_out, motif_out, dist_out = make_raw(
            anno_df, pl.DataFrame(), motif_df, pl.DataFrame(), 2, 7, 7, 7
        )

        assert motif_out.item(0, "motif") == "ATTGG"
        assert motif_out.item(0, "label") == "B"

    def test_gap_rows_converted_to_raw_motifs(self):
        motif_df = self._make_motif_df(
            [{"id": 0, "motif": "ATTGG", "copyNumber": 0.0, "label": "C"}]
        )
        anno_df = self._make_anno_df([
            {"chrom": "s3", "length": 8, "start": 0, "end": 4, "motif": 0, "orientation": "+", "sequence": "ATTGG", "score": 10, "cigar": "5=/"},
            {"chrom": "s3", "length": 8, "start": 5, "end": 7, "motif": None, "orientation": None, "sequence": "AAA", "score": -21, "cigar": "3N"},
        ])
        anno_out, concise_out, motif_out, dist_out = make_raw(
            anno_df, pl.DataFrame(), motif_df, pl.DataFrame(), 2, 7, 7, 7
        )

        # The gap row is converted to a raw motif block.
        gap_row = anno_out.filter(pl.col("sequence") == "AAA").row(0, named=True)
        assert gap_row["cigar"] == "3=/"
        assert gap_row["copyNumber"] == 1.0
        assert gap_row["motif"] is not None
        # The converted gap appears in the motif catalog with the skipped label.
        skipped = motif_out.filter(pl.col("label") == "skipped")
        assert skipped.shape[0] == 1
        assert skipped.item(0, "motif") == "AAA"
