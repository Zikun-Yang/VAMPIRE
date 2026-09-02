"""Tests for rectangular MSA output in vampire.anno.tl._align.

Regression tests for the bug where iterative refinement in ``_msa_core``
re-aligned each raw sequence to the consensus independently and discarded
the consensus side of each pairwise alignment.  Samples with private
insertions relative to the consensus (e.g. extra VNTR copies) then
produced longer rows, and the trailing columns of those rows were
silently truncated when the profile was rebuilt.  The resulting ragged
"motif array" later crashed ``_build_haplotype_consensus`` with an
``IndexError``.
"""

import numpy as np
import pandas as pd
import pytest
import anndata as ad

from vampire.anno.tl._align import (
    _merge_pairwise_alignments,
    _msa_core,
    sample_msa,
)
from vampire.anno.tl._haplotype import _build_haplotype_consensus


def _toy_sub_matrix(n_motifs: int = 2) -> np.ndarray:
    """Simple substitution matrix: match=2, mismatch=-3."""
    sub = np.full((n_motifs, n_motifs), -3, dtype=int)
    np.fill_diagonal(sub, 2)
    return sub


def _strip_gaps(row):
    return [m for m in row if m != "-"]


class TestMergePairwiseAlignments:
    """Unit tests for the star-merge helper."""

    def test_no_insertions_passthrough(self):
        pairs = {
            "s1": (["0", "1", "0"], ["0", "1", "0"]),
            "s2": (["0", "-", "0"], ["0", "1", "0"]),
        }
        rows = _merge_pairwise_alignments(pairs, n_cons=3)
        assert rows["s1"] == ["0", "1", "0"]
        assert rows["s2"] == ["0", "-", "0"]

    def test_insertion_becomes_shared_column(self):
        # s2 has two extra copies inserted after consensus column 3.
        pairs = {
            "s1": (["0", "0", "-"], ["0", "0", "0"]),
            "s2": (["0", "0", "0", "0", "0"], ["0", "0", "0", "-", "-"]),
        }
        rows = _merge_pairwise_alignments(pairs, n_cons=3)
        # Two shared insertion columns appended at boundary 3.
        assert rows["s2"] == ["0", "0", "0", "0", "0"]
        assert rows["s1"] == ["0", "0", "-", "-", "-"]
        assert len(rows["s1"]) == len(rows["s2"]) == 5

    def test_leading_and_trailing_insertions(self):
        pairs = {
            "s1": (["1", "0", "0", "1"], ["-", "0", "0", "-"]),
            "s2": (["0", "0"], ["0", "0"]),
        }
        rows = _merge_pairwise_alignments(pairs, n_cons=2)
        # boundary 0 width 1, boundary 2 width 1
        assert rows["s1"] == ["1", "0", "0", "1"]
        assert rows["s2"] == ["-", "0", "0", "-"]

    def test_different_run_widths_same_boundary(self):
        pairs = {
            "s1": (["0", "1", "1", "0"], ["0", "-", "-", "0"]),
            "s2": (["0", "1", "0"], ["0", "-", "0"]),
            "s3": (["0", "0"], ["0", "0"]),
        }
        rows = _merge_pairwise_alignments(pairs, n_cons=2)
        assert rows["s1"] == ["0", "1", "1", "0"]
        assert rows["s2"] == ["0", "1", "-", "0"]
        assert rows["s3"] == ["0", "-", "-", "0"]
        assert len({len(r) for r in rows.values()}) == 1

    def test_consensus_count_mismatch_raises(self):
        pairs = {"s1": (["0", "0"], ["0", "-"])}  # only 1 non-gap, n_cons=2
        with pytest.raises(AssertionError):
            _merge_pairwise_alignments(pairs, n_cons=2)


class TestMsaCoreRectangular:
    """_msa_core must always return a rectangular alignment."""

    @staticmethod
    def _assert_rectangular(result: dict):
        lengths = {len(v) for v in result.values()}
        assert len(lengths) == 1, f"ragged rows: {lengths}"

    def test_vntr_copy_number_variation(self):
        """Samples with 2/3/5 copies of the same motif (TYMS-like VNTR)."""
        sequences = {
            "s2": ["0"] * 2,
            "s3": ["0"] * 3,
            "s5": ["0"] * 5,
        }
        result, consensus = _msa_core(
            sequences,
            _toy_sub_matrix(),
            gap_open_penalty=-5,
            gap_extend_penalty=-1,
            refine=True,
            max_refine_iter=3,
        )
        self._assert_rectangular(result)
        # No motif loss: each row keeps exactly its original copies.
        for name, seq in sequences.items():
            assert _strip_gaps(result[name]) == seq

    def test_private_motif_insertion(self):
        """A sample carrying a private motif between shared columns."""
        sequences = {
            "s1": ["0", "0", "0"],
            "s2": ["0", "1", "0", "0"],  # private "1" insertion
            "s3": ["0", "0", "0"],
        }
        result, consensus = _msa_core(
            sequences,
            _toy_sub_matrix(),
            gap_open_penalty=-5,
            gap_extend_penalty=-1,
            refine=True,
            max_refine_iter=3,
        )
        self._assert_rectangular(result)
        for name, seq in sequences.items():
            assert _strip_gaps(result[name]) == seq

    def test_identical_sequences_unchanged(self):
        sequences = {f"s{i}": ["0", "1", "0"] for i in range(4)}
        result, consensus = _msa_core(
            sequences,
            _toy_sub_matrix(),
            gap_open_penalty=-5,
            gap_extend_penalty=-1,
            refine=True,
            max_refine_iter=3,
        )
        self._assert_rectangular(result)
        for name in sequences:
            assert result[name] == ["0", "1", "0"]

    def test_same_length_substitutions_unchanged(self):
        """Rectangular inputs without insertions stay column-identical."""
        sequences = {
            "s1": ["0", "0", "1"],
            "s2": ["0", "1", "1"],
            "s3": ["0", "0", "0"],
        }
        result, _ = _msa_core(
            sequences,
            _toy_sub_matrix(),
            gap_open_penalty=-5,
            gap_extend_penalty=-1,
            refine=True,
            max_refine_iter=3,
        )
        self._assert_rectangular(result)
        assert len(result["s1"]) == 3  # no spurious insertion columns
        for name, seq in sequences.items():
            assert _strip_gaps(result[name]) == seq

    def test_without_refinement(self):
        sequences = {"s1": ["0"] * 2, "s2": ["0"] * 5}
        result, _ = _msa_core(
            sequences,
            _toy_sub_matrix(),
            gap_open_penalty=-5,
            gap_extend_penalty=-1,
            refine=False,
        )
        self._assert_rectangular(result)


def _make_vntr_adata(copy_numbers, n_motifs=1):
    """Build a minimal AnnData for sample_msa with VNTR-style copy numbers."""
    names = [f"sample{i}" for i in range(len(copy_numbers))]
    obs = pd.DataFrame(index=names)
    var = pd.DataFrame(
        {"motif_length": [10.0] * n_motifs},
        index=[str(i) for i in range(n_motifs)],
    )
    adata = ad.AnnData(obs=obs, var=var)
    adata.uns["motif_array"] = {
        name: ["0"] * cn for name, cn in zip(names, copy_numbers)
    }
    adata.uns["orientation_array"] = {
        name: ["+"] * cn for name, cn in zip(names, copy_numbers)
    }
    dist = np.zeros((n_motifs, n_motifs))
    adata.varp["motif_distance"] = dist
    adata.varp["rc_motif_distance"] = dist
    return adata


class TestSampleMsaRectangular:
    def test_vntr_samples_rectangular_output(self):
        adata = _make_vntr_adata([2, 3, 3, 5, 4])
        adata = sample_msa(adata)
        aligned = adata.uns["aligned_motif_array"]
        lengths = {len(v) for v in aligned.values()}
        assert len(lengths) == 1, f"ragged aligned_motif_array: {lengths}"
        # copy numbers preserved
        for name, row in aligned.items():
            expected_cn = int(name.replace("sample", ""))
            assert _strip_gaps(row) == ["0"] * [2, 3, 3, 5, 4][expected_cn]

    def test_orientation_array_same_length(self):
        adata = _make_vntr_adata([2, 5, 3])
        adata = sample_msa(adata)
        motifs = adata.uns["aligned_motif_array"]
        oris = adata.uns["aligned_orientation_array"]
        for name in motifs:
            assert len(motifs[name]) == len(oris[name])


class TestConsensusDefensiveCheck:
    def test_ragged_input_raises_clear_error(self):
        aligned_motifs = {
            "s1": ["0", "0", "0"],
            "s2": ["0", "0"],  # ragged
        }
        labels = np.array([0, 0])
        with pytest.raises(ValueError, match="not rectangular"):
            _build_haplotype_consensus(
                aligned_motifs, labels, ["s1", "s2"]
            )

    def test_rectangular_input_ok(self):
        aligned_motifs = {
            "s1": ["0", "1", "-"],
            "s2": ["0", "-", "0"],
        }
        labels = np.array([0, 0])
        consensus = _build_haplotype_consensus(
            aligned_motifs, labels, ["s1", "s2"]
        )
        # NOTE: gaps participate in the majority vote (ties resolve to the
        # first-seen token), so column 3 (["-", "0"]) yields "-".
        assert consensus == {"H1": ["0", "1", "-"]}
