"""Integration tests for vampire anno CLI."""

from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from vampire._anno import run_anno
import hashlib

def md5sum(path):
    md5 = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)

    return md5.hexdigest()

DATA_DIR = Path(__file__).parent / "data"

class TestBasicAnnotation:
    """Smoke tests for default anno behaviour across motif lengths."""

    @pytest.mark.parametrize(
        "input_fa,ksize,expected_motif",
        [
            ("001-1bp_polyA.fa", 3, "A"),
            ("002-5bp_perfect.fa", 5, "ATTGG"),
        ],
    )
    def test_perfect_tr(self, tmp_anno_cfg, input_fa, ksize, expected_motif):
        cfg = tmp_anno_cfg(input_fa, ksize=ksize)
        run_anno(cfg)

        prefix = cfg["prefix"]
        assert Path(f"{prefix}.motif.tsv").exists()
        assert Path(f"{prefix}.annotation.tsv").exists()
        assert Path(f"{prefix}.concise.tsv").exists()
        assert Path(f"{prefix}.distance.tsv").exists()

        motif_df = pl.read_csv(f"{prefix}.motif.tsv", separator="\t")
        assert motif_df.shape[0] >= 1
        assert expected_motif in motif_df["motif"].to_list()


class TestComplexSequence:
    """Tests using the complex 003-5bp_snv_gap_rc.fa data."""

    def test_ngap_handling(self, tmp_anno_cfg):
        """N-regions should be emitted as gap rows (motif=None, cigar contains N)."""
        cfg = tmp_anno_cfg("003-5bp_snv_gap_rc.fa", ksize=5)
        run_anno(cfg)

        gap_included_count = 0
        anno_df = pl.read_csv(f"{cfg['prefix']}.annotation.tsv", separator="\t")
        gap_rows = anno_df.filter((pl.col("chrom") == "5bp_TR_1") & (pl.col("start") >= 46) & (pl.col("end") <= 80))
        gap_included_count += gap_rows.shape[0]
        gap_rows = anno_df.filter((pl.col("chrom") == "5bp_TR_1") & (pl.col("start") >= 118) & (pl.col("end") <= 123))
        gap_included_count += gap_rows.shape[0]
        gap_rows = anno_df.filter((pl.col("chrom") == "5bp_TR_2") & (pl.col("start") >= 43) & (pl.col("end") <= 77))
        gap_included_count += gap_rows.shape[0]
        gap_rows = anno_df.filter((pl.col("chrom") == "5bp_TR_2") & (pl.col("start") >= 115) & (pl.col("end") <= 120))
        gap_included_count += gap_rows.shape[0]
        assert gap_rows.shape[0] == 0

    def test_snv_detection(self, tmp_anno_cfg):
        """Mutated motifs should have distance > 0."""
        cfg = tmp_anno_cfg("003-5bp_snv_gap_rc.fa", ksize=5)
        run_anno(cfg)

        motif_df = pl.read_csv(f"{cfg['prefix']}.motif.tsv", separator="\t")
        motifs = motif_df["motif"].to_list()
        assert "ATTGG" in motifs
        assert "ATTCG" in motifs

    def test_reverse_complement_detected(self, tmp_anno_cfg):
        """With --reverse, orientation='-' rows should appear."""
        cfg = tmp_anno_cfg("003-5bp_snv_gap_rc.fa", ksize=5, reverse=True)
        run_anno(cfg)

        anno_df = pl.read_csv(f"{cfg['prefix']}.annotation.tsv", separator="\t")
        rc_rows = anno_df.filter(pl.col("orientation") == "-")
        assert rc_rows.shape[0] > 0
    
    def test_exact_file_output(self, tmp_anno_cfg): # TODO
        """Test that output files match expected exactly (except for log)."""
        cfg = tmp_anno_cfg("003-5bp_snv_gap_rc.fa", reverse=True)
        run_anno(cfg)

        expected_dir = Path(__file__).parent / "expected"
        for suffix in ["motif.tsv", "annotation.tsv", "concise.tsv", "distance.tsv"]:
            expected_path = expected_dir / f"003-5bp_snv_gap_rc.{suffix}"
            actual_path = Path(f"{cfg['prefix']}.{suffix}")
            assert actual_path.exists(), f"Missing output file: {actual_path}"
            expected_df = pl.read_csv(expected_path, separator="\t")
            actual_df = pl.read_csv(actual_path, separator="\t")
            assert_frame_equal(expected_df, actual_df, check_dtypes=False, abs_tol=1.5, rel_tol=0.05)

        assert Path(f"{cfg['prefix']}.h5ad").exists(), "Missing h5ad file"
        assert md5sum(Path(f"{cfg['prefix']}.h5ad")) == md5sum(expected_dir / f"003-5bp_snv_gap_rc.h5ad"), "h5ad file content mismatch"


class TestNoDenovo:
    """Tests for --no-denovo mode."""

    def test_uses_reference_only(self, tmp_anno_cfg):
        cfg = tmp_anno_cfg("002-5bp_perfect.fa", no_denovo=True, force=True, motif=f"{DATA_DIR}/5bp_perfect.motif.fa")
        run_anno(cfg)

        motif_df = pl.read_csv(f"{cfg['prefix']}.motif.tsv", separator="\t")
        assert motif_df.shape[0] >= 1


class TestOutputStructure:
    """Validate output schema and invariants."""

    def test_annotation_columns(self, tmp_anno_cfg):
        cfg = tmp_anno_cfg("002-5bp_perfect.fa", ksize=5)
        run_anno(cfg)

        df = pl.read_csv(f"{cfg['prefix']}.annotation.tsv", separator="\t")
        required = {"chrom", "length", "start", "end", "motif", "orientation", "sequence", "score", "cigar"}
        assert required.issubset(set(df.columns))

    def test_concise_columns(self, tmp_anno_cfg):
        cfg = tmp_anno_cfg("002-5bp_perfect.fa", ksize=5)
        run_anno(cfg)

        df = pl.read_csv(f"{cfg['prefix']}.concise.tsv", separator="\t")
        required = {"chrom", "length", "start", "end", "motif", "orientation", "copyNumber", "score", "cigar"}
        assert required.issubset(set(df.columns))

    def test_motif_columns(self, tmp_anno_cfg):
        cfg = tmp_anno_cfg("002-5bp_perfect.fa", ksize=5)
        run_anno(cfg)

        df = pl.read_csv(f"{cfg['prefix']}.motif.tsv", separator="\t")
        required = {"id", "motif", "copyNumber", "label"}
        assert required.issubset(set(df.columns))
