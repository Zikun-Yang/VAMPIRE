"""Shared fixtures for anno tests."""

from pathlib import Path

import pytest

# Set alignment globals before numba compiles the jitted functions.
import vampire._anno as _anno

_anno.MATCH_SCORE = 2
_anno.MISMATCH_PENALTY = 7
_anno.GAP_OPEN_PENALTY = 7
_anno.GAP_EXTEND_PENALTY = 7

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def tmp_anno_cfg(tmp_path):
    """Factory fixture that returns a cfg dict for run_anno."""

    def _make(input_fa: str, **overrides):
        cfg = {
            "input": str(DATA_DIR / input_fa),
            "prefix": str(tmp_path / "output"),
            "job_dir": str(tmp_path / ".vampire"),
            "threads": 1,
            "no_auto": False,
            "ksize": 5,
            "motif": "base",
            "motifnum": 30,
            "kratio": 0.01,
            "kmin": 3,
            "seq_win_size": 5000,
            "seq_ovlp_size": 1000,
            "resource": 50,
            "auto": False,
            "no_denovo": False,
            "force": False,
            "reverse": False,
            "annotation_min_similarity": 0.6,
            "finding_min_similarity": 0.8,
            "match_score": 2,
            "mismatch_penalty": 7,
            "gap_open_penalty": 7,
            "gap_extend_penalty": 7,
            "debug": False,
            "skip_report": False,
            "skip_h5ad": False,
        }
        cfg.update(overrides)
        return cfg

    return _make
