from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    import anndata as ad

def _get_data_path(filename: str) -> Path:
    """Return path to bundled data file."""
    from pathlib import Path
    return Path(__file__).parent / "data" / filename

def ancestry() -> Dict[str, str]:
    """
    Load ancestry information for the HPRCp1, HGSVCp1, APGp1 samples.

    Returns
    -------
    Dict[str, str]
        A dictionary mapping sample ID to ancestry label.
    """
    path = _get_data_path("ancestry.csv")
    ancestry_dict = {}
    with open(path, "r") as f:
        for line in f:
            sample_id, ancestry_label = line.strip().split(",")
            ancestry_dict[sample_id] = ancestry_label
    
    return ancestry_dict

def wdr7_hprc() -> ad.AnnData:
    """
    Load 69 bp VNTR locus in the intron of the gene WDR7 among the 94 HPRC samples and T2T-CHM13v2.0.
    The coordinates on T2T-CHM13v2.0 are `chr18:57,226,379-57,227,527`.

    This dataset contains TODO.

    Returns
    -------
    ad.AnnData
        Raw AnnData object without any processing.
    """
    from pathlib import Path
    import anndata as ad

    path = _get_data_path("wdr7_hprc.h5ad")
    return ad.read_h5ad(path)
