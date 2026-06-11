from __future__ import annotations
from typing import TYPE_CHECKING

import pooch

if TYPE_CHECKING:
    from pathlib import Path
    import anndata as ad

_DATA_FETCHER = pooch.create(
    path=pooch.os_cache("vampire"),
    base_url=(
        "https://github.com/Zikun-Yang/VAMPIRE/"
        "releases/download/v0.4.0a2/"
    ),
    registry={
        "chm13_cen1_tracks.pkl": "sha256:ffa6c087468d63f3259da10e2500e9f288973737816d2b8ed304c4ea0eb32b57",  # <-- FILL IN hash after upload
    },
)

def _get_data_path(filename: str) -> Path:
    """Return path to bundled data file."""
    from pathlib import Path
    return Path(__file__).parent / "data" / filename

def ancestry() -> dict[str, str]:
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

    This dataset contains annotations of a 69-bp motif VNTR locus (`chr18:57,226,379-57,227,527` in T2T-CHM13v2.0) 
    located within an intron of the *`WDR7`* gene across 95 human haplotype assemblies, 
    including 47 individuals from Phase 1 of the `Human Pangenome Reference Consortium (HPRC)` and 
    the `T2T-CHM13v2.0` reference genome.

    Returns
    -------
    ad.AnnData
        Raw AnnData object without any processing.
    """
    from pathlib import Path
    import anndata as ad

    path = _get_data_path("wdr7_hprc.h5ad")
    return ad.read_h5ad(path)


def chm13_cen1_tracks() -> list[dict]:
    """
    Load example tracks for the T2T-CHM13 centromere 1 region.

    This dataset is intended for demonstrating :func:`vampire.anno.pl.tracksplot`.
    It contains a list of track configuration dictionaries (bedgraph, bed, heatmap)
    covering the chm13_chr1 centromeric HOR array region.

    The data is downloaded on first use and cached under
    ``~/.cache/vampire/`` via `pooch`.

    Returns
    -------
    list[dict]
        List of track dicts ready to pass to :func:`vampire.anno.pl.tracksplot`.

    Examples
    --------
    >>> import vampire as vp
    >>> tracks = vp.datasets.chm13_cen1_tracks()
    >>> fig = vp.anno.pl.tracksplot(tracks, "chm13_chr1:121119216-127324115")
    """
    import pickle

    path = _DATA_FETCHER.fetch("chm13_cen1_tracks.pkl")

    with open(path, "rb") as f:
        tracks = pickle.load(f)

    return tracks
