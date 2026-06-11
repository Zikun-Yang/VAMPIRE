from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import anndata as ad

import logging

logger = logging.getLogger(__name__)


def markdup(
    adata: ad.AnnData,
) -> ad.AnnData:
    """
    Deduplicate samples by identical (motif, orientation) sequences.

    Parameters
    ----------
    adata: ad.AnnData
        Annotated data with ``motif_array`` and ``orientation_array`` in ``uns``.

    Returns
    -------
    ad.AnnData

    Examples
    --------
    >>> import vampire as vp
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.pp.markdup(adata)
    """
    if "motif_array" not in list(adata.uns.keys()):
        raise KeyError("motif_array is not in adata.uns")
    if "orientation_array" not in list(adata.uns.keys()):
        raise KeyError("orientation_array is not in adata.uns")
    
    motif_dict: dict[str, list[str]] = adata.uns["motif_array"]
    orientation_dict: dict[str, list[str]] = adata.uns["orientation_array"]

    seen: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
    unique_names: list[str] = []
    name_to_group: dict[str, int] = {}
    group_to_names: dict[int, list[str]] = {}

    group_id = 0
    for name in motif_dict:
        if len(motif_dict[name]) != len(orientation_dict[name]):
            raise ValueError(
                f"Sequence and orientation length mismatch for sample "
                f"'{name}': {len(motif_dict[name])} != {len(orientation_dict[name])}"
            )
        key = (tuple(motif_dict[name]), tuple(orientation_dict[name]))
        if key not in seen:
            seen[key] = group_id
            unique_names.append(name)
            gid = group_id
            group_id += 1
        else:
            gid = seen[key]
        name_to_group[name] = gid
        group_to_names.setdefault(gid, []).append(name)

    logger.info(
        "markdup completed. "
        f"{len(group_to_names)} unique groups identified from {len(motif_dict)} samples."
    )

    adata.obs["unique_group"] = adata.obs.index.map(name_to_group)

    return adata