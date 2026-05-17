from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List, Dict, Tuple, Optional

if TYPE_CHECKING:
    import numpy as np
    import anndata as ad

import logging

logger = logging.getLogger(__name__)

"""
#
# haplotype clustering function
#
"""
def haplotype(
    adata: ad.AnnData,
    aligned_key: str = "aligned",
    n_clusters: Optional[int] = None,
    gap_penalty: Optional[float] = None,
    linkage_method: str = "average",
    store_key: str = "haplotype",
    max_k: int = 10,
) -> ad.AnnData:
    """
    Cluster samples into haplotypes based on aligned motif arrays.

    Computes a pairwise distance matrix from the aligned motif arrays,
    then performs hierarchical clustering to group samples into
    haplotypes. A consensus sequence is built for each haplotype.

    When ``n_clusters`` is ``None`` (default), the function automatically
    searches the optimal number of clusters in the range ``[2, max_k]``
    using the silhouette score.  If the best score is below 0.25, all
    samples are assigned to a single haplotype.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with aligned motif arrays (from ``tl.align()``).
    aligned_key : str, default="aligned"
        Key prefix for aligned arrays in ``adata.uns``.
        Expects ``{aligned_key}_motif_array``.
    n_clusters : int, optional
        Number of haplotype clusters.  If ``None``, automatically
        determines the optimal number (see ``max_k``).
    gap_penalty : float, optional
        Distance penalty for motif-vs-gap positions. Default is
        ``max_motif_distance * 0.3``, giving a mild penalty that
        tolerates copy-number variation without dominating the distance.
    linkage_method : str, default="average"
        Linkage method for hierarchical clustering
        (e.g. ``"single"``, ``"complete"``, ``"average"``, ``"ward"``).
    store_key : str, default="haplotype"
        Key prefix for storing results in ``adata.uns`` and ``adata.obs``.
    max_k : int, default=10
        Maximum number of clusters to try when ``n_clusters`` is ``None``.

    Returns
    -------
    ad.AnnData
        Updated AnnData with haplotype labels in ``obs``:

        - ``obs[f"{store_key}"]`` — haplotype label (e.g. "H1", "H2")
        - ``uns[f"{store_key}_consensus"]`` — consensus per haplotype
        - ``uns[f"{store_key}_linkage"]`` — linkage matrix for dendrogram
        - ``uns[f"{store_key}_evaluation"]`` — evaluation curve (when ``n_clusters`` is ``None``)
        - ``obsp[f"{store_key}_structural_distance"]`` — structural distance matrix
        - ``obsp[f"{store_key}_cnv_distance"]`` — CNV distance matrix
        - ``obsp[f"{store_key}_combined_distance"]`` — combined (standardised Euclidean) distance matrix

    Examples
    --------
    >>> import vampire as vp
    >>> adata = vp.anno.pp.read_anno("results.annotation.tsv")
    >>> adata = vp.anno.tl.align(adata)
    >>> adata = vp.anno.tl.haplotype(adata)
    >>> adata.obs["haplotype"]
    sample1    H1
    sample2    H1
    sample3    H2
    Name: haplotype, dtype: category
    """
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    aligned_motifs: Dict[str, List[str]] = adata.uns.get(f"{aligned_key}_motif_array")
    if aligned_motifs is None:
        raise KeyError(
            f"aligned motif array not found at uns['{aligned_key}_motif_array']. "
            f"Run vp.anno.tl.align() first."
        )

    names: List[str] = list(aligned_motifs.keys())
    n = len(names)

    if n == 0:
        return adata
    if n == 1:
        adata.obs[store_key] = "H1"
        adata.uns[f"{store_key}_consensus"] = {"H1": aligned_motifs[names[0]]}
        return adata

    # Default gap penalty: mild, tolerates CNV
    if gap_penalty is None:
        max_dist = float(adata.varp["motif_distance"].max())
        gap_penalty = max_dist * 0.3 if max_dist > 0 else 1.0

    # Compute distance matrices (structural + CNV)
    structural_mat, cnv_mat = _compute_aligned_distance(
        aligned_motifs, names, adata.varp["motif_distance"], gap_penalty
    )

    # Align to adata.obs_names
    obs_names = list(adata.obs_names)
    if names != obs_names:
        name_to_idx = {name: i for i, name in enumerate(names)}
        idx_map = [name_to_idx[name] for name in obs_names]
        structural_mat = structural_mat[np.ix_(idx_map, idx_map)]
        cnv_mat = cnv_mat[np.ix_(idx_map, idx_map)]

    # Standardise each component by its own std, then combine by Euclidean norm
    s_std = structural_mat[structural_mat > 0].std() or 1.0
    c_std = cnv_mat[cnv_mat > 0].std() or 1.0
    combined_dist = np.sqrt((structural_mat / s_std) ** 2 + (cnv_mat / c_std) ** 2)

    # Hierarchical clustering
    Z = linkage(squareform(combined_dist), method=linkage_method)

    if n_clusters is None:
        eval_result = _evaluate_clusters(combined_dist, Z, max_k)
        best_k = eval_result["best_k"]

        if best_k == 1:
            labels = np.ones(n, dtype=int)
            logger.warning(
                f"Weak clustering structure (best silhouette {eval_result['best_score']:.3f}). "
                f"Assigning all {n} samples to a single haplotype."
            )
        else:
            labels = fcluster(Z, t=best_k, criterion="maxclust")
            logger.info(
                f"Auto-selected k={best_k} (silhouette={eval_result['best_score']:.3f})."
            )

        adata.uns[f"{store_key}_evaluation"] = eval_result
        n_clusters = best_k
    else:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Build consensus for each haplotype
    haplo_consensus = _build_haplotype_consensus(aligned_motifs, labels, names)

    # Store results
    adata.obs[store_key] = [f"H{int(l)}" for l in labels]
    adata.obs[store_key] = adata.obs[store_key].astype("category")
    adata.uns[f"{store_key}_consensus"] = haplo_consensus
    adata.uns[f"{store_key}_linkage"] = Z
    adata.obsp[f"{store_key}_structural_distance"] = structural_mat
    adata.obsp[f"{store_key}_cnv_distance"] = cnv_mat
    adata.obsp[f"{store_key}_combined_distance"] = combined_dist

    logger.info(
        f"Clustered {n} samples into {n_clusters} haplotypes. "
        f"Cluster sizes: {dict(adata.obs[store_key].value_counts().sort_index())}"
    )

    return adata

def _compute_aligned_distance(
    aligned_motifs: Dict[str, List[str]],
    names: List[str],
    motif_distance: np.ndarray,
    gap_penalty: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pairwise structural and CNV distance matrices.

    Returns two independent matrices:

    - *structural_mat*: average motif mismatch within the overlap region
      (positions where both samples have a motif). This reflects similarity
      of motif order / pattern and is insensitive to CNV.
    - *cnv_mat*: gap_count * gap_penalty / max(copy_i, copy_j). This
      reflects copy-number variation and is bounded by ``gap_penalty``.
    """
    import numpy as np

    max_dist = float(motif_distance.max())
    n = len(names)
    structural_mat = np.zeros((n, n), dtype=float)
    cnv_mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            seq_i = aligned_motifs[names[i]]
            seq_j = aligned_motifs[names[j]]

            structural_dist: float = 0.0
            overlap: int = 0
            gap_count: int = 0

            for a, b in zip(seq_i, seq_j):
                if a == "-" and b == "-":
                    continue
                if a == "-" or b == "-":
                    gap_count += 1
                else:
                    overlap += 1
                    if a != b:
                        structural_dist += motif_distance[int(a), int(b)]

            # Sequence lengths (needed for both structural and CNV)
            len_i = sum(1 for x in seq_i if x != "-")
            len_j = sum(1 for x in seq_j if x != "-")
            min_len = min(len_i, len_j)
            max_len = max(len_i, len_j)

            # Normalise structural distance by overlap length (identity)
            if overlap > 0:
                avg_dist = structural_dist / overlap
            else:
                avg_dist = max_dist

            # Incorporate coverage: penalise uncovered regions at max_dist.
            # Use min_len so that length differences are handled by CNV, not
            # structural distance.  If the shorter sequence is fully covered,
            # coverage = 1 and structural distance depends only on identity.
            # similarity = (1 - avg_dist/max_dist) * (overlap/min_len)
            # structural_dist = max_dist * (1 - similarity)
            if min_len > 0:
                structural_dist = (
                    avg_dist * overlap + max_dist * (min_len - overlap)
                ) / min_len
            else:
                structural_dist = max_dist

            # CNV penalty normalised by max copy number
            cnv_penalty = (
                gap_count * gap_penalty / max_len if max_len > 0 else 0.0
            )

            structural_mat[i, j] = structural_mat[j, i] = structural_dist # TODO whether use log-transform for structural distance
            cnv_mat[i, j] = cnv_mat[j, i] = cnv_penalty # TODO whether use log-transform for CNV distance

    return structural_mat, cnv_mat

def _build_haplotype_consensus(
    aligned_motifs: Dict[str, List[str]],
    labels: np.ndarray,
    names: List[str],
) -> Dict[str, List[str]]:
    """
    Build consensus sequence for each haplotype cluster.
    """
    from collections import Counter

    consensus = {}
    unique_labels = sorted(set(labels))

    for h in unique_labels:
        samples_in_h = [names[i] for i in range(len(names)) if labels[i] == h]
        if not samples_in_h:
            continue

        # Get aligned length from first sample
        seq_len = len(aligned_motifs[samples_in_h[0]])
        h_consensus = []

        for pos in range(seq_len):
            motifs = [
                aligned_motifs[s][pos]
                for s in samples_in_h
                if aligned_motifs[s][pos] != "-"
            ]
            if motifs:
                h_consensus.append(Counter(motifs).most_common(1)[0][0])
            else:
                h_consensus.append("-")

        consensus[f"H{h}"] = h_consensus

    return consensus

def _evaluate_clusters(
    dist_mat: np.ndarray,
    Z: np.ndarray,
    max_k: int,
    silhouette_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    Evaluate cluster quality for k=2..max_k using silhouette score.

    Returns an evaluation dict.  If the best silhouette score is below
    ``silhouette_threshold``, ``best_k`` is set to 1 (all samples in one
    cluster).
    """
    import numpy as np
    from sklearn.metrics import silhouette_score
    from scipy.cluster.hierarchy import fcluster

    n = dist_mat.shape[0]
    k_range = list(range(2, min(max_k + 1, n)))

    if len(k_range) == 0:
        return {
            "k_range": [],
            "silhouette": [],
            "best_k": 1,
            "best_score": 0.0,
        }

    scores: List[float] = []
    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        score = float(silhouette_score(dist_mat, labels, metric="precomputed"))
        scores.append(score)

    best_idx = int(np.argmax(scores))
    best_k = k_range[best_idx]
    best_score = scores[best_idx]

    if best_score < silhouette_threshold:
        best_k = 1

    return {
        "k_range": k_range,
        "silhouette": scores,
        "best_k": best_k,
        "best_score": best_score,
    }


