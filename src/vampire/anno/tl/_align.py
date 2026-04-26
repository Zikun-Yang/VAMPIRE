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
# DNA logo function
#
"""
def align(
    adata: ad.AnnData,
    reference: Optional[str] = None,
    match_score: float = 2.0,
    gap_open: float = 5.0,
    gap_extend: float = 1.0,
    refine: bool = True,
    store_key: str = "aligned",
) -> ad.AnnData:
    """
    Multiple sequence alignment of motif arrays across samples.

    Uses a greedy progressive alignment strategy: iteratively merge the
    closest pair of profiles (or single sequences) until one MSA remains.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with ``motif_array`` and ``orientation_array`` in
        ``uns``, and ``motif_distance`` in ``varp``.
    reference : str, optional
        Sample name to use as the initial reference. If ``None``, the
        sample with the minimum average pairwise distance to all others
        is selected automatically.
    match_score : float, default=2.0
        Score for aligning identical motifs.
    gap_open : float, default=5.0
        Penalty for opening a gap.
    gap_extend : float, default=1.0
        Penalty for extending a gap.
    refine : bool, default=True
        Whether to perform one round of consensus-based refinement
        after the initial progressive alignment.
    store_key : str, default="aligned"
        Key prefix for storing results in ``adata.uns``.
        Stores ``{store_key}_motif_array`` and
        ``{store_key}_orientation_array``.

    Returns
    -------
    ad.AnnData
        The updated AnnData with alignment results in ``uns``.

    Examples
    --------
    >>> import vampire as vp
    >>> adata = vp.anno.pp.read_anno("results.annotation.tsv")
    >>> adata = vp.anno.tl.align(adata, match_score=2.0, gap_open=5.0)
    >>> adata.uns["aligned_motif_array"]["sample1"]
    ['0', '1', '-', '3', '2']
    """
    import numpy as np
    from copy import deepcopy

    sequences: Dict[str, List[str]] = adata.uns["motif_array"]
    orientations: Dict[str, List[str]] = adata.uns["orientation_array"]
    names: List[str] = list(sequences.keys())
    n: int = len(names)

    if n == 0:
        return adata
    if n == 1:
        adata.uns[f"{store_key}_motif_array"] = deepcopy(sequences)
        adata.uns[f"{store_key}_orientation_array"] = deepcopy(orientations)
        return adata

    # Build substitution matrix
    sub_matrix = _build_sub_matrix(adata, match_score)

    # Initialize profiles: each sequence as a single-sequence profile
    profiles: Dict[int, List[List[str]]] = {
        i: [[sequences[names[i]][k]] for k in range(len(sequences[names[i]]))]
        for i in range(n)
    }
    ori_profiles: Dict[int, List[List[str]]] = {
        i: [[orientations[names[i]][k]] for k in range(len(orientations[names[i]]))]
        for i in range(n)
    }
    seq_indices: Dict[int, List[int]] = {i: [i] for i in range(n)}

    # Compute pairwise distances
    dist_mat = np.zeros((2 * n, 2 * n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            score = _nw_score(sequences[names[i]], sequences[names[j]], sub_matrix, gap_open, gap_extend)
            dist_mat[i, j] = dist_mat[j, i] = -score

    # Greedy progressive merge
    active = set(range(n))
    next_id = n

    while len(active) > 1:
        min_dist = float("inf")
        pair: Optional[Tuple[int, int]] = None
        for i in active:
            for j in active:
                if i < j and dist_mat[i, j] < min_dist:
                    min_dist = dist_mat[i, j]
                    pair = (i, j)

        if pair is None:
            break

        i, j = pair

        # Get consensus for each profile
        cons_i = _profile_consensus(profiles[i])
        cons_j = _profile_consensus(profiles[j])

        # Align consensus sequences
        aligned_i, aligned_j = _nw(cons_i, cons_j, sub_matrix, gap_open, gap_extend)

        # Merge profiles
        merged = _merge_profiles(profiles[i], profiles[j], aligned_i, aligned_j)
        merged_ori = _merge_profiles(ori_profiles[i], ori_profiles[j], aligned_i, aligned_j)

        profiles[next_id] = merged
        ori_profiles[next_id] = merged_ori
        seq_indices[next_id] = seq_indices[i] + seq_indices[j]

        # Update distances
        for k in active:
            if k != i and k != j:
                cons_k = _profile_consensus(profiles[k])
                cons_new = _profile_consensus(merged)
                score = _nw_score(cons_new, cons_k, sub_matrix, gap_open, gap_extend)
                dist_mat[next_id, k] = dist_mat[k, next_id] = -score

        active.remove(i)
        active.remove(j)
        active.add(next_id)
        next_id += 1

    # Extract final MSA
    final_id = list(active)[0]
    final_profile = profiles[final_id]
    final_ori = ori_profiles[final_id]
    final_indices = seq_indices[final_id]

    # Build aligned arrays
    aligned_motifs: Dict[str, List[str]] = {}
    aligned_oris: Dict[str, List[str]] = {}

    for pos, seq_idx in enumerate(final_indices):
        name = names[seq_idx]
        aligned_motifs[name] = [col[pos] for col in final_profile]
        aligned_oris[name] = [col[pos] for col in final_ori]

    # Optional: consensus refinement
    if refine:
        consensus = _profile_consensus(final_profile)

        refined_motifs: Dict[str, List[str]] = {}
        refined_oris: Dict[str, List[str]] = {}

        for name in names:
            aligned_seq, aligned_cons = _nw(
                sequences[name], consensus, sub_matrix, gap_open, gap_extend
            )
            refined_motifs[name] = aligned_seq
            # Align orientation: follow motif positions, insert '-' where motif has gap
            ori = orientations[name]
            refined_ori = []
            ori_idx = 0
            for m in aligned_seq:
                if m == "-":
                    refined_ori.append("-")
                else:
                    refined_ori.append(ori[ori_idx])
                    ori_idx += 1
            refined_oris[name] = refined_ori

        adata.uns[f"{store_key}_motif_array"] = refined_motifs
        adata.uns[f"{store_key}_orientation_array"] = refined_oris
        adata.uns[f"{store_key}_consensus"] = consensus
    else:
        adata.uns[f"{store_key}_motif_array"] = aligned_motifs
        adata.uns[f"{store_key}_orientation_array"] = aligned_oris
        adata.uns[f"{store_key}_consensus"] = _profile_consensus(final_profile)

    logger.info(
        f"Aligned {n} samples. "
        f"Original lengths: {[len(sequences[n]) for n in names]}. "
        f"Aligned length: {len(aligned_motifs[names[0]])}."
    )

    return adata

def _build_sub_matrix(
    adata: ad.AnnData,
    match_score: float,
) -> np.ndarray:
    """Build substitution matrix from motif_distance."""
    import numpy as np

    n_motifs = len(adata.var)
    dist_mat = adata.varp["motif_distance"]

    max_dist = dist_mat.max()
    if max_dist == 0:
        scale = 1.0
    else:
        scale = match_score / max_dist

    sub_matrix = np.zeros((n_motifs, n_motifs), dtype=float)
    for i in range(n_motifs):
        for j in range(n_motifs):
            if i == j:
                sub_matrix[i, j] = match_score
            else:
                sub_matrix[i, j] = -dist_mat[i, j] * scale

    return sub_matrix


def _nw_score(
    seq_a: List[str],
    seq_b: List[str],
    sub_matrix: np.ndarray,
    gap_open: float,
    gap_extend: float,
) -> float:
    """Needleman-Wunsch optimal alignment score (affine gap)."""
    import numpy as np

    n = len(seq_a)
    m = len(seq_b)

    M = np.zeros((n + 1, m + 1), dtype=float)
    I = np.full((n + 1, m + 1), float("-inf"), dtype=float)
    D = np.full((n + 1, m + 1), float("-inf"), dtype=float)

    for i in range(1, n + 1):
        I[i, 0] = gap_open + (i - 1) * gap_extend

    for j in range(1, m + 1):
        D[0, j] = gap_open + (j - 1) * gap_extend

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = sub_matrix[int(seq_a[i - 1]), int(seq_b[j - 1])]
            M[i, j] = max(M[i - 1, j - 1], I[i - 1, j - 1], D[i - 1, j - 1]) + score
            I[i, j] = max(M[i - 1, j] + gap_open, I[i - 1, j] + gap_extend)
            D[i, j] = max(M[i, j - 1] + gap_open, D[i, j - 1] + gap_extend)

    return float(max(M[n, m], I[n, m], D[n, m]))


def _nw(
    seq_a: List[str],
    seq_b: List[str],
    sub_matrix: np.ndarray,
    gap_open: float,
    gap_extend: float,
) -> Tuple[List[str], List[str]]:
    """Needleman-Wunsch global alignment with affine gap penalty."""
    import numpy as np

    n = len(seq_a)
    m = len(seq_b)

    M = np.zeros((n + 1, m + 1), dtype=float)
    I = np.full((n + 1, m + 1), float("-inf"), dtype=float)
    D = np.full((n + 1, m + 1), float("-inf"), dtype=float)

    for i in range(1, n + 1):
        I[i, 0] = gap_open + (i - 1) * gap_extend

    for j in range(1, m + 1):
        D[0, j] = gap_open + (j - 1) * gap_extend

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = sub_matrix[int(seq_a[i - 1]), int(seq_b[j - 1])]
            M[i, j] = max(M[i - 1, j - 1], I[i - 1, j - 1], D[i - 1, j - 1]) + score
            I[i, j] = max(M[i - 1, j] + gap_open, I[i - 1, j] + gap_extend)
            D[i, j] = max(M[i, j - 1] + gap_open, D[i, j - 1] + gap_extend)

    # Traceback
    aligned_a: List[str] = []
    aligned_b: List[str] = []

    i, j = n, m
    curr_score = max(M[i, j], I[i, j], D[i, j])
    if curr_score == M[i, j]:
        curr = "M"
    elif curr_score == I[i, j]:
        curr = "I"
    else:
        curr = "D"

    while i > 0 or j > 0:
        if curr == "M":
            aligned_a.append(seq_a[i - 1])
            aligned_b.append(seq_b[j - 1])
            score = sub_matrix[int(seq_a[i - 1]), int(seq_b[j - 1])]
            prev_val = M[i, j] - score
            if i > 0 and j > 0 and abs(M[i - 1, j - 1] - prev_val) < 1e-9:
                curr = "M"
            elif i > 0 and j > 0 and abs(I[i - 1, j - 1] - prev_val) < 1e-9:
                curr = "I"
            else:
                curr = "D"
            i -= 1
            j -= 1
        elif curr == "I":
            aligned_a.append(seq_a[i - 1])
            aligned_b.append("-")
            if i > 0 and abs(M[i - 1, j] + gap_open - I[i, j]) < 1e-9:
                curr = "M"
            else:
                curr = "I"
            i -= 1
        else:  # "D"
            aligned_a.append("-")
            aligned_b.append(seq_b[j - 1])
            if j > 0 and abs(M[i, j - 1] + gap_open - D[i, j]) < 1e-9:
                curr = "M"
            else:
                curr = "D"
            j -= 1

    aligned_a.reverse()
    aligned_b.reverse()
    return aligned_a, aligned_b


def _profile_consensus(profile: List[List[str]]) -> List[str]:
    """Build consensus sequence from a profile."""
    from collections import Counter

    consensus: List[str] = []
    for col in profile:
        motifs = [m for m in col if m != "-"]
        if motifs:
            consensus.append(Counter(motifs).most_common(1)[0][0])
        else:
            consensus.append("-")
    return consensus


def _merge_profiles(
    profile_a: List[List[str]],
    profile_b: List[List[str]],
    aligned_a: List[str],
    aligned_b: List[str],
) -> List[List[str]]:
    """Merge two profiles based on aligned consensus sequences."""
    merged: List[List[str]] = []
    idx_a = 0
    idx_b = 0

    len_a = len(profile_a[0]) if profile_a else 0
    len_b = len(profile_b[0]) if profile_b else 0

    for ca, cb in zip(aligned_a, aligned_b):
        if ca == "-" and cb != "-":
            col_a = ["-"] * len_a
            col_b = profile_b[idx_b]
            merged.append(col_a + col_b)
            idx_b += 1
        elif ca != "-" and cb == "-":
            col_a = profile_a[idx_a]
            col_b = ["-"] * len_b
            merged.append(col_a + col_b)
            idx_a += 1
        else:
            col_a = profile_a[idx_a] if ca != "-" else ["-"] * len_a
            col_b = profile_b[idx_b] if cb != "-" else ["-"] * len_b
            merged.append(col_a + col_b)
            if ca != "-":
                idx_a += 1
            if cb != "-":
                idx_b += 1

    return merged




def _compute_aligned_distance(
    aligned_motifs: Dict[str, List[str]],
    names: List[str],
    motif_distance: np.ndarray,
    gap_penalty: float,
) -> np.ndarray:
    """Compute pairwise distance matrix from aligned motif arrays."""
    import numpy as np

    n = len(names)
    dist_mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            seq_i = aligned_motifs[names[i]]
            seq_j = aligned_motifs[names[j]]
            total_dist = 0.0
            valid = 0

            for a, b in zip(seq_i, seq_j):
                if a == "-" and b == "-":
                    continue  # 双 gap 不计入
                valid += 1
                if a == "-" or b == "-":
                    total_dist += gap_penalty
                elif a != b:
                    total_dist += motif_distance[int(a), int(b)]
                # else: same motif, dist = 0

            if valid > 0:
                dist_mat[i, j] = dist_mat[j, i] = total_dist / valid
            else:
                dist_mat[i, j] = dist_mat[j, i] = 0.0

    return dist_mat


def _build_haplotype_consensus(
    aligned_motifs: Dict[str, List[str]],
    labels: np.ndarray,
    names: List[str],
) -> Dict[str, List[str]]:
    """Build consensus sequence for each haplotype cluster."""
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


def haplotype(
    adata: ad.AnnData,
    aligned_key: str = "aligned",
    n_clusters: int = 3,
    gap_penalty: Optional[float] = None,
    linkage_method: str = "average",
    store_key: str = "haplotype",
) -> ad.AnnData:
    """
    Cluster samples into haplotypes based on aligned motif arrays.

    Computes a pairwise distance matrix from the aligned motif arrays,
    then performs hierarchical clustering to group samples into
    haplotypes. A consensus sequence is built for each haplotype.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with aligned motif arrays (from ``tl.align()``).
    aligned_key : str, default="aligned"
        Key prefix for aligned arrays in ``adata.uns``.
        Expects ``{aligned_key}_motif_array``.
    n_clusters : int, default=3
        Number of haplotype clusters.
    gap_penalty : float, optional
        Distance penalty for motif-vs-gap positions. Default is
        ``max_motif_distance * 0.3``, giving a mild penalty that
        tolerates copy-number variation without dominating the distance.
    linkage_method : str, default="average"
        Linkage method for hierarchical clustering
        (e.g. ``"single"``, ``"complete"``, ``"average"``, ``"ward"``).
    store_key : str, default="haplotype"
        Key prefix for storing results in ``adata.uns`` and ``adata.obs``.

    Returns
    -------
    ad.AnnData
        Updated AnnData with haplotype labels in ``obs``:

        - ``obs[f"{store_key}"]`` — haplotype label (e.g. "H1", "H2")
        - ``uns[f"{store_key}_consensus"]`` — consensus per haplotype
        - ``uns[f"{store_key}_linkage"]`` — linkage matrix for dendrogram
        - ``uns[f"{store_key}_distance"]`` — sample distance matrix

    Examples
    --------
    >>> import vampire as vp
    >>> adata = vp.anno.pp.read_anno("results.annotation.tsv")
    >>> adata = vp.anno.tl.align(adata)
    >>> adata = vp.anno.tl.haplotype(adata, n_clusters=3)
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
            f"Run tl.align() first."
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

    # Compute distance matrix
    dist_mat = _compute_aligned_distance(
        aligned_motifs, names, adata.varp["motif_distance"], gap_penalty
    )

    # Hierarchical clustering
    Z = linkage(squareform(dist_mat), method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Build consensus for each haplotype
    haplo_consensus = _build_haplotype_consensus(aligned_motifs, labels, names)

    # Store results
    adata.obs[store_key] = [f"H{int(l)}" for l in labels]
    adata.obs[store_key] = adata.obs[store_key].astype("category")
    adata.uns[f"{store_key}_consensus"] = haplo_consensus
    adata.uns[f"{store_key}_linkage"] = Z
    adata.uns[f"{store_key}_distance"] = dist_mat

    logger.info(
        f"Clustered {n} samples into {n_clusters} haplotypes. "
        f"Cluster sizes: {dict(adata.obs[store_key].value_counts().sort_index())}"
    )

    return adata
