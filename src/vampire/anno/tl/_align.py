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
# motif array aligning function
#
"""
def align(
    adata: ad.AnnData,
    reference: Optional[str] = None,
    match_score: int = 2,
    mismatch_penalty: int = -3,
    gap_open_penalty: int = -5,
    gap_extend_penalty: int = -1,
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
    match_score : int, default=2
        Penalty coefficient for aligning motifs. The substitution score is
        ``avg_motif_length - distance * match_score - distance * mismatch_penalty``.
    mismatch_penalty : int, default=-3
        Additional penalty coefficient for mismatched bases.
    gap_open_penalty : int, default=-5
        Penalty for opening a gap.
    gap_extend_penalty : int, default=-1
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
    >>> adata = vp.anno.tl.align(adata)
    >>> adata.uns["aligned_motif_array"]["sample1"]
    ['0', '1', '-', '3', '2']
    """
    import numpy as np
    from copy import deepcopy

    if match_score <= 0:
        raise ValueError("match_score should be positive.")
    if mismatch_penalty >= 0:
        raise ValueError("mismatch_penalty should be negative.")
    if gap_open_penalty >= 0:
        raise ValueError("gap_open_penalty should be negative.")
    if gap_extend_penalty >= 0:
        raise ValueError("gap_extend_penalty should be negative.")

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
    sub_matrix = _build_sub_matrix(adata, match_score, mismatch_penalty)

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
            score = _nw_score(sequences[names[i]], sequences[names[j]], sub_matrix, gap_open_penalty, gap_extend_penalty)
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
        aligned_i, aligned_j = _nw(cons_i, cons_j, sub_matrix, gap_open_penalty, gap_extend_penalty)

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
                score = _nw_score(cons_new, cons_k, sub_matrix, gap_open_penalty, gap_extend_penalty)
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
                sequences[name], consensus, sub_matrix, gap_open_penalty, gap_extend_penalty
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
    match_score: int,
    mismatch_penalty: int,
) -> np.ndarray:
    """Build substitution matrix from motif_distance and motif_length.

    Score formula::

        score = (avg_motif_length - distance) * match_score + distance * mismatch_penalty

    Positive scores reward alignment, negative scores penalise it.
    """
    import numpy as np

    n_motifs = len(adata.var)
    dist_mat = adata.varp["motif_distance"]
    motif_lengths = adata.var["motif_length"].to_numpy()

    sub_matrix = np.zeros((n_motifs, n_motifs), dtype=int)
    for i in range(n_motifs):
        for j in range(n_motifs):
            dist = dist_mat[i, j]
            avg_len = (motif_lengths[i] + motif_lengths[j]) / 2.0
            sub_matrix[i, j] = int(round((avg_len - dist) * match_score + dist * mismatch_penalty))

    return sub_matrix


def _nw_score(
    seq_a: List[str],
    seq_b: List[str],
    sub_matrix: np.ndarray,
    gap_open_penalty: int,
    gap_extend_penalty: int,
) -> float:
    """Needleman-Wunsch optimal alignment score (affine gap)."""
    import numpy as np

    n = len(seq_a)
    m = len(seq_b)
    NEG_INF = -10 ** 9
    go = int(gap_open_penalty)
    ge = int(gap_extend_penalty)

    M = np.zeros((n + 1, m + 1), dtype=int)
    I = np.full((n + 1, m + 1), NEG_INF, dtype=int)
    D = np.full((n + 1, m + 1), NEG_INF, dtype=int)

    for i in range(1, n + 1):
        I[i, 0] = go + (i - 1) * ge

    for j in range(1, m + 1):
        D[0, j] = go + (j - 1) * ge

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = sub_matrix[int(seq_a[i - 1]), int(seq_b[j - 1])]
            M[i, j] = max(M[i - 1, j - 1], I[i - 1, j - 1], D[i - 1, j - 1]) + score
            I[i, j] = max(M[i - 1, j] + go, I[i - 1, j] + ge)
            D[i, j] = max(M[i, j - 1] + go, D[i, j - 1] + ge)

    return float(max(M[n, m], I[n, m], D[n, m]))


def _nw(
    seq_a: List[str],
    seq_b: List[str],
    sub_matrix: np.ndarray,
    gap_open_penalty: int,
    gap_extend_penalty: int,
) -> Tuple[List[str], List[str]]:
    """Needleman-Wunsch global alignment with affine gap penalty."""
    import numpy as np

    n = len(seq_a)
    m = len(seq_b)
    NEG_INF = -10 ** 9
    go = int(gap_open_penalty)
    ge = int(gap_extend_penalty)

    M = np.zeros((n + 1, m + 1), dtype=int)
    I = np.full((n + 1, m + 1), NEG_INF, dtype=int)
    D = np.full((n + 1, m + 1), NEG_INF, dtype=int)

    for i in range(1, n + 1):
        I[i, 0] = go + (i - 1) * ge

    for j in range(1, m + 1):
        D[0, j] = go + (j - 1) * ge

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = sub_matrix[int(seq_a[i - 1]), int(seq_b[j - 1])]
            M[i, j] = max(M[i - 1, j - 1], I[i - 1, j - 1], D[i - 1, j - 1]) + score
            I[i, j] = max(M[i - 1, j] + go, I[i - 1, j] + ge)
            D[i, j] = max(M[i, j - 1] + go, D[i, j - 1] + ge)

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
        # Boundary protection: force correct path at edges
        if i > 0 and j > 0:
            pass
        elif i > 0:
            curr = "I"
        elif j > 0:
            curr = "D"
        else:
            break

        if curr == "M":
            aligned_a.append(seq_a[i - 1])
            aligned_b.append(seq_b[j - 1])
            score = sub_matrix[int(seq_a[i - 1]), int(seq_b[j - 1])]
            prev_val = M[i, j] - score
            if i > 0 and j > 0 and M[i - 1, j - 1] == prev_val:
                curr = "M"
            elif i > 0 and j > 0 and I[i - 1, j - 1] == prev_val:
                curr = "I"
            else:
                curr = "D"
            i -= 1
            j -= 1
        elif curr == "I":
            aligned_a.append(seq_a[i - 1])
            aligned_b.append("-")
            if i > 0 and M[i - 1, j] + go == I[i, j]:
                curr = "M"
            else:
                curr = "I"
            i -= 1
        else:  # "D"
            aligned_a.append("-")
            aligned_b.append(seq_b[j - 1])
            if j > 0 and M[i, j - 1] + go == D[i, j]:
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
