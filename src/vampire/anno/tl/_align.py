from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import anndata as ad

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Function call-graph overview
# =============================================================================
#
# [PUBLIC] sample_msa  — sample-level progressive MSA
#   Input:  adata.uns["motif_array"], adata.uns["orientation_array"],
#           adata.varp["motif_distance"], adata.var["motif_length"]
#   Output: adata.uns["sample_msa_motif_array"],
#           adata.uns["sample_msa_orientation_array"],
#           adata.uns["sample_msa_consensus"],
#           adata.uns["sample_msa_consensus_orientation"]
#
#   ├─ Deduplication: collapse identical (motif, orientation) sequences
#   ├─ _build_sub_matrix(score, penalty) → (sub_matrix, rc_sub_matrix)
#   │   ├─ sub_matrix  : forward-forward substitution scores
#   │   └─ rc_sub_matrix: forward-rc substitution scores (from rc_motif_distance)
#   │   └─ score = (avg_len - dist) * match_score + dist * mismatch_penalty
#   ├─ Orientation encoding (before _msa_core):
#   │   Motif IDs are offset by +n_motifs when orientation == "-".
#   │   This lets the generic NW engine pick the correct matrix cell:
#   │     forward  vs forward  → sub_matrix[fw, fw]
#   │     forward  vs rc       → rc_sub_matrix[fw, fw]
#   │     rc       vs rc       → sub_matrix[fw, fw]  (same relative strand)
#   │   The substitution matrix is extended to 2n × 2n blocks:
#   │         0..n-1   n..2n-1
#   │       ┌────────┬────────┐
#   │   0..n│  sub   │ rc_sub │
#   │       ├────────┼────────┤
#   │ n..2n │rc_subᵀ │  sub   │
#   │       └────────┴────────┘
#   ├─ _msa_core(adjusted_sequences, extended_sub_matrix, ...) → (aligned, consensus)
#   │   ├─ _nw_score(seq_a, seq_b, ...) → float
#   │   ├─ UPGMA tree from pairwise distance matrix
#   │   ├─ Bottom-up merge:
#   │   │   ├─ _profile_consensus(profile) → consensus_seq
#   │   │   ├─ _nw(cons_a, cons_b, ...) → (aligned_a, aligned_b)
#   │   │   └─ _merge_profiles(prof_a, prof_b, aln_a, aln_b) → merged
#   │   └─ Iterative refinement (if refine=True):
#   │       └─ Re-align each raw sequence vs consensus via _nw(...)
#   └─ Restore: map alignment back to all original samples
#       ├─ Motif IDs  : mod n_motifs to strip the orientation offset
#       └─ orientation_array: rebuilt with gaps inserted, original strand kept
#
# [PUBLIC] motif_msa  — motif-level MSA or pairwise reference alignment
#   Input:  adata.var["motif"]  (DNA sequences)
#   Output: adata.uns["motif_msa"]  (alignment + consensus / variants)
#
#   ├─ If reference is None  →  MSA mode (reuse _msa_core)
#   │   ├─ Map ACGT → string indices (tokens)
#   │   ├─ Build 4×4 DNA substitution matrix
#   │   ├─ _msa_core(tokens, sub_matrix, ...) → (aligned_tokens, consensus)
#   │   │   └─ Same engine as sample_msa (see above)
#   │   └─ Map tokens → ACGT strings
#   │
#   └─ If reference given  →  Pairwise mode
#       ├─ parasail.nw_trace_striped_16(seq, ref_seq, ...) → traceback
#       └─ _pairwise_alignment_to_variants(ref_aln, seq_aln, ref, id) → variant records
#           ├─ Match        → skip
#           ├─ Substitution → type="sub", pos, ref, alt
#           ├─ Insertion    → type="ins", pos, seq
#           └─ Deletion     → type="del", pos, ref, length
#
# =============================================================================


def sample_msa(
    adata: ad.AnnData,
    *,
    match_score: int = 2,
    mismatch_penalty: int = -3,
    gap_open_penalty: int = -5,
    gap_extend_penalty: int = -1,
    refine: bool = True,
    max_refine_iter: int = 3,
    store_key: str = "aligned",
) -> ad.AnnData:
    """
    Multiple sequence alignment of motif arrays across samples.

    Uses a guide-tree progressive alignment strategy: build a UPGMA tree
    from pairwise distances, then merge profiles bottom-up.  Optionally
    refines the MSA via iterative consensus realignment.

    Duplicate sequences are automatically deduplicated before alignment
    and mapped back afterwards.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with ``motif_array`` and ``orientation_array`` in
        ``uns``, and ``motif_distance`` in ``varp``.
    match_score : int, default=2
        Reward coefficient for matching motifs. The substitution score is
        ``(avg_len - distance) * match_score + distance * mismatch_penalty``,
        where ``avg_len`` is the average motif length of the two motifs.
    mismatch_penalty : int, default=-3
        Penalty coefficient for mismatched motifs.
    gap_open_penalty : int, default=-5
        Penalty for opening a gap.
    gap_extend_penalty : int, default=-1
        Penalty for extending a gap.
    refine : bool, default=True
        Whether to perform iterative consensus-based refinement
        after the initial progressive alignment.
    max_refine_iter : int, default=3
        Maximum number of refinement iterations.  Ignored when
        ``refine=False``.
    store_key : str, default="aligned"
        Key prefix for storing results in ``adata.uns``.
        Stores ``{store_key}_motif_array``,
        ``{store_key}_orientation_array``,
        ``{store_key}_consensus``, and
        ``{store_key}_consensus_orientation``.

    Returns
    -------
    ad.AnnData
        The updated AnnData with alignment results in ``uns``.

    Examples
    --------
    >>> import vampire as vp
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.tl.sample_msa(adata)
    """
    from copy import deepcopy
    import numpy as np
    from ..pp._markdup import markdup

    if match_score <= 0:
        raise ValueError("match_score should be positive.")
    if mismatch_penalty >= 0:
        raise ValueError("mismatch_penalty should be negative.")
    if gap_open_penalty >= 0:
        raise ValueError("gap_open_penalty should be negative.")
    if gap_extend_penalty >= 0:
        raise ValueError("gap_extend_penalty should be negative.")
    if refine and max_refine_iter <= 0:
        raise ValueError("max_refine_iter must be positive when refine=True.")

    if "motif_array" not in adata.uns:
        raise KeyError("adata.uns['motif_array'] not found.")
    if "orientation_array" not in adata.uns:
        raise KeyError("adata.uns['orientation_array'] not found.")
    if "motif_distance" not in adata.varp:
        raise KeyError("adata.varp['motif_distance'] not found.")
    if "motif_length" not in adata.var:
        raise KeyError("adata.var['motif_length'] not found.")

    sequences: dict[str, list[str]] = adata.uns["motif_array"]
    orientations: dict[str, list[str]] = adata.uns["orientation_array"]
    all_names: list[str] = list(sequences.keys())
    n_total: int = len(all_names)

    if n_total == 0:
        return adata
    if n_total == 1:
        adata.uns[f"{store_key}_motif_array"] = deepcopy(sequences)
        adata.uns[f"{store_key}_orientation_array"] = deepcopy(orientations)
        adata.obs["unique_group"] = 0
        return adata

    # Deduplication: identical (motif, orientation) sequences are collapsed
    if "unique_group" not in adata.obs.columns:
        logger.warning(
            "unique_group not found in adata.obs. "
            "vp.anno.pp.markdup() has not been run. Running it automatically."
        )
        adata = markdup(adata)
    
    name_to_group: dict[str, int] = adata.obs["unique_group"].to_dict()
    group_to_names: dict[int, list[str]] = {}
    unique_names: list[str] = []
    seen_groups: set[int] = set()
    for name, group in name_to_group.items():
        group_to_names.setdefault(group, []).append(name)
        if group not in seen_groups:
            seen_groups.add(group)
            unique_names.append(name)
    n_unique: int = len(unique_names)

    # Build rep_to_names for map-back after MSA (rep key = first sample name)
    rep_to_names = {
        name: group_to_names[name_to_group[name]] for name in unique_names
    }
    logger.info(
        "Aligning %d samples: %d unique sequences (%d duplicates removed).",
        n_total,
        n_unique,
        n_total - n_unique,
    )

    # Build substitution matrices (forward-forward and forward-rc)
    sub_matrix, rc_sub_matrix = _build_sub_matrix(
        adata, match_score, mismatch_penalty
    )

    # Extend substitution matrix to encode orientation in motif IDs:
    #   0..n-1          → forward motifs
    #   n..2n-1         → reverse-complement motifs
    n_motifs = len(adata.var)
    extended_sub_matrix = np.zeros((2 * n_motifs, 2 * n_motifs), dtype=int)
    extended_sub_matrix[:n_motifs, :n_motifs] = sub_matrix
    extended_sub_matrix[:n_motifs, n_motifs:] = rc_sub_matrix
    extended_sub_matrix[n_motifs:, :n_motifs] = rc_sub_matrix.T
    extended_sub_matrix[n_motifs:, n_motifs:] = sub_matrix

    # Adjust sequences so that reverse-oriented motifs get offset IDs
    unique_sequences: dict[str, list[str]] = {}
    for name in unique_names:
        seq = sequences[name]
        ori = orientations[name]
        adjusted = [
            str(int(m) + n_motifs) if o == "-" else m
            for m, o in zip(seq, ori)
        ]
        unique_sequences[name] = adjusted

    # Run generic MSA engine on orientation-aware sequences
    result_motifs, result_consensus = _msa_core(
        unique_sequences,
        extended_sub_matrix,
        gap_open_penalty,
        gap_extend_penalty,
        refine=refine,
        max_refine_iter=max_refine_iter,
    )

    # Map alignment results back to all original samples (including duplicates)
    final_motifs: dict[str, list[str]] = {}
    final_oris: dict[str, list[str]] = {}
    for rep_name in unique_names:
        rep_motif_adj = result_motifs[rep_name]
        # Map extended IDs back to original IDs
        rep_motif: list[str] = [
            "-" if m == "-" else str(int(m) % n_motifs)
            for m in rep_motif_adj
        ]
        # Map orientation (preserving original strand info)
        rep_ori: list[str] = []
        ori_idx = 0
        ori_src = orientations[rep_name]
        for m in rep_motif_adj:
            if m == "-":
                rep_ori.append("-")
            else:
                rep_ori.append(ori_src[ori_idx])
                ori_idx += 1

        for orig_name in rep_to_names[rep_name]:
            final_motifs[orig_name] = rep_motif
            final_oris[orig_name] = rep_ori

    # Build consensus with orientation
    consensus_motifs: list[str] = []
    consensus_oris: list[str] = []
    for m in result_consensus:
        if m == "-":
            consensus_motifs.append("-")
            consensus_oris.append("-")
        else:
            mid = int(m)
            consensus_motifs.append(str(mid % n_motifs))
            consensus_oris.append("-" if mid >= n_motifs else "+")

    adata.uns[f"{store_key}_motif_array"] = final_motifs
    adata.uns[f"{store_key}_orientation_array"] = final_oris
    adata.uns[f"{store_key}_consensus"] = consensus_motifs
    adata.uns[f"{store_key}_consensus_orientation"] = consensus_oris

    logger.info(
        "Aligned %d samples (%d unique). "
        "Original lengths: %s. "
        "Aligned length: %d.",
        n_total,
        n_unique,
        [len(sequences[n]) for n in all_names],
        len(final_motifs[all_names[0]]),
    )

    return adata


def _build_sub_matrix(
    adata: ad.AnnData,
    match_score: int,
    mismatch_penalty: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build substitution matrices from motif_distance, rc_motif_distance and motif_length.

    Returns both the forward-forward matrix and the forward-rc (rc-forward)
    matrix so that ``sample_msa`` can respect per-motif orientation.

    Score formula::

        score = (avg_len - distance) * match_score + distance * mismatch_penalty

    where ``avg_len`` is the average motif length of the two motifs.
    """
    import numpy as np

    n_motifs = len(adata.var)
    dist_mat = adata.varp["motif_distance"]
    rc_dist_mat = adata.varp.get("rc_motif_distance", dist_mat)
    motif_lengths = adata.var["motif_length"].to_numpy()

    sub_matrix = np.zeros((n_motifs, n_motifs), dtype=int)
    rc_sub_matrix = np.zeros((n_motifs, n_motifs), dtype=int)
    for i in range(n_motifs):
        for j in range(n_motifs):
            dist = dist_mat[i, j]
            rc_dist = rc_dist_mat[i, j]
            avg_len = (motif_lengths[i] + motif_lengths[j]) / 2.0
            sub_matrix[i, j] = int(
                round((avg_len - dist) * match_score + dist * mismatch_penalty)
            )
            rc_sub_matrix[i, j] = int(
                round((avg_len - rc_dist) * match_score + rc_dist * mismatch_penalty)
            )

    return sub_matrix, rc_sub_matrix


def _nw_score(
    seq_a: list[str],
    seq_b: list[str],
    sub_matrix: np.ndarray,
    gap_open_penalty: int,
    gap_extend_penalty: int,
) -> float:
    """Needleman-Wunsch optimal alignment score (affine gap)."""
    import numpy as np

    n = len(seq_a)
    m = len(seq_b)
    NEG_INF = -10**9
    go = int(gap_open_penalty)
    ge = int(gap_extend_penalty)

    M = np.full((n + 1, m + 1), NEG_INF, dtype=int)
    M[0, 0] = 0
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
    seq_a: list[str],
    seq_b: list[str],
    sub_matrix: np.ndarray,
    gap_open_penalty: int,
    gap_extend_penalty: int,
) -> tuple[list[str], list[str]]:
    """Needleman-Wunsch global alignment with affine gap penalty."""
    import numpy as np

    n = len(seq_a)
    m = len(seq_b)
    NEG_INF = -10**9
    go = int(gap_open_penalty)
    ge = int(gap_extend_penalty)

    M = np.full((n + 1, m + 1), NEG_INF, dtype=int)
    M[0, 0] = 0
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
    aligned_a: list[str] = []
    aligned_b: list[str] = []

    i, j = n, m
    curr_score = max(M[i, j], I[i, j], D[i, j])
    if curr_score == M[i, j]:
        curr = "M"
    elif curr_score == I[i, j]:
        curr = "I"
    else:
        curr = "D"

    while i > 0 or j > 0:
        if i == 0:
            curr = "D"
        elif j == 0:
            curr = "I"

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


def _profile_consensus(profile: list[list[str]]) -> list[str]:
    """Build consensus sequence from a profile (all-gap columns are skipped)."""
    from collections import Counter

    consensus: list[str] = []
    for col in profile:
        motifs = [m for m in col if m != "-"]
        if motifs:
            consensus.append(Counter(motifs).most_common(1)[0][0])
    return consensus


def _merge_profiles(
    profile_a: list[list[str]],
    profile_b: list[list[str]],
    aligned_a: list[str],
    aligned_b: list[str],
) -> list[list[str]]:
    """Merge two profiles based on aligned consensus sequences."""
    merged: list[list[str]] = []
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


def _find_homopolymers(consensus: list[str]) -> list[tuple[int, int]]:
    """Find contiguous runs of identical non-gap tokens in consensus.

    Returns list of (start, end) half-open intervals where each run
    has length >= 2 and consensus[start] == ... == consensus[end-1] != "-".
    """
    if not consensus:
        return []
    regions: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(consensus)):
        if consensus[i] != consensus[i - 1] or consensus[i] == "-":
            if i - start >= 2 and consensus[start] != "-":
                regions.append((start, i))
            start = i
    if len(consensus) - start >= 2 and consensus[start] != "-":
        regions.append((start, len(consensus)))
    return regions


def _find_homopolymers_ignore_gaps(consensus: list[str]) -> list[tuple[int, int]]:
    """Find homopolymer runs, ignoring gaps when computing length.

    Gaps inside a run do **not** split it, so ``A-A-A-A`` is treated as a
    single run of 4 A's spanning positions 0..7.
    """
    if not consensus:
        return []
    no_gap = [(i, c) for i, c in enumerate(consensus) if c != "-"]
    if len(no_gap) < 2:
        return []
    regions: list[tuple[int, int]] = []
    start_idx = 0
    for i in range(1, len(no_gap)):
        if no_gap[i][1] != no_gap[i - 1][1]:
            if i - start_idx >= 2:
                regions.append((no_gap[start_idx][0], no_gap[i - 1][0] + 1))
            start_idx = i
    if len(no_gap) - start_idx >= 2:
        regions.append((no_gap[start_idx][0], no_gap[-1][0] + 1))
    return regions


def _reposition_homopolymer_insertions(
    ref_aln: list[str],
    query_aln: list[str],
    consensus: list[str],
    divergence_scores: list[int] | None = None,
) -> list[str]:
    """Move same-base insertions to the most divergent positions.

    Only acts when the inserted nucleotide is identical to the homopolymer
    consensus base (e.g. inserting A into a poly-A run).  In that case the
    insertion can slide freely because the query base already matches the
    consensus; only the ref gap needs to be relocated.

    Parameters
    ----------
    ref_aln
        Aligned reference sequence (may contain gaps ``'-'``).
    query_aln
        Aligned query sequence.
    consensus
        Consensus sequence (same length as ``ref_aln`` / ``query_aln``).
    divergence_scores
        Optional per-position divergence counts.  If ``None`` they are
        computed on the fly from deletion / substitution positions.

    Returns
    -------
    list[str]
        Updated reference alignment with repositioned gaps.
    """
    regions = _find_homopolymers_ignore_gaps(consensus)
    if not regions:
        return list(ref_aln)

    new_ref = list(ref_aln)

    for start, end in regions:
        # Identify the consensus base for this run (skip any leading gap)
        base = consensus[start]
        while base == "-" and start < end:
            start += 1
            base = consensus[start]
        if base == "-":
            continue

        # Build divergence scores from non-insertion variants
        scores = [0] * (end - start)
        if divergence_scores is not None:
            scores = divergence_scores[start:end]
        else:
            for j in range(start, end):
                if new_ref[j] != "-" and query_aln[j] != consensus[j]:
                    scores[j - start] += 1

        # Locate consecutive same-base insertions in this region
        i = start
        while i < end:
            if new_ref[i] == "-" and query_aln[i] == base:
                ins_start = i
                while i < end and new_ref[i] == "-" and query_aln[i] == base:
                    i += 1
                ins_end = i
                ins_len = ins_end - ins_start

                # Find the best window (highest sum of divergence scores)
                best_pos = ins_start
                best_score = -1
                for pos in range(start, end - ins_len + 1):
                    # Skip windows that overlap existing gaps
                    if any(new_ref[p] == "-" for p in range(pos, pos + ins_len)):
                        continue
                    score = sum(scores[pos - start : pos - start + ins_len])
                    if score > best_score:
                        best_score = score
                        best_pos = pos

                if best_pos != ins_start:
                    # Restore bases at old insertion site
                    for j in range(ins_start, ins_end):
                        new_ref[j] = base
                    # Place gaps at new site
                    for j in range(best_pos, best_pos + ins_len):
                        new_ref[j] = "-"
            else:
                i += 1

    return new_ref


def _reposition_homopolymer_insertions_in_variants(
    variants_df: "pl.DataFrame",
    ref_seq: str,
) -> "pl.DataFrame":
    """Move same-base insertion variants to the most divergent positions.

    Acts on insertions where every inserted base matches the reference base
    at or next to the insertion point. This covers two cases:

    1. Inside a pre-existing homopolymer.
    2. Adjacent to a matching singleton base, creating a new homopolymer.

    Such insertions can slide freely within the effective homopolymer region.
    They are repositioned to the window with the highest divergence from
    other samples.
    """
    import polars as pl

    if variants_df.is_empty():
        return variants_df

    # Build homopolymer lookup
    regions = _find_homopolymers(list(ref_seq))

    pos_to_region: dict[int, tuple[int, int]] = {}

    for start, end in regions:
        for p in range(start, end):
            pos_to_region[p] = (start, end)

    # Helper: determine whether an insertion is slidable and return its effective sliding region.
    def _get_effective_region(
        pos: int,
        seq: str,
    ) -> tuple[int, int] | None:
        if not seq:
            return None

        # only homopolymer insertions can slide
        if len(set(seq)) != 1:
            return None

        base = seq[0]
        ins_len = len(seq)

        # Case 1: insertion occurs inside an existing homopolymer
        region = pos_to_region.get(pos)

        if region is not None:
            start, end = region

            if (
                ref_seq[start] == base
                and end - start >= ins_len
            ):
                return region

            return None

        # Case 2: insertion is adjacent to matching base(s)
        matches_next = (
            0 <= pos < len(ref_seq)
            and ref_seq[pos] == base
        )

        matches_prev = (
            0 < pos <= len(ref_seq)
            and ref_seq[pos - 1] == base
        )

        if not matches_next and not matches_prev:
            return None

        region_start = pos
        region_end = pos

        if matches_prev:
            region_start = pos - 1

            while (
                region_start > 0
                and ref_seq[region_start - 1] == base
            ):
                region_start -= 1

        if matches_next:
            region_end = pos + 1

            while (
                region_end < len(ref_seq)
                and ref_seq[region_end] == base
            ):
                region_end += 1

        if region_end - region_start < ins_len:
            return None

        return region_start, region_end

    # Compute divergence map
    divergence: dict[int, int] = {}

    for row in variants_df.iter_rows(named=True):

        pos = row["pos"]
        var_type = row["type"]

        if var_type in {"sub", "del"}:

            length = row.get("length", 1)

            for i in range(length):
                divergence[pos + i] = (
                    divergence.get(pos + i, 0) + 1
                )

        elif var_type == "ins":

            seq = row.get("seq", "") or ""

            # Ignore homopolymer-expanding insertions
            if _get_effective_region(pos, seq) is not None:
                continue

            divergence[pos] = (
                divergence.get(pos, 0)
                + max(1, len(seq))
            )

    # Reposition slidable insertions
    new_rows: list[dict] = []

    for row in variants_df.iter_rows(named=True):

        if row["type"] != "ins":
            new_rows.append(row)
            continue

        pos = row["pos"]
        seq = row.get("seq", "") or ""

        region = _get_effective_region(pos, seq)

        if region is None:
            new_rows.append(row)
            continue

        start, end = region
        ins_len = len(seq)

        best_pos = pos
        best_score = sum(
            divergence.get(pos + i, 0)
            for i in range(ins_len)
        )

        for candidate in range(
            start,
            end - ins_len + 1 + 1,
        ):

            score = sum(
                divergence.get(candidate + i, 0)
                for i in range(ins_len)
            )

            if score > best_score or (
                score == best_score and candidate < best_pos
            ):
                best_score = score
                best_pos = candidate

        if best_pos != pos:
            new_row = dict(row)
            new_row["pos"] = best_pos
            new_rows.append(new_row)
        else:
            new_rows.append(row)

    return pl.DataFrame(new_rows)


def _squeeze_homopolymer(
    sub_seq: list[str],
    sub_consensus: list[str],
    anchor: int | None = None,
) -> list[str]:
    """Squeeze gaps and mismatches into a contiguous block.

    Within a homopolymer region every position is equivalent for scoring,
    so we can freely slide anomalies.  The greedy strategy:
    1. Collect gaps and mismatches (including insertions).
    2. Choose the anchor position (global median across samples if provided,
       otherwise the local median of this single sequence).
    3. Build a contiguous block [gap...gap, mismatch...mismatch] centred
       on the anchor.
    4. Fill remaining positions with the consensus base.

    This maximises gap continuity (fewer gap-open penalties) and clusters
    all anomalies at a single locus.
    """
    n = len(sub_seq)
    if n < 2:
        return sub_seq

    base = sub_consensus[0]

    # Classify positions
    gaps = [i for i, s in enumerate(sub_seq) if s == "-"]
    mismatches = [
        s for i, (s, c) in enumerate(zip(sub_seq, sub_consensus))
        if s != c and s != "-"
    ]
    n_gap = len(gaps)
    n_mm = len(mismatches)

    if n_gap + n_mm == 0:
        return sub_seq

    # Anchor: use provided global anchor, or fall back to local median
    if anchor is None:
        anomaly_pos = gaps + [
            i for i, (s, c) in enumerate(zip(sub_seq, sub_consensus))
            if s != c and s != "-"
        ]
        import numpy as np
        anchor = int(np.median(anomaly_pos))

    # Build anomaly block: gaps first, then mismatches
    block = ["-"] * n_gap + mismatches
    block_start = max(0, min(anchor - len(block) // 2, n - len(block)))

    new_sub = [base] * n
    for i, char in enumerate(block):
        new_sub[block_start + i] = char
    return new_sub


def _squeeze_homopolymer_regions(
    aligned_seq: list[str],
    consensus: list[str],
    global_anchors: dict[tuple[int, int], int] | None = None,
) -> list[str]:
    """Apply _squeeze_homopolymer to every homopolymer region in consensus.

    Parameters
    ----------
    aligned_seq
        The aligned sequence (may contain gaps).
    consensus
        The consensus sequence (no gaps within homopolymer regions).
    global_anchors
        Optional mapping from (start, end) region tuple to a pre-computed
        anchor position.  When provided, all sequences in the same region
        use the same anchor, forcing cross-sample anomaly alignment.
    """
    regions = _find_homopolymers(consensus)
    if not regions:
        return aligned_seq

    new_seq = list(aligned_seq)
    for start, end in regions:
        anchor = (
            global_anchors.get((start, end))
            if global_anchors is not None
            else None
        )
        new_seq[start:end] = _squeeze_homopolymer(
            new_seq[start:end], consensus[start:end], anchor
        )
    return new_seq


def _align_homopolymer_variants(
    aligned_seq: list[str],
    consensus: list[str],
    global_anchors: dict[tuple[int, int], int] | None = None,
) -> list[str]:
    """Align variant positions within homopolymer regions to common anchors.

    For each homopolymer region, slides the anomaly block so that the
    first anomaly aligns with the global anchor, preserving the original
    order and relative spacing of anomalies.

    Parameters
    ----------
    aligned_seq
        The aligned sequence (may contain gaps).
    consensus
        The consensus sequence (no gaps within homopolymer regions).
    global_anchors
        Mapping from (start, end) region tuple to a pre-computed anchor
        position.  All sequences in the same region use the same anchor,
        forcing cross-sample variant alignment.

    Returns
    -------
    list[str]
        Sequence with normalized variant positions.
    """
    regions = _find_homopolymers(consensus)
    if not regions:
        return list(aligned_seq)

    new_seq = list(aligned_seq)
    for start, end in regions:
        anchor = (
            global_anchors.get((start, end))
            if global_anchors is not None
            else None
        )

        sub_seq = new_seq[start:end]
        sub_cons = consensus[start:end]
        n = len(sub_seq)
        if n < 2:
            continue
        base = sub_cons[0]

        # Collect anomalies in original order
        anomalies: list[tuple[int, str]] = []
        for i, (s, c) in enumerate(zip(sub_seq, sub_cons)):
            if s != c:
                anomalies.append((i, s))

        if not anomalies:
            continue

        # Determine anchor: use global anchor or local median of first anomalies
        if anchor is None:
            import numpy as np

            anchor = int(np.median([pos for pos, _ in anomalies]))

        # Compute offset to align block *center* with anchor
        first_pos = anomalies[0][0]
        block_len = anomalies[-1][0] - anomalies[0][0] + 1
        block_center = (anomalies[0][0] + anomalies[-1][0]) / 2
        offset = round(anchor - block_center)

        # Ensure the anomaly block fits within the region after sliding
        if first_pos + offset < 0:
            offset = -first_pos
        if first_pos + offset + block_len > n:
            offset = n - block_len - first_pos

        # Apply slide
        new_sub = [base] * n
        for orig_pos, char in anomalies:
            new_pos = orig_pos + offset
            if 0 <= new_pos < n:
                new_sub[new_pos] = char

        new_seq[start:end] = new_sub

    return new_seq


def _msa_core(
    sequences: dict[str, list[str]],
    sub_matrix: np.ndarray,
    gap_open_penalty: int,
    gap_extend_penalty: int,
    refine: bool = True,
    max_refine_iter: int = 3,
) -> tuple[dict[str, list[str]], list[str]]:
    """
    Generic progressive MSA engine.

    Parameters
    ----------
    sequences
        Mapping from sequence name to list of token strings.
    sub_matrix
        Substitution score matrix.
    gap_open_penalty
        Gap open penalty (negative).
    gap_extend_penalty
        Gap extension penalty (negative).
    refine
        Whether to perform iterative consensus refinement.
    max_refine_iter
        Maximum refinement iterations.

    Returns
    -------
    aligned_sequences
        Mapping from name to aligned token list.
    consensus
        Consensus token list.
    """
    import numpy as np
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform

    names = list(sequences.keys())
    n = len(names)

    if n == 0:
        return {}, []
    if n == 1:
        seq = sequences[names[0]]
        return {names[0]: list(seq)}, list(seq)

    # Initialize profiles
    profiles: dict[int, list[list[str]]] = {
        i: [[sequences[names[i]][k]] for k in range(len(sequences[names[i]]))]
        for i in range(n)
    }
    seq_indices: dict[int, list[int]] = {i: [i] for i in range(n)}

    # Compute pairwise distances
    score_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            score = _nw_score(
                sequences[names[i]],
                sequences[names[j]],
                sub_matrix,
                gap_open_penalty,
                gap_extend_penalty,
            )
            score_mat[i, j] = score_mat[j, i] = score

    max_score = float(score_mat.max())
    dist_mat = max_score - score_mat
    np.fill_diagonal(dist_mat, 0)

    # UPGMA progressive merge
    condensed_dist = squareform(dist_mat, checks=False)
    Z = linkage(condensed_dist, method="average")
    root = to_tree(Z, rd=False)

    profiles_cache: dict[int, list[list[str]]] = {}
    seq_indices_cache: dict[int, list[int]] = {}

    def _merge_node(node) -> int:
        if node.is_leaf():
            idx = node.id
            profiles_cache[idx] = profiles[idx]
            seq_indices_cache[idx] = seq_indices[idx]
            return idx

        left_idx = _merge_node(node.get_left())
        right_idx = _merge_node(node.get_right())

        cons_left = _profile_consensus(profiles_cache[left_idx])
        cons_right = _profile_consensus(profiles_cache[right_idx])
        aligned_left, aligned_right = _nw(
            cons_left,
            cons_right,
            sub_matrix,
            gap_open_penalty,
            gap_extend_penalty,
        )

        merged = _merge_profiles(
            profiles_cache[left_idx],
            profiles_cache[right_idx],
            aligned_left,
            aligned_right,
        )
        merged_indices = seq_indices_cache[left_idx] + seq_indices_cache[right_idx]

        new_idx = node.id
        profiles_cache[new_idx] = merged
        seq_indices_cache[new_idx] = merged_indices
        return new_idx

    final_id = _merge_node(root)
    final_profile = profiles_cache[final_id]
    final_indices = seq_indices_cache[final_id]

    # Build aligned arrays
    aligned: dict[str, list[str]] = {}
    for pos, seq_idx in enumerate(final_indices):
        name = names[seq_idx]
        aligned[name] = [col[pos] for col in final_profile]

    # Iterative refinement: re-align raw sequences to consensus
    if refine:
        current = aligned
        consensus: list[str] | None = None

        for iteration in range(max_refine_iter):
            msa_len = len(current[names[0]])
            temp_profile = [
                [current[name][pos] for name in names]
                for pos in range(msa_len)
            ]
            consensus = _profile_consensus(temp_profile)
            old_consensus = consensus

            # Re-align every raw sequence to the consensus
            new_aligned: dict[str, list[str]] = {}
            for name in names:
                aligned_seq, _ = _nw(
                    sequences[name],
                    consensus,
                    sub_matrix,
                    gap_open_penalty,
                    gap_extend_penalty,
                )
                new_aligned[name] = aligned_seq

            # Rebuild consensus from new alignments
            new_msa_len = len(new_aligned[names[0]])
            new_profile = [
                [new_aligned[name][pos] for name in names]
                for pos in range(new_msa_len)
            ]
            new_consensus = _profile_consensus(new_profile)

            if new_consensus == old_consensus:
                logger.info(
                    "Refinement converged after %d iteration(s).",
                    iteration + 1,
                )
                current = new_aligned
                break

            current = new_aligned
        else:
            logger.info(
                "Refinement completed after %d iterations.",
                max_refine_iter,
            )
            # Build final consensus if loop exited without convergence
            msa_len = len(current[names[0]])
            temp_profile = [
                [current[name][pos] for name in names]
                for pos in range(msa_len)
            ]
            consensus = _profile_consensus(temp_profile)

        result = current
    else:
        result = aligned
        consensus = _profile_consensus(final_profile)

    # ---- One-pass homopolymer variant position alignment ----
    regions = _find_homopolymers(consensus)
    global_anchors: dict[tuple[int, int], int] = {}
    for start, end in regions:
        mismatch_pos: list[int] = []
        all_anomaly_pos: list[int] = []
        for name in names:
            sub_seq = result[name][start:end]
            sub_cons = consensus[start:end]
            for i, (s, c) in enumerate(zip(sub_seq, sub_cons)):
                if s != c:
                    all_anomaly_pos.append(i)
                    if s != "-":
                        mismatch_pos.append(i)
        if not all_anomaly_pos:
            continue

        # Density-priority anchor: choose the position with the most anomalies
        # (gaps + mismatches).  If tied, prefer positions that have mismatches.
        from collections import Counter
        pos_counts = Counter(all_anomaly_pos)
        max_count = max(pos_counts.values())
        candidates = [p for p, c in pos_counts.items() if c == max_count]
        if len(candidates) == 1:
            anchor = candidates[0]
        else:
            mismatch_set = set(mismatch_pos)
            mismatch_candidates = [p for p in candidates if p in mismatch_set]
            if mismatch_candidates:
                anchor = mismatch_candidates[0]
            else:
                anchor = candidates[0]
        global_anchors[(start, end)] = anchor

    # Normalize variant positions
    normalized: dict[str, list[str]] = {}
    for name in names:
        normalized[name] = _align_homopolymer_variants(
            result[name],
            consensus,
            global_anchors,
        )

    # Rebuild consensus after normalization
    norm_msa_len = len(normalized[names[0]])
    norm_profile = [
        [normalized[name][pos] for name in names]
        for pos in range(norm_msa_len)
    ]
    result_consensus = _profile_consensus(norm_profile)

    return normalized, result_consensus


def motif_msa(
    adata: ad.AnnData,
    reference: str | int | None = None,
    *,
    store_key: str = "motif_msa",
    match_score: int = 2,
    mismatch_penalty: int = -3,
    gap_open_penalty: int = -5,
    gap_extend_penalty: int = -1,
) -> ad.AnnData:
    """
    Align motif sequences using progressive MSA or pairwise reference alignment.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with ``var["motif"]`` containing motif sequences.
    reference : str | int | None, default=None
        Reference motif. If ``None``, performs a progressive MSA of
        all motifs. If an ``int`` or ``str``, performs pairwise alignments
        of each motif against the specified reference.
    store_key : str, default="motif_msa"
        Key under which results are stored in ``adata.uns``.
    match_score : int, default=2
        Match score for alignment.
    mismatch_penalty : int, default=-3
        Mismatch penalty for alignment.
    gap_open_penalty : int, default=-5
        Gap open penalty.
    gap_extend_penalty : int, default=-1
        Gap extension penalty.

    Returns
    -------
    ad.AnnData
        The updated AnnData with alignment results in ``uns``.

    Examples
    --------
    >>> import vampire as vp
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.tl.motif_msa(adata)
    """
    import numpy as np
    import polars as pl
    import parasail

    if match_score <= 0:
        raise ValueError("match_score should be positive.")
    if mismatch_penalty >= 0:
        raise ValueError("mismatch_penalty should be negative.")
    if gap_open_penalty >= 0:
        raise ValueError("gap_open_penalty should be negative.")
    if gap_extend_penalty >= 0:
        raise ValueError("gap_extend_penalty should be negative.")
    if "motif" not in adata.var.columns:
        raise ValueError("adata.var['motif'] not found")

    motifs = adata.var["motif"]

    if reference is None:
        # MSA mode: use generic progressive MSA engine
        logger.info("Performing MSA of %d motifs...", len(motifs))

        # Map DNA bases to string indices so _nw's int() lookup works
        bases = "ACGT"
        base_to_stridx = {b: str(i) for i, b in enumerate(bases)}
        stridx_to_base = {str(i): b for i, b in enumerate(bases)}

        # Build DNA substitution matrix (4x4)
        sub_matrix = np.full((len(bases), len(bases)), mismatch_penalty, dtype=int)
        np.fill_diagonal(sub_matrix, match_score)

        # Convert sequences to string-index token lists
        sequences_str: dict[str, list[str]] = {}
        for motif_id, seq in motifs.items():
            tokens = [base_to_stridx[c] for c in str(seq).upper() if c in base_to_stridx]
            sequences_str[str(motif_id)] = tokens

        if len(sequences_str) == 0:
            adata.uns[store_key] = {
                "mode": "msa",
                "reference": "",
                "reference_id": "consensus",
                "alignment": {},
                "consensus": "",
                "n_motifs": 0,
                "variants": pl.DataFrame(),
            }
            return

        # Run MSA engine
        aligned_tokens, consensus_tokens = _msa_core(
            sequences_str,
            sub_matrix,
            gap_open_penalty,
            gap_extend_penalty,
            refine=True,
            max_refine_iter=3,
        )

        # Map back to DNA strings
        alignment = {
            k: "".join(stridx_to_base.get(t, t) for t in v)
            for k, v in aligned_tokens.items()
        }
        consensus = "".join(stridx_to_base.get(t, t) for t in consensus_tokens)

        # Build aligned consensus (gaps at all-gap columns) for unified format
        seqs = list(alignment.values())
        n_cols = len(seqs[0])
        all_gap_cols = {i for i in range(n_cols) if all(s[i] == "-" for s in seqs)}
        consensus_aln_list: list[str] = []
        ci = 0
        for i in range(n_cols):
            if i in all_gap_cols:
                consensus_aln_list.append("-")
            else:
                consensus_aln_list.append(consensus[ci])
                ci += 1
        alignment["reference"] = "".join(consensus_aln_list)

        # Compute variants directly from MSA alignment (reflects squeeze)
        variants_list = _msa_alignment_to_variants(alignment, consensus)
        variants_df = pl.DataFrame(variants_list)

        adata.uns[store_key] = {
            "mode": "msa",
            "reference": consensus,
            "reference_id": "consensus",
            "alignment": alignment,
            "consensus": consensus,
            "n_motifs": len(motifs),
            "variants": variants_df,
        }
        logger.info(
            "MSA completed. Aligned length: %d. Found %d variants.",
            len(consensus),
            len(variants_df),
        )

    else:
        # Pairwise mode: align each motif against the reference
        if isinstance(reference, int):
            ref_id = str(reference)
            if ref_id not in adata.var.index:
                raise KeyError(f"Motif id '{ref_id}' not found in adata.var.index")
            ref_seq = str(adata.var.loc[ref_id, "motif"])
        else:
            ref_id = None
            ref_seq = reference

        logger.info(
            "Reference specified; performing pairwise alignments of %d motifs against '%s'",
            len(motifs),
            ref_seq,
        )

        matrix = parasail.matrix_create("ACGT", match_score, mismatch_penalty)

        # Step 1: Collect all raw pairwise alignments
        raw_alignments: dict[str, tuple[str, str]] = {}
        for motif_id, seq in motifs.items():
            if str(seq) == ref_seq:
                raw_alignments[str(motif_id)] = (ref_seq, ref_seq)
            else:
                result = parasail.nw_trace_striped_16(
                    seq,
                    ref_seq,
                    -gap_open_penalty,
                    -gap_extend_penalty,
                    matrix,
                )
                raw_alignments[str(motif_id)] = (result.traceback.ref, result.traceback.query)

        # Step 2: Find homopolymer regions in reference
        ref_regions = _find_homopolymers(list(ref_seq))

        # Step 3: Compute global anchors from ALL alignments
        global_anchors: dict[tuple[int, int], int] = {}
        for start, end in ref_regions:
            mismatch_pos: list[int] = []
            all_anomaly_pos: list[int] = []
            for motif_id, (ref_aln, query_aln) in raw_alignments.items():
                if motif_id == "reference":
                    continue
                ref_idx = 0
                for i, char in enumerate(ref_aln):
                    if char != "-":
                        if start <= ref_idx < end:
                            if query_aln[i] != char:
                                rel_pos = ref_idx - start
                                all_anomaly_pos.append(rel_pos)
                                if query_aln[i] != "-":
                                    mismatch_pos.append(rel_pos)
                        ref_idx += 1
            if not all_anomaly_pos:
                continue

            # Density-priority anchor: choose the position with the most anomalies
            # (gaps + mismatches).  If tied, prefer positions that have mismatches.
            from collections import Counter
            pos_counts = Counter(all_anomaly_pos)
            max_count = max(pos_counts.values())
            candidates = [p for p, c in pos_counts.items() if c == max_count]
            if len(candidates) == 1:
                anchor = candidates[0]
            else:
                mismatch_set = set(mismatch_pos)
                mismatch_candidates = [p for p in candidates if p in mismatch_set]
                if mismatch_candidates:
                    anchor = mismatch_candidates[0]
                else:
                    anchor = candidates[0]
            global_anchors[(start, end)] = anchor

        # Step 4: Normalize variant positions in each query alignment
        normalized_alignments: dict[str, tuple[str, str]] = {}
        for motif_id, (ref_aln, query_aln) in raw_alignments.items():
            new_query = list(query_aln)
            for start, end in ref_regions:
                anchor = global_anchors.get((start, end))
                if anchor is None:
                    continue

                # Extract region from alignment (map ref_seq coords to aln cols)
                region_ref: list[str] = []
                region_query: list[str] = []
                ref_idx = 0
                for i, char in enumerate(ref_aln):
                    if char != "-":
                        if start <= ref_idx < end:
                            region_ref.append(char)
                            region_query.append(query_aln[i])
                        ref_idx += 1

                if not region_ref:
                    continue

                # Find anomalies in region
                anomalies: list[tuple[int, str]] = []
                for i, (q, r) in enumerate(zip(region_query, region_ref)):
                    if q != r:
                        anomalies.append((i, q))

                if not anomalies:
                    continue

                # Slide anomalies so first anomaly aligns with anchor
                first_pos = anomalies[0][0]
                offset = anchor - first_pos
                block_len = anomalies[-1][0] - anomalies[0][0] + 1
                if first_pos + offset < 0:
                    offset = -first_pos
                if first_pos + offset + block_len > len(region_ref):
                    offset = len(region_ref) - block_len - first_pos

                new_region = [region_ref[0]] * len(region_ref)
                for orig_pos, char in anomalies:
                    new_pos = orig_pos + offset
                    if 0 <= new_pos < len(region_ref):
                        new_region[new_pos] = char

                # Map back to alignment coordinates
                ref_idx = 0
                region_idx = 0
                for i, char in enumerate(ref_aln):
                    if char != "-":
                        if start <= ref_idx < end:
                            new_query[i] = new_region[region_idx]
                            region_idx += 1
                        ref_idx += 1

            normalized_alignments[motif_id] = (ref_aln, "".join(new_query))

        # Step 5: Build final alignment and variants from normalized alignments
        variants_list: list[dict] = []
        alignment: dict[str, str] = {"reference": ref_seq}
        for motif_id, seq in motifs.items():
            if str(seq) == ref_seq:
                alignment[str(motif_id)] = ref_seq
                continue
            ref_aln, query_aln = normalized_alignments[str(motif_id)]
            alignment[str(motif_id)] = query_aln
            sample_variants = _pairwise_alignment_to_variants(
                ref_aln,
                query_aln,
                ref_seq,
                str(motif_id),
            )
            variants_list.extend(sample_variants)

        variants_df = pl.DataFrame(variants_list)

        # Step 5b: Reposition same-base insertions to most divergent positions
        variants_df = _reposition_homopolymer_insertions_in_variants(
            variants_df, ref_seq
        )

        adata.uns[store_key] = {
            "mode": "pairwise",
            "reference": ref_seq,
            "reference_id": ref_id,
            "alignment": alignment,
            "consensus": None,
            "n_motifs": len(motifs),
            "variants": variants_df,
        }
        logger.info(
            "Aligned %d motifs against reference '%s'. Found %d variants.",
            len(motifs),
            ref_seq,
            len(variants_df),
        )

    return adata


def _pairwise_alignment_to_variants(
    ref_aln: str,
    seq_aln: str,
    ref: str,
    motif_id: str,
) -> list[dict]:
    """
    Convert a parasail traceback to variant records.

    Parameters
    ----------
    ref_aln
        Aligned reference sequence from parasail (may contain gaps ``'-'``).
    seq_aln
        Aligned query sequence from parasail (may contain gaps ``'-'``).
    ref
        Original reference motif sequence.
    motif_id
        Motif identifier (used as ``sample`` in output).

    Returns
    -------
    list[dict]
        Variant records with keys ``sample``, ``pos``, ``type``,
        ``ref``, ``alt``, ``seq``, ``length``.
    """
    variants: list[dict] = []
    ref_idx = 0

    i = 0
    while i < len(ref_aln):
        r = ref_aln[i]
        s = seq_aln[i]

        # Skip any position where both are gaps (should not happen,
        # but defensive).
        if r == "-" and s == "-":
            i += 1
            continue

        # Match
        if r != "-" and s != "-" and r == s:
            ref_idx += 1
            i += 1
            continue

        # Substitution
        if r != "-" and s != "-" and r != s:
            variants.append(
                {
                    "sample": motif_id,
                    "pos": ref_idx,
                    "type": "sub",
                    "ref": r,
                    "alt": s,
                    "seq": None,
                    "length": 1,
                }
            )
            ref_idx += 1
            i += 1
            continue

        # Insertion in query (gap in reference)
        if r == "-" and s != "-":
            ins_seq = ""
            while i < len(ref_aln) and ref_aln[i] == "-" and seq_aln[i] != "-":
                ins_seq += seq_aln[i]
                i += 1
            variants.append(
                {
                    "sample": motif_id,
                    "pos": ref_idx,
                    "type": "ins",
                    "ref": None,
                    "alt": None,
                    "seq": ins_seq,
                    "length": len(ins_seq),
                }
            )
            continue

        # Deletion in query (gap in query)
        if r != "-" and s == "-":
            del_len = 0
            del_start = ref_idx
            del_seq = ""
            while i < len(ref_aln) and ref_aln[i] != "-" and seq_aln[i] == "-":
                del_seq += ref_aln[i]
                del_len += 1
                ref_idx += 1
                i += 1
            variants.append(
                {
                    "sample": motif_id,
                    "pos": del_start,
                    "type": "del",
                    "ref": del_seq,
                    "alt": None,
                    "seq": None,
                    "length": del_len,
                }
            )
            continue

    return variants


def _msa_alignment_to_variants(
    alignment: dict[str, str],
    consensus: str,
) -> list[dict]:
    """Convert MSA alignment to variant records relative to consensus.

    Parameters
    ----------
    alignment
        Mapping from sample name to aligned DNA sequence (may contain gaps).
        Must include a ``"reference"`` key with the aligned consensus.
    consensus
        Consensus sequence without gaps.

    Returns
    -------
    list[dict]
        Variant records with keys ``sample``, ``pos``, ``type``,
        ``ref``, ``alt``, ``seq``, ``length``.
    """
    if not alignment:
        return []

    seqs = [v for k, v in alignment.items() if k != "reference"]
    if not seqs:
        return []

    n_cols = len(seqs[0])
    ref_aln = alignment.get("reference", "")

    variants: list[dict] = []
    for sample, seq_aln in alignment.items():
        if sample == "reference":
            continue

        ref_idx = 0
        i = 0
        while i < n_cols:
            r = ref_aln[i] if i < len(ref_aln) else "-"
            s = seq_aln[i]

            # Skip positions where both are gaps
            if r == "-" and s == "-":
                i += 1
                continue

            # Match
            if r != "-" and s != "-" and r == s:
                ref_idx += 1
                i += 1
                continue

            # Substitution
            if r != "-" and s != "-" and r != s:
                variants.append(
                    {
                        "sample": sample,
                        "pos": ref_idx,
                        "type": "sub",
                        "ref": r,
                        "alt": s,
                        "seq": None,
                        "length": 1,
                    }
                )
                ref_idx += 1
                i += 1
                continue

            # Insertion in query (gap in reference)
            if r == "-" and s != "-":
                ins_seq = ""
                while (
                    i < n_cols
                    and (ref_aln[i] if i < len(ref_aln) else "-") == "-"
                    and seq_aln[i] != "-"
                ):
                    ins_seq += seq_aln[i]
                    i += 1
                variants.append(
                    {
                        "sample": sample,
                        "pos": ref_idx,
                        "type": "ins",
                        "ref": None,
                        "alt": None,
                        "seq": ins_seq,
                        "length": len(ins_seq),
                    }
                )
                continue

            # Deletion in query (gap in query)
            if r != "-" and s == "-":
                del_len = 0
                del_start = ref_idx
                del_seq = ""
                while (
                    i < n_cols
                    and (ref_aln[i] if i < len(ref_aln) else "-") != "-"
                    and seq_aln[i] == "-"
                ):
                    del_seq += ref_aln[i]
                    del_len += 1
                    ref_idx += 1
                    i += 1
                variants.append(
                    {
                        "sample": sample,
                        "pos": del_start,
                        "type": "del",
                        "ref": del_seq,
                        "alt": None,
                        "seq": None,
                        "length": del_len,
                    }
                )
                continue

    return variants
