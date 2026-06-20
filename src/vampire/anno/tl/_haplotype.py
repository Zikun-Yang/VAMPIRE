from __future__ import annotations
from typing import TYPE_CHECKING, Sequence, Literal, Any
from collections.abc import Callable
import numpy as np
from ..pp._markdup import markdup

if TYPE_CHECKING:
    import anndata as ad

import logging

logger = logging.getLogger(__name__)

Metric = str | Callable[[Any], np.ndarray]


def _structural(
    adata: "ad.AnnData",
    aligned_key: str,
) -> np.ndarray:
    """
    Compute pairwise structural distance from aligned motif arrays.

    Gaps (``-``) are used as block delimiters, so copy-number changes that
    insert or delete whole tandem-repeat copies are handled naturally.  The
    distance blends **block-internal motif mismatches** with a **gap penalty**
    for informative positions that fall outside any block.

    Algorithm
    ---------
    1. Split the alignment into gap-delimited blocks.
    2. For each block compute the average normalised motif distance
       ``motif_distance / max(motif_distance)``.
    3. Compute *coverage* = ``total_block_len / total_informative``.
    4. Compute *continuity* = ``max_block_len / total_block_len``.
       One long block gives continuity ≈ 1 (highly reliable); many short
       scattered blocks give continuity ≪ 1 (less reliable).
    5. Adjust coverage downward when continuity is poor:
       ``coverage_adj = coverage * (0.5 + 0.5 * continuity)``.
    6. Auto-tune ``gap_penalty`` from the data as
       ``median(non-zero motif_distance) / max_dist``.
    7. Final distance:

       .. math::

          D = block_{dist} \times coverage_{adj}
              + gap_{penalty} \times (1 - coverage_{adj})

    8. If no overlapping motifs exist at all, ``D = 1.0``.

    The result is bounded in ``[0, 1]``.

    Parameters
    ----------
    adata
        Annotated data with ``uns[f"{aligned_key}_motif_array"]``.
    aligned_key
        Key prefix for aligned arrays in ``adata.uns``.

    Returns
    -------
    np.ndarray
        ``(n, n)`` symmetric distance matrix aligned to ``adata.obs_names``.
    """
    import numpy as np

    aligned: dict[str, list[str]] = adata.uns[f"{aligned_key}_motif_array"]
    names: list[str] = list(adata.obs_names)

    motif_distance = adata.varp["motif_distance"]
    max_dist = float(motif_distance.max())

    # Auto gap_penalty: median non-zero motif distance / max_dist
    # This means "a gap is penalised as much as an average motif mismatch".
    nonzero = motif_distance[motif_distance > 0]
    gap_penalty = (
        float(np.median(nonzero)) / max_dist if len(nonzero) > 0 else 0.5
    )

    n = len(names)
    D = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            seq_i = aligned[names[i]]
            seq_j = aligned[names[j]]

            blocks = []
            current = []
            n_both_gap = 0

            for a, b in zip(seq_i, seq_j):
                if a == "-" and b == "-":
                    n_both_gap += 1
                    continue
                if a == "-" or b == "-":
                    if current:
                        blocks.append(current)
                        current = []
                    continue
                current.append((a, b))

            if current:
                blocks.append(current)

            total_informative = len(seq_i) - n_both_gap

            if total_informative == 0:
                D[i, j] = D[j, i] = 0.0
                continue

            if not blocks:
                # No overlapping motifs at all — maximum structural distance
                D[i, j] = D[j, i] = 1.0
                continue

            total_block_cost = 0.0
            total_block_len = 0
            max_block_len = 0

            for block in blocks:
                block_len = len(block)
                if block_len == 0:
                    continue
                block_cost = 0.0
                for a, b in block:
                    if a != b:
                        block_cost += motif_distance[int(a), int(b)] / max_dist
                total_block_cost += block_cost
                total_block_len += block_len
                if block_len > max_block_len:
                    max_block_len = block_len

            block_dist = (
                total_block_cost / total_block_len if total_block_len > 0 else 0.0
            )
            coverage = total_block_len / total_informative

            # Continuity: how concentrated the matches are.
            # One long block  -> continuity ≈ 1  (highly reliable)
            # Many short blocks -> continuity low (scattered, less reliable)
            continuity = (
                max_block_len / total_block_len if total_block_len > 0 else 0.0
            )

            # Adjust coverage down when continuity is poor.
            coverage_adj = coverage * (0.5 + 0.5 * continuity)

            D[i, j] = D[j, i] = (
                block_dist * coverage_adj + gap_penalty * (1.0 - coverage_adj)
            )

    return D


def _cnv(
    adata: "ad.AnnData",
) -> np.ndarray:
    """
    Compute pairwise copy-number variation (CNV) distance.

    Uses the absolute difference between per-sample ``copy_number`` values
    stored in ``adata.obs``.  This is intentionally kept as a raw absolute
    difference rather than a normalised metric, because the CNV distance is
    usually fused with other metrics (structural, composition) downstream and
    normalisation is handled at the graph-fusion stage.

    Parameters
    ----------
    adata
        Annotated data with ``obs['copy_number']``.

    Returns
    -------
    np.ndarray
        ``(n, n)`` symmetric distance matrix aligned to ``adata.obs_names``.
    """
    import numpy as np

    copy_numbers: dict[str, float] = adata.obs['copy_number'].to_dict()
    names: list[str] = list(adata.obs_names)

    n = len(names)
    D = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            cn_i = copy_numbers[names[i]]
            cn_j = copy_numbers[names[j]]

            D[i, j] = D[j, i] = abs(cn_i - cn_j)

    return D


def _composition(
    adata: ad.AnnData,
) -> np.ndarray:
    """
    Compute pairwise Jensen-Shannon distance between motif composition vectors.

    Each sample is first row-normalised to a compositional (percentage) vector
    over motifs, then the Jensen-Shannon distance is computed for every pair.
    The metric is bounded in ``[0, 1]``, symmetric, and naturally handles
    zero-abundance motifs.

    Parameters
    ----------
    adata
        Annotated data with motif counts or abundances in ``X``.

    Returns
    -------
    np.ndarray
        ``(n_obs, n_obs)`` symmetric distance matrix aligned to
        ``adata.obs_names``.
    """
    from scipy.spatial.distance import pdist, squareform

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    # Row-normalise to percentages (compositional vectors)
    row_sums = X.sum(axis=1, keepdims=True)
    P = np.divide(X, row_sums, out=np.zeros_like(X), where=row_sums != 0)

    # Jensen-Shannon distance (bounded, symmetric, robust to zeros)
    D = squareform(pdist(P, metric="jensenshannon"))
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)

    return D


BUILTIN_METRICS: dict[str, Callable[..., np.ndarray]] = {
    "structural": _structural,
    "cnv": _cnv,
    "composition": _composition,
}


def _compress_adata(
    adata: ad.AnnData,
) -> ad.AnnData:
    """
    Compress adata with duplicates into a compressed adata, 
    where each unique group only have one representative sample

    Parameters
    ----------
    adata: ad.AnnData

    Returns
    -------
    ad.AnnData
    """

    name_to_group: dict[str, int] = adata.obs["unique_group"].to_dict()

    groups: list[int] = list(set(name_to_group.values()))
    n_unique: int = len(groups)
    expected = set(range(n_unique))
    actual = set(groups)
    assert actual == expected, (
        "unique_group encoding is invalid.\n"
        f"Expected groups: {expected}\n"
        f"Found groups: {actual}\n"
        f"n_unique (from compressed_adata): {n_unique}\n"
        "Hint: unique_group must be contiguous integers starting from 0. "
        "Run markdup() to encode groups."
    )

    group_to_name: dict[int, str] = {}
    for name, group in name_to_group.items():
        group_to_name.setdefault(group, name)

    # sort by unique_group id (0...n-1)
    unique_names = [
        group_to_name[g]
        for g in sorted(group_to_name)
    ]

    # subset 
    compressed_adata = adata[unique_names, :].copy()

    return compressed_adata


def _expand_matrix(
    adata: ad.AnnData,
    compressed_matrix: np.ndarray,
) -> np.ndarray:
    """
    Expand an n_unique × n_unique matrix computed on compressed_adata
    back to an n_obs × n_obs matrix.

    Parameters
    ----------
    adata
        Original adata containing ``obs['unique_group']``.

    compressed_matrix
        Matrix computed on compressed_adata.
        Shape = (n_unique, n_unique)

    Returns
    -------
    np.ndarray
        Expanded matrix with shape (n_obs, n_obs).
    """
    groups: list[int] = list(adata.obs["unique_group"].to_numpy())
    n_unique: int = len(set(groups))
    expected = set(range(n_unique))
    actual = set(groups)
    assert actual == expected, (
        "unique_group encoding is invalid.\n"
        f"Expected groups: {expected}\n"
        f"Found groups: {actual}\n"
        f"n_unique (from compressed_adata): {n_unique}\n"
        "Hint: unique_group must be contiguous integers starting from 0. "
        "Run markdup() to encode groups."
    )

    groups = adata.obs["unique_group"].to_numpy()

    n_obs = adata.n_obs
    n_unique = compressed_matrix.shape[0]

    if groups.max() + 1 != n_unique:
        raise ValueError(
            f"Incompatible dimensions: "
            f"found {groups.max()+1} groups but "
            f"compressed matrix has shape {compressed_matrix.shape}."
        )

    expanded = compressed_matrix[
        groups[:, None],
        groups[None, :]
    ]

    return expanded


def haplotype_neighbor(
    adata: ad.AnnData,
    *,
    aligned_key: str = "aligned",
    metrics: list[Metric] | None = None,
    k: int | None = None,
    fusion_method: Literal["max", "mean", "min"] = "max",
    store_key: str = "haplotype",
) -> ad.AnnData:
    """
    Build a fused kNN graph from aligned motif arrays for haplotype analysis.

    Computes the requested pairwise distance matrices from aligned motif
    arrays, converts each into an adaptive kNN similarity graph, fuses the
    graphs, and stores the fused connectivities in ``adata.obsp``.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with aligned motif arrays (from ``tl.sample_msa()``).
    aligned_key : str
        Key prefix for aligned arrays in ``adata.uns``.
    metrics : list of str or callables
        Distance metrics to compute and fuse. Supported values are
        ``"structural"``, ``"cnv"`` and ``"composition"`` and callable functions.
        Self-defined callables must take ``adata`` as input and return an ``(n, n)``
        distance matrix aligned to ``adata.obs_names``. If ``None``, defaults to
        ``["structural", "cnv", "composition"]``.
    k : int | None
        Number of neighbours for the kNN graph. If ``None``, set to
        ``max(5, min(15, int(sqrt(n_samples))))``.
    fusion_method : Literal["max", "mean", "min"]
        Graph fusion method: ``"max"`` (default), ``"mean"``, ``"min"``. 
        With ``"max"``, an edge is retained if it appears in any of the individual
        graphs, and its similarity weight is set to the maximum weight observed
        across those graphs. This strategy encourages cells to be grouped together
        whenever they are highly similar under at least one metric. 
        Instead, with ``"min"``, the similarity weight is set to the minimum weight 
        observed across the individual graphs. This strategy emphasizes consensus 
        among metrics and encourages cells to be grouped together only when they 
        are consistently similar in every aspect (e.g. strutural, composition and cnv).
    store_key
        Key prefix for storing results.

    Returns
    -------
    adata
        Updated in-place with:

        - ``obsp[f"{store_key}_structural_distance"]`` — if requested
        - ``obsp[f"{store_key}_cnv_distance"]`` — if requested
        - ``obsp[f"{store_key}_composition_distance"]`` — if requested
        - ``obsp[f"{store_key}_{func_name}_distance"]`` — for any custom metric provided as a callable
        - ``obsp[f"{store_key}_connectivities"]`` — fused graph
        - ``uns[store_key]`` — parameters and graph statistics

    Examples
    --------
    >>> import vampire as vp
    >>> vp.anno.pl.set_default_plotstyle()
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.tl.sample_msa(adata)
    >>> vp.anno.tl.haplotype_neighbor(adata, metrics=["structural", "composition"])
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    aligned_motifs: dict[str, list[str]] = adata.uns.get(f"{aligned_key}_motif_array")
    if aligned_motifs is None:
        raise KeyError(
            f"Aligned motif array not found at uns['{aligned_key}_motif_array']. "
            f"Run vp.anno.tl.sample_msa() first."
        )

    names = list(aligned_motifs.keys())
    n_obs = len(names)

    if n_obs == 0:
        logger.warning("No samples found in aligned motif array. Skipping graph build.")
        return adata

    if metrics is None:
        metrics = ["structural", "cnv", "composition"]

    # deduplication
    if "unique_group" not in adata.obs.columns:
        logger.warning(
            "unique_group not found in adata.obs. "
            "vp.anno.pp.markdup() has not been run. Running it automatically."
        )
        adata = markdup(adata)

    compressed_adata = _compress_adata(adata)
    n_unique = compressed_adata.n_obs
    if n_unique < n_obs:
        logger.info(
            "%d samples collapsed to %d unique sequences. (%d duplicates removed)",
            n_obs,
            n_unique,
            n_obs - n_unique,
        )

    if n_unique == 1:
        logger.info("Only one unique sample available; building trivial 1-node graph.")
        # Resolve metric names for storage
        metric_names: list[str] = []
        for m in metrics:
            if callable(m):
                metric_names.append(getattr(m, "__name__", "custom"))
            else:
                metric_names.append(str(m).lower())
        adata.obsp[f"{store_key}_connectivities"] = csr_matrix(np.ones((n_obs, n_obs)))
        adata.uns[store_key] = {
            "params": {
                "aligned_key": aligned_key,
                "metrics": metric_names,
                "k": 0,
                "fusion_method": fusion_method,
            },
            "stats": {
                "n_connected_components": 1,
                "n_samples": n_obs,
                "n_unique": n_unique,
            },
        }
        return adata

    if k is None:
        k = max(5, min(15, int(np.sqrt(n_unique))))
    k = min(k, n_unique - 1)

    # Parse metrics: resolve names and compute distance matrices
    dist_mats: dict[str, np.ndarray] = {}
    metric_names: list[str] = []

    # Context passed to custom metric callables so they can access
    # haplotype_neighbor parameters by name.
    _metric_context: dict[str, Any] = {
        "aligned_key": aligned_key,
        "k": k,
        "fusion_method": fusion_method,
        "store_key": store_key,
    }

    for m in metrics:
        if callable(m):
            name = getattr(m, "__name__", "custom")
            # Ensure unique name
            base_name = name
            suffix = 1
            while name in metric_names:
                name = f"{base_name}_{suffix}"
                suffix += 1
            metric_names.append(name)

            import inspect

            sig = inspect.signature(m)
            param_names = list(sig.parameters.keys())
            if len(param_names) == 1:
                compressed_D = m(compressed_adata)
            else:
                kwargs = {p: _metric_context[p] for p in param_names[1:] if p in _metric_context}
                compressed_D = m(compressed_adata, **kwargs)
            if compressed_D.shape[0] != n_unique:
                raise ValueError(
                    f"Custom metric '{name}' returned distance matrix with shape "
                    f"{compressed_D.shape}, expected ({n_unique}, {n_unique}). "
                    f"Custom metrics should compute distances on the provided adata."
                )
            dist_mats[name] = compressed_D
            logger.info("Computed custom metric '%s' from callable", name)
        elif isinstance(m, str):
            m_lower = m.lower()
            if m_lower not in BUILTIN_METRICS:
                valid_names = ", ".join(BUILTIN_METRICS.keys())
                raise ValueError(
                    f"Unknown metric '{m}'. Choose from {valid_names} or provide a callable."
                )
            metric_names.append(m_lower)
            match m_lower:
                case "structural":
                    dist_mats[m_lower] = _structural(compressed_adata, aligned_key)
                case "cnv":
                    dist_mats[m_lower] = _cnv(compressed_adata)
                case "composition":
                    dist_mats[m_lower] = _composition(compressed_adata)
            logger.info("Computed built-in metric '%s'", m_lower)
        else:
            raise TypeError(f"Metric must be str or callable, got {type(m)}")

    # Store individual distance matrices (expand m x m to n_obs x n_obs)
    obs_names = list(adata.obs_names)
    for metric, D_compressed in dist_mats.items():
        dist_key = f"{store_key}_{metric}_distance"
        if D_compressed.shape[0] != n_obs:
            D = _expand_matrix(adata, D_compressed)
        else:
            D = D_compressed
        adata.obsp[dist_key] = D
        logger.info("Stored %s distance matrix in obsp['%s']", metric, dist_key)

    # Build adaptive kNN graphs on m x m
    logger.info(
        "Building adaptive kNN graphs (k=%d, n=%d)", k, n_unique
    )
    graphs = []
    for metric in metric_names:
        A = _distance_to_knn_graph(dist_mats[metric], k)
        A_sym = _symmetrize_graph(A, method="union")
        graphs.append(A_sym)
        logger.info(
            "Built kNN graph for '%s': %d undirected edges",
            metric,
            A_sym.nnz // 2,
        )

    # Fuse graphs
    if len(graphs) == 1:
        compressed_A_fused = graphs[0]
        logger.info("Single metric requested; skipping graph fusion")
    else:
        logger.info("Fusing %d graphs using '%s'", len(graphs), fusion_method)
        compressed_A_fused = _fuse_graphs(graphs, method=fusion_method)
        logger.info(
            "Fused graph: %d undirected edges", compressed_A_fused.nnz // 2
        )

    # Check connectivity
    n_components, _ = connected_components(
        compressed_A_fused, directed=False, return_labels=True
    )
    if n_components > 1:
        logger.warning(
            "Fused graph is disconnected (%d components). "
            "Consider increasing k (currently %d).",
            n_components,
            k,
        )

    A_fused = _expand_matrix(adata, compressed_A_fused)
    adata.obsp[f"{store_key}_connectivities"] = A_fused

    adata.uns[store_key] = {
        "params": {
            "aligned_key": aligned_key,
            "metrics": metric_names,
            "k": k,
            "fusion_method": fusion_method,
        },
        "stats": {
            "n_samples": n_obs,
            "n_unique": n_unique,
            "n_connected_components": int(n_components),
        },
    }

    logger.info(
        "Haplotype neighbour graph ready: %d unique nodes, %d edges, %d component(s)",
        n_unique,
        compressed_A_fused.nnz // 2,
        n_components,
    )

    return adata


def haplotype_leiden(
    adata: ad.AnnData,
    resolution: float = 1.0,
    *,
    random_state: int = 0,
    sort_by: str = "sample_size",
    store_key: str = "haplotype",
) -> ad.AnnData:
    """
    Cluster samples into haplotypes using Leiden on the fused graph.

    Reads the fused connectivities built by :func:`haplotype_neighbor`,
    runs the Leiden algorithm, builds a consensus sequence for each
    detected cluster, and stores haplotype labels in ``adata.obs``.

    Parameters
    ----------
    adata
        Annotated data with fused connectivities.
    resolution
        Leiden resolution parameter. Higher values yield more clusters.
    random_state
        Random seed for Leiden.
    sort_by
        How to order the resulting haplotype labels.
        ``"sample_size"`` (default) sorts by the number of samples in
        each cluster (largest → H1).  ``"copy_number"`` sorts by the
        average non-gap motif count per cluster (largest → H1).
    store_key
        Key prefix. Must match the value used in
        :func:`haplotype_neighbor`.

    Returns
    -------
    adata
        Updated in-place with:

        - ``obs[f"{store_key}"]`` — haplotype labels (category)
        - ``uns[f"{store_key}_consensus"]`` — consensus per haplotype

     Examples
    --------
    >>> import vampire as vp
    >>> vp.anno.pl.set_default_plotstyle()
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.tl.sample_msa(adata)
    >>> vp.anno.tl.haplotype_neighbor(adata, metrics=["structural", "composition"])
    >>> vp.anno.tl.haplotype_leiden(
    >>>     adata,
    >>>     resolution = 1.0
    >>> )
    """
    import numpy as np
    import pandas as pd

    connectivities_key = f"{store_key}_connectivities"
    if connectivities_key not in adata.obsp:
        raise KeyError(
            f"Fused graph not found at obsp['{connectivities_key}']. "
            f"Run vp.anno.tl.haplotype_neighbor() first."
        )

    A = adata.obsp[connectivities_key]
    n_obs: int = adata.n_obs
    n_unique: int = len(set(adata.obs["unique_group"].to_list()))
    compressed_adata = _compress_adata(adata)
    compressed_A = compressed_adata.obsp[connectivities_key]

    if n_unique == 1:
        labels_unique = np.array([0])
        logger.info("Single unique sample; assigned to haplotype H1")
    else:
        logger.info(
            "Running Leiden clustering (resolution=%.2f, random_state=%d)",
            resolution,
            random_state,
        )
        labels_unique = _leiden_clustering(
            compressed_A, resolution=resolution, random_state=random_state
        )
        logger.info("Leiden found %d cluster(s)", len(np.unique(labels_unique)))

    # Expand labels from n_unique unique groups to n_obs original samples
    if n_unique != n_obs:
        if "unique_group" not in adata.obs.columns:
            raise KeyError(
                f"Connectivities shape {n_unique} does not match n_obs {n_obs} "
                f"and 'unique_group' column is missing. "
                f"Run vp.anno.tl.haplotype_neighbor() first."
            )
        unique_groups = adata.obs["unique_group"].to_numpy(dtype=int)
        labels = labels_unique[unique_groups]
        n = n_obs
    else:
        labels = labels_unique
        n = n_unique

    # Build consensus sequences
    neighbor_meta = adata.uns.get(store_key, {})
    aligned_key = neighbor_meta.get("params", {}).get("aligned_key", "aligned")
    aligned_motifs = adata.uns.get(f"{aligned_key}_motif_array")

    # ---- Sort haplotype labels ----
    if sort_by is not None and len(np.unique(labels)) > 1:
        unique_labels = sorted(set(labels))

        match sort_by:
            case "sample_size":
                metric = {ul: sum(labels == ul) for ul in unique_labels}
            case "copy_number":
                if aligned_motifs is not None:
                    names_tmp = list(aligned_motifs.keys())
                    obs_names_tmp = list(adata.obs_names)
                    if names_tmp != obs_names_tmp:
                        name_to_idx_obs = {name: i for i, name in enumerate(obs_names_tmp)}
                        idx_map = [name_to_idx_obs[name] for name in names_tmp]
                        labels_tmp = labels[idx_map]
                    else:
                        labels_tmp = labels

                    metric = {}
                    for ul in unique_labels:
                        members = [
                            names_tmp[i]
                            for i in range(len(names_tmp))
                            if labels_tmp[i] == ul
                        ]
                        avg_len = sum(
                            len([m for m in aligned_motifs[name] if m != "-"])
                            for name in members
                        ) / max(len(members), 1)
                        metric[ul] = avg_len
                else:
                    logger.warning(
                        "Aligned motifs not found; falling back to sample_size for sorting."
                    )
                    metric = {ul: sum(labels == ul) for ul in unique_labels}
            case _:
                raise ValueError(
                    f"sort_by must be 'sample_size' or 'copy_number', got {sort_by}"
                )

        # Sort by metric descending, then by original label for stability
        sorted_labels = sorted(unique_labels, key=lambda ul: (-metric[ul], ul))
        remap = {old: new for new, old in enumerate(sorted_labels)}
        labels = np.array([remap[l] for l in labels])
        logger.info("Sorted haplotypes by %s", sort_by)

    if aligned_motifs is not None:
        names = list(aligned_motifs.keys())
        obs_names = list(adata.obs_names)
        if names != obs_names:
            name_to_idx_obs = {name: i for i, name in enumerate(obs_names)}
            idx_map = [name_to_idx_obs[name] for name in names]
            labels_for_consensus = labels[idx_map]
        else:
            labels_for_consensus = labels

        consensus = _build_haplotype_consensus(
            aligned_motifs, labels_for_consensus, names
        )
        adata.uns[f"{store_key}_consensus"] = consensus
        logger.info("Built consensus for %d haplotype(s)", len(consensus))
    else:
        logger.warning(
            "Aligned motifs not found at uns['%s_motif_array']; "
            "skipping consensus build",
            aligned_key,
        )
        adata.uns[f"{store_key}_consensus"] = {}

    # Store labels (1-indexed for display)
    adata.obs[store_key] = pd.Categorical(
        [f"H{int(l) + 1}" for l in labels]
    )

    # Update metadata
    if store_key not in adata.uns:
        adata.uns[store_key] = {}
    adata.uns[store_key].setdefault("params", {})
    adata.uns[store_key].setdefault("stats", {})
    adata.uns[store_key]["params"]["resolution"] = resolution
    adata.uns[store_key]["params"]["random_state"] = random_state
    adata.uns[store_key]["params"]["sort_by"] = sort_by
    adata.uns[store_key]["stats"]["n_clusters"] = len(np.unique(labels))

    cluster_sizes = dict(adata.obs[store_key].value_counts().sort_index())
    cluster_sizes = {k: int(v) for k, v in cluster_sizes.items()}
    logger.info(
        "Haplotype assignment complete: %d sample(s) → %d cluster(s). "
        "Sizes: %s",
        n,
        len(np.unique(labels)),
        dict(cluster_sizes),
    )

    return adata


def haplotype_leiden_res_scan(
    adata: ad.AnnData,
    resolution_min: float = 0.1,
    resolution_max: float = 2.0,
    resolution_step: float = 0.1,
    *,
    random_state: int = 0,
    store_key: str = "haplotype",
) -> float | None:
    """
    Scan Leiden resolution and record cluster counts + modularity.

    Runs Leiden clustering across a resolution grid, computing modularity
    for each resolution.  Results are stored in
    ``uns[f"{store_key}_evaluation"]`` for downstream plotting.

    Parameters
    ----------
    adata
        Annotated data with fused connectivities from
        :func:`haplotype_neighbor`.
    resolution_min
        Lower bound of the resolution scan.
    resolution_max
        Upper bound of the resolution scan.
    resolution_step
        Step size between consecutive resolution values.
    random_state
        Random seed passed to Leiden.
    store_key
        Key prefix. Must match the value used in
        :func:`haplotype_neighbor`.

    Returns
    -------
    best_resolution : float | None
        The resolution with highest modularity score. If no resolution is found, return None.

    Examples
    --------
    >>> import vampire as vp
    >>> vp.anno.pl.set_default_plotstyle()
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.tl.sample_msa(adata)
    >>> vp.anno.tl.haplotype_neighbor(adata, metrics=["structural", "composition"])
    >>> vp.anno.tl.haplotype_leiden_res_scan(
    >>>     adata,
    >>>     resolution_min = 0.1,
    >>>     resolution_max = 2.0,
    >>>     resolution_step: float = 0.1,
    >>> )
    """
    import numpy as np

    # deduplication
    if "unique_group" not in adata.obs.columns:
        logger.warning(
            "unique_group not found in adata.obs. "
            "vp.anno.pp.markdup() has not been run. Running it automatically."
        )
        adata = markdup(adata)
    compressed_adata = _compress_adata(adata)

    connectivities_key = f"{store_key}_connectivities"
    if connectivities_key not in compressed_adata.obsp:
        raise KeyError(
            f"Fused graph not found at obsp['{connectivities_key}']. "
            f"Run vp.anno.tl.haplotype_neighbor() first."
        )

    A = compressed_adata.obsp[connectivities_key]
    n = A.shape[0]

    if n <= 1:
        logger.warning("Too few samples (%d) for resolution scan. Skipping.", n)
        adata.uns[f"{store_key}_evaluation"] = {
            "resolution_range": [],
            "n_clusters": [],
            "metric": {},
            "params": {
                "resolution_min": resolution_min,
                "resolution_max": resolution_max,
                "resolution_step": resolution_step,
            },
        }
        return None

    try:
        import leidenalg
        import igraph as ig
    except ImportError:
        raise ImportError(
            "haplotype_leiden_scan requires 'leidenalg' and 'igraph'. "
            "Install them via: pip install leidenalg igraph"
        )

    g = ig.Graph.Weighted_Adjacency(
        A.toarray().tolist(), mode="undirected", attr="weight"
    )

    resolutions = np.arange(
        resolution_min,
        resolution_max + resolution_step / 2,
        resolution_step,
    )
    resolutions = np.round(resolutions, 6)

    mod_scores: list[float] = []
    k_values: list[int] = []

    for res in resolutions:
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(res),
            seed=random_state,
        )
        labels = np.array(partition.membership)
        k = int(len(np.unique(labels)))
        k_values.append(k)

        # standard modularity (resolution-independent)
        mod = g.modularity(labels.tolist(), weights="weight")
        mod_scores.append(float(mod))

    best_mod_idx = int(np.argmax(mod_scores))
    best_mod_res = float(resolutions[best_mod_idx])
    best_mod_score = float(mod_scores[best_mod_idx])

    adata.uns[f"{store_key}_evaluation"] = {
        "resolution_range": resolutions.tolist(),
        "n_clusters": k_values,
        "metric": {
            "name": "modularity",
            "scores": mod_scores,
            "best_resolution": best_mod_res,
            "best_score": best_mod_score,
        },
        "params": {
            "resolution_min": resolution_min,
            "resolution_max": resolution_max,
            "resolution_step": resolution_step,
        },
    }

    logger.info(
        "Resolution scan complete: %d resolutions → best modularity %.3f at res=%.2f (k=%d)",
        len(resolutions),
        best_mod_score,
        best_mod_res,
        k_values[best_mod_idx],
    )

    return best_mod_res


def _distance_to_knn_graph(
    D: np.ndarray,
    k: int,
) -> np.ndarray:
    """Convert a distance matrix to an adaptive kNN similarity graph.

    For each node, the local scale ``sigma`` is set to the distance of its
    *k*-th nearest neighbour. Similarity is ``exp(-D / sigma)``.

    Returns a directed ``csr_matrix``.
    """
    import numpy as np
    from scipy.sparse import csr_matrix

    n = D.shape[0]
    knn_indices = np.argsort(D, axis=1)[:, 1 : k + 1]
    knn_dists = np.take_along_axis(D, knn_indices, axis=1)

    sigma = knn_dists[:, -1] + 1e-10

    rows = np.repeat(np.arange(n), k)
    cols = knn_indices.ravel()
    vals = np.exp(-knn_dists.ravel() / sigma[rows])

    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def _symmetrize_graph(
    A: np.ndarray,
    method: str = "union",
) -> np.ndarray:
    """Symmetrise a directed kNN graph.

    ``"union"`` (default) keeps an edge if *either* direction has it.
    ``"intersection"`` keeps an edge only if *both* directions have it.
    """
    if method == "union":
        return A.maximum(A.T)
    elif method == "intersection":
        return A.minimum(A.T)
    else:
        raise ValueError(
            f"method must be 'union' or 'intersection', got '{method}'"
        )


def _fuse_graphs(
    graphs: list[np.ndarray],
    method: str = "max",
) -> np.ndarray:
    """Fuse multiple similarity graphs into one.

    All graphs must have the same shape. For small TR datasets the graphs
    are converted to dense arrays, fused element-wise, and returned as a
    sparse matrix.
    """
    import numpy as np
    from scipy.sparse import csr_matrix

    if len(graphs) == 1:
        return graphs[0]

    mats = [g.toarray() for g in graphs]
    stacked = np.stack(mats, axis=0)

    if method == "mean":
        fused = np.mean(stacked, axis=0)
    elif method == "min":
        fused = np.min(stacked, axis=0)
    elif method == "max":
        fused = np.max(stacked, axis=0)
    else:
        raise ValueError(f"Unknown fusion method: '{method}'")

    return csr_matrix(fused)


def _leiden_clustering(
    adjacency: np.ndarray,
    resolution: float = 1.0,
    random_state: int = 0,
) -> np.ndarray:
    """Run Leiden community detection on a weighted adjacency matrix.

    Returns 0-indexed cluster labels.
    """
    try:
        import leidenalg
        import igraph as ig
    except ImportError:
        raise ImportError(
            "haplotype_leiden requires 'leidenalg' and 'igraph'. "
            "Install them via: pip install leidenalg igraph"
        )

    A = adjacency.toarray()
    g = ig.Graph.Weighted_Adjacency(
        A.tolist(), mode="undirected", attr="weight"
    )

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=random_state,
    )
    return np.array(partition.membership)


def _build_haplotype_consensus(
    aligned_motifs: dict[str, list[str]],
    labels: np.ndarray,
    names: list[str],
) -> dict[str, list[str]]:
    """Build a consensus sequence for each haplotype cluster.

    At each alignment position the most frequent non-gap motif among cluster
    members is chosen. If all members have a gap at that position, ``"-"``
    is used.
    """
    from collections import Counter

    consensus: dict[str, list[str]] = {}
    unique_labels = sorted(set(labels))

    for h in unique_labels:
        samples_in_h = [
            names[i] for i in range(len(names)) if labels[i] == h
        ]
        if not samples_in_h:
            continue

        seq_len = len(aligned_motifs[samples_in_h[0]])
        h_consensus: list[str] = []

        for pos in range(seq_len):
            motifs = [aligned_motifs[s][pos] for s in samples_in_h]
            h_consensus.append(Counter(motifs).most_common(1)[0][0])

        consensus[f"H{int(h) + 1}"] = h_consensus

    return consensus
