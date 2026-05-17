from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Tuple, Sequence, Literal
import numpy as np

if TYPE_CHECKING:
    import anndata as ad

import logging

logger = logging.getLogger(__name__)


def haplotype_neighbor(
    adata: ad.AnnData,
    aligned_key: str = "aligned",
    metrics: Sequence[str] = ("structural", "cnv", "composition"),
    k: int | None = None,
    fusion: Literal["max", "mean", "min"] = "max",
    gap_penalty: float | None = None,
    store_key: str = "haplotype",
) -> ad.AnnData:
    """Build a fused kNN graph from aligned motif arrays for haplotype analysis.

    Computes the requested pairwise distance matrices from aligned motif
    arrays, converts each into an adaptive kNN similarity graph, fuses the
    graphs, and stores the fused connectivities in ``adata.obsp``.

    Parameters
    ----------
    adata
        Annotated data with aligned motif arrays (from ``tl.align()``).
    aligned_key
        Key prefix for aligned arrays in ``adata.uns``.
    metrics
        Distance metrics to compute and fuse. Supported values are
        ``"structural"``, ``"cnv"`` and ``"composition"``.
    k
        Number of neighbours for the kNN graph. If ``None``, set to
        ``max(5, min(10, int(sqrt(n_samples))))``.
    fusion
        Graph fusion method: ``"max"`` (default), ``"mean"``, ``"min"`` or ``"max"``.
    gap_penalty
        Distance penalty for motif-vs-gap positions. If ``None``, derived
        as ``max(motif_distance) * 0.3``.
    store_key
        Key prefix for storing results.

    Returns
    -------
    adata
        Updated in-place with:

        - ``obsp[f"{store_key}_structural_distance"]`` — if requested
        - ``obsp[f"{store_key}_cnv_distance"]`` — if requested
        - ``obsp[f"{store_key}_composition_distance"]`` — if requested
        - ``obsp[f"{store_key}_connectivities"]`` — fused graph
        - ``uns[store_key]`` — parameters and graph statistics
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    aligned_motifs: Dict[str, List[str]] = adata.uns.get(f"{aligned_key}_motif_array")
    if aligned_motifs is None:
        raise KeyError(
            f"Aligned motif array not found at uns['{aligned_key}_motif_array']. "
            f"Run vp.anno.tl.align() first."
        )

    names = list(aligned_motifs.keys())
    n = len(names)

    if n == 0:
        logger.warning("No samples found in aligned motif array. Skipping graph build.")
        return adata

    if n == 1:
        logger.info("Only one sample available; building trivial 1-node graph.")
        adata.obsp[f"{store_key}_connectivities"] = csr_matrix(
            ([1.0], ([0], [0])), shape=(1, 1)
        )
        adata.uns[store_key] = {
            "params": {
                "aligned_key": aligned_key,
                "metrics": list(metrics),
                "k": 0,
                "fusion": fusion,
            },
            "stats": {"n_connected_components": 1, "n_samples": 1},
        }
        return adata

    # Normalise metric names
    metrics = [m.lower() for m in metrics]
    valid = {"structural", "cnv", "composition"}
    invalid = set(metrics) - valid
    if invalid:
        raise ValueError(
            f"Unknown metrics: {invalid}. Choose from {valid}."
        )

    if gap_penalty is None:
        max_dist = float(adata.varp["motif_distance"].max())
        gap_penalty = max_dist * 0.3 if max_dist > 0 else 1.0

    if k is None:
        k = max(5, min(10, int(np.sqrt(n))))
    k = min(k, n - 1)

    logger.info(
        "Computing distance matrices for metrics: %s", metrics
    )

    dist_mats: Dict[str, np.ndarray] = {}

    alignment_metrics = [m for m in metrics if m in ("structural", "cnv")]
    if alignment_metrics:
        dist_mats.update(
            _compute_alignment_distances(
                aligned_motifs,
                names,
                adata.varp["motif_distance"],
                gap_penalty,
                alignment_metrics,
            )
        )

    # Align alignment-based distances to adata.obs_names.
    # Composition distance is computed from adata.X and is already aligned.
    obs_names = list(adata.obs_names)
    if names != obs_names:
        logger.info(
            "Reordering alignment distance matrices to match adata.obs_names"
        )
        name_to_idx = {name: i for i, name in enumerate(names)}
        idx_map = [name_to_idx[name] for name in obs_names]
        for key in list(dist_mats.keys()):
            dist_mats[key] = dist_mats[key][np.ix_(idx_map, idx_map)]

    if "composition" in metrics:
        dist_mats["composition"] = _compute_composition_distance(adata)
        logger.info("Computed composition distance from adata.X")

    # Store individual distance matrices
    for metric, D in dist_mats.items():
        dist_key = f"{store_key}_{metric}_distance"
        adata.obsp[dist_key] = D
        logger.info("Stored %s distance matrix in obsp['%s']", metric, dist_key)

    # Build adaptive kNN graphs
    logger.info(
        "Building adaptive kNN graphs (k=%d, n=%d)", k, n
    )
    graphs = []
    for metric in metrics:
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
        A_fused = graphs[0]
        logger.info("Single metric requested; skipping graph fusion")
    else:
        logger.info("Fusing %d graphs using '%s'", len(graphs), fusion)
        A_fused = _fuse_graphs(graphs, method=fusion)
        logger.info(
            "Fused graph: %d undirected edges", A_fused.nnz // 2
        )

    # Check connectivity
    n_components, _ = connected_components(
        A_fused, directed=False, return_labels=True
    )
    if n_components > 1:
        logger.warning(
            "Fused graph is disconnected (%d components). "
            "Consider increasing k (currently %d) or using fusion='mean'.",
            n_components,
            k,
        )

    adata.obsp[f"{store_key}_connectivities"] = A_fused

    adata.uns[store_key] = {
        "params": {
            "aligned_key": aligned_key,
            "metrics": metrics,
            "k": k,
            "fusion": fusion,
            "gap_penalty": gap_penalty,
        },
        "stats": {
            "n_samples": n,
            "n_connected_components": int(n_components),
        },
    }

    logger.info(
        "Haplotype neighbour graph ready: %d nodes, %d edges, %d component(s)",
        n,
        A_fused.nnz // 2,
        n_components,
    )

    return adata


def haplotype_leiden(
    adata: ad.AnnData,
    resolution: float = 1.0,
    random_state: int = 0,
    store_key: str = "haplotype",
) -> ad.AnnData:
    """Cluster samples into haplotypes using Leiden on the fused graph.

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
    store_key
        Key prefix. Must match the value used in
        :func:`haplotype_neighbor`.

    Returns
    -------
    adata
        Updated in-place with:

        - ``obs[f"{store_key}"]`` — haplotype labels (category)
        - ``uns[f"{store_key}_consensus"]`` — consensus per haplotype
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
    n = A.shape[0]

    if n == 1:
        labels = np.array([0])
        logger.info("Single sample; assigned to haplotype H1")
    else:
        logger.info(
            "Running Leiden clustering (resolution=%.2f, random_state=%d)",
            resolution,
            random_state,
        )
        labels = _leiden_clustering(
            A, resolution=resolution, random_state=random_state
        )
        logger.info("Leiden found %d cluster(s)", len(np.unique(labels)))

    # Build consensus sequences
    neighbor_meta = adata.uns.get(store_key, {})
    aligned_key = neighbor_meta.get("params", {}).get("aligned_key", "aligned")
    aligned_motifs = adata.uns.get(f"{aligned_key}_motif_array")

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


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _compute_alignment_distances(
    aligned_motifs: Dict[str, List[str]],
    names: List[str],
    motif_distance: np.ndarray,
    gap_penalty: float,
    metrics: List[str],
) -> Dict[str, np.ndarray]:
    """Compute pairwise distance matrices for the requested metrics.

    Returns a dict mapping metric name to an ``(n, n)`` distance matrix.
    """
    import numpy as np

    max_dist = float(motif_distance.max())
    n = len(names)
    dist_mats = {m: np.zeros((n, n), dtype=float) for m in metrics}

    for i in range(n):
        for j in range(i + 1, n):
            seq_i = aligned_motifs[names[i]]
            seq_j = aligned_motifs[names[j]]

            structural_dist = 0.0
            overlap = 0
            gap_count = 0

            for a, b in zip(seq_i, seq_j):
                if a == "-" and b == "-":
                    continue
                if a == "-" or b == "-":
                    gap_count += 1
                else:
                    overlap += 1
                    if a != b:
                        structural_dist += motif_distance[int(a), int(b)]

            len_i = sum(1 for x in seq_i if x != "-")
            len_j = sum(1 for x in seq_j if x != "-")
            min_len = min(len_i, len_j)
            max_len = max(len_i, len_j)

            if "structural" in metrics:
                if overlap > 0:
                    avg_dist = structural_dist / overlap
                else:
                    avg_dist = max_dist

                if min_len > 0:
                    struct_val = (
                        avg_dist * overlap + max_dist * (min_len - overlap)
                    ) / min_len
                else:
                    struct_val = max_dist

                dist_mats["structural"][i, j] = dist_mats["structural"][j, i] = struct_val

            if "cnv" in metrics:
                cnv_val = (
                    gap_count * gap_penalty / max_len if max_len > 0 else 0.0
                )
                dist_mats["cnv"][i, j] = dist_mats["cnv"][j, i] = cnv_val

    return dist_mats


def _compute_composition_distance(
    adata: ad.AnnData,
) -> np.ndarray:
    """Compute pairwise Jensen-Shannon distance between motif composition vectors.

    Row-normalises ``adata.X`` to motif percentages (compositional vectors),
    then computes the Jensen-Shannon distance for all sample pairs.
    The result is an ``(n_obs, n_obs)`` symmetric distance matrix aligned to
    ``adata.obs_names``.

    Parameters
    ----------
    adata
        Annotated data with motif counts/abundance in ``X``.

    Returns
    -------
    np.ndarray
        Pairwise Jensen-Shannon distance matrix.
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
    graphs: List[np.ndarray],
    method: str = "mean",
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
    aligned_motifs: Dict[str, List[str]],
    labels: np.ndarray,
    names: List[str],
) -> Dict[str, List[str]]:
    """Build a consensus sequence for each haplotype cluster.

    At each alignment position the most frequent non-gap motif among cluster
    members is chosen. If all members have a gap at that position, ``"-"``
    is used.
    """
    from collections import Counter

    consensus: Dict[str, List[str]] = {}
    unique_labels = sorted(set(labels))

    for h in unique_labels:
        samples_in_h = [
            names[i] for i in range(len(names)) if labels[i] == h
        ]
        if not samples_in_h:
            continue

        seq_len = len(aligned_motifs[samples_in_h[0]])
        h_consensus: List[str] = []

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

        consensus[f"H{int(h) + 1}"] = h_consensus

    return consensus
