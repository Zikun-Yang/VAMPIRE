from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Optional

if TYPE_CHECKING:
    import numpy as np
    import anndata as ad

import logging

logger = logging.getLogger(__name__)


def motif_abundance_pca(
    adata: ad.AnnData,
    layer: Optional[str] = None,
    clr_transform: bool = False,
    n_components: int = 10,
) -> ad.AnnData:
    """PCA on motif abundance percentage vectors.

    Row-normalises the motif abundance matrix to percentages, optionally
    applies a centered log-ratio (CLR) transform, then performs PCA.
    Results are stored in ``adata.obs``, ``adata.var``, and ``adata.uns``.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with motif abundance in ``X`` or ``layers``.
    layer : str, optional
        Layer key to use instead of ``adata.X``.
    clr_transform : bool, default=False
        If ``True``, apply a centered log-ratio transform before PCA.
    n_components : int, default=10
        Number of principal components to compute.

    Returns
    -------
    ad.AnnData
        The updated AnnData with PCA results.

    Notes
    -----
    Stores the following fields (following scanpy conventions):

    - ``obsm["X_motif_abundance_pca"]`` — PC coordinates (ndarray, n_obs × n_components)
    - ``varm["motif_abundance_PCs"]`` — motif loadings (ndarray, n_vars × n_components)
    - ``uns["motif_abundance_pca"]`` — PCA metadata (variance, variance_ratio)

    Examples
    --------
    >>> import vampire as vp
    >>> adata = vp.anno.tl.motif_abundance_pca(adata)
    """
    import numpy as np
    from sklearn.decomposition import PCA

    X = adata.X if layer is None else adata.layers[layer]
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    # Row-normalise to percentages
    row_sums = X.sum(axis=1, keepdims=True)
    X_pct = np.divide(X, row_sums, out=np.zeros_like(X), where=row_sums != 0)

    # Optional CLR transform
    if clr_transform:
        X_pct = np.clip(X_pct, a_min=1e-10, a_max=None)
        log_x = np.log(X_pct)
        X_pct = log_x - log_x.mean(axis=1, keepdims=True)

    if n_components > X_pct.shape[1]:
        logger.warning(
            f"Requested n_components={n_components} exceeds number of features ({X_pct.shape[1]}). "
            f"Reducing n_components to {X_pct.shape[1]}."
        )
        n_components = X_pct.shape[1]

    # PCA
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X_pct)

    # Store results (scanpy convention)
    adata.obsm["X_motif_abundance_pca"] = pcs
    adata.varm["motif_abundance_PCs"] = pca.components_.T
    adata.uns["motif_abundance_pca"] = {
        "variance": pca.explained_variance_.tolist(),
        "variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_components": n_components,
        "clr_transform": clr_transform,
    }

    logger.info(
        f"PCA on motif abundance: {n_components} components. "
        f"Explained variance: {', '.join(f'{v * 100:.1f}%' for v in pca.explained_variance_ratio_)}"
    )

    return adata
