from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any, Literal, Sequence
import numpy as np
import polars as pl

if TYPE_CHECKING:
    import anndata as ad
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go
    import plotly.subplots as sp

import logging
logger = logging.getLogger(__name__)

from . import _sizing
from ._setting import _save_figure
from ._setting import _COLORMAP_OPTIONS # dict[str, list[str] | dict[str, str]]


def motif_abundance_pca(
    adata: "ad.AnnData",
    color_by: str | None = None,
    shape_by: str | None = None,
    components: tuple[int, int] = (1, 2),
    figsize: tuple[int | None, int | None] = (None, None),
    title: str | None = None,
    marker_size: int = 10,
    colormap: str | list[str] | None = None,
    show_variance: bool = True,
    save: str | bool | None = None,
    **kwargs,
) -> "go.Figure":
    """Plot pairwise principal components from motif abundance PCA.

    Reads pre-computed PCA results stored by ``vp.anno.tl.motif_abundance_pca()``.
    Color and marker shape can be mapped to columns in ``adata.obs``.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with PCA results from ``vp.anno.tl.motif_abundance_pca()``.
    color_by : str, optional
        Column in ``adata.obs`` for marker color.  Categorical columns use
        a discrete palette; numeric columns use a continuous colorscale.
    shape_by : str, optional
        Column in ``adata.obs`` for marker shape.  Must be categorical.
    components : tuple[int, int], default=(1, 2)
        Which two PCs to plot.  1-based indexing, e.g. ``(1, 2)`` for PC1
        vs PC2, ``(2, 3)`` for PC2 vs PC3.
    figsize : tuple[int | None, int | None], default=(None, None)
        Figure size in pixels.
    title : str | None, default=None
        Plot title.
    marker_size : int, default=10
        Marker size.
    colormap : str | list[str] | None, default=None
        Plotly colormap name for numeric ``color_by``.  Defaults to
        ``"Viridis"``.
    show_variance : bool, default=True
        Append explained-variance percentages to axis titles.
    **kwargs
        Additional keyword arguments passed to ``fig.update_layout()``.
    save : str | bool | None, default=None
        If ``True`` or a ``str``, save the figure. A string is appended to the default filename.
        Infer the filetype if ending on {``'.pdf'``, ``'.png'``, ``'.svg'``}.

    Returns
    -------
    go.Figure
        Plotly scatter figure of the chosen PCs.

    Examples
    --------
    >>> import vampire as vp
    >>> vp.anno.pl.set_default_plotstyle()
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.tl.motif_abundance_pca(adata)
    >>> vp.anno.pl.motif_abundance_pca(adata, color_by="copy_number", components=(1, 2))
    """
    import pandas as pd
    import plotly.graph_objects as go

    # ---- read pre-computed PCA results ----
    pc_mat = adata.obsm.get("X_motif_abundance_pca")
    if pc_mat is None:
        raise KeyError(
            "PCA results not found at obsm['X_motif_abundance_pca']. "
            "Run vp.anno.tl.motif_abundance_pca() first."
        )

    n_pcs = pc_mat.shape[1]
    x_idx = components[0] - 1
    y_idx = components[1] - 1
    if x_idx < 0 or y_idx < 0 or x_idx >= n_pcs or y_idx >= n_pcs:
        raise ValueError(
            f"components={components} out of range. "
            f"Only {n_pcs} PCs available (use 1-based indexing)."
        )

    pca_info = adata.uns.get("motif_abundance_pca", {})
    evr = pca_info.get("variance_ratio", [])

    pc_x = pc_mat[:, x_idx]
    pc_y = pc_mat[:, y_idx]
    pc_x_name = f"PC{components[0]}"
    pc_y_name = f"PC{components[1]}"

    # ---- colour mapping ----
    color_series = None
    color_is_numeric = False
    color_map: dict[str, str] | None = None
    if color_by is not None:
        if color_by not in adata.obs.columns:
            raise KeyError(f"color_by column '{color_by}' not found in adata.obs")
        color_series = adata.obs[color_by]
        color_is_numeric = pd.api.types.is_numeric_dtype(color_series)
        if not color_is_numeric:
            color_map = {
                str(v): _RAINBOW_COLORMAP[i % len(_RAINBOW_COLORMAP)]
                for i, v in enumerate(sorted(set(color_series.dropna().astype(str))))
            }

    # ---- shape mapping ----
    shape_series = None
    shape_map: dict[str, str] | None = None
    _SYMBOLS = [
        "circle", "square", "diamond", "cross", "x", "triangle-up",
        "triangle-down", "star", "hexagon", "pentagon", "octagon",
        "star-triangle-up", "star-square", "diamond-tall", "diamond-wide",
        "hourglass", "bowtie", "circle-cross", "square-cross",
        "triangle-left", "triangle-right",
    ]
    if shape_by is not None:
        if shape_by not in adata.obs.columns:
            raise KeyError(f"shape_by column '{shape_by}' not found in adata.obs")
        shape_series = adata.obs[shape_by]
        shape_map = {
            str(v): _SYMBOLS[i % len(_SYMBOLS)]
            for i, v in enumerate(sorted(set(shape_series.dropna().astype(str))))
        }

    # ---- build traces ----
    fig = go.Figure()

    def _make_hover(fmt: str) -> str:
        return fmt.replace("{pc_x}", pc_x_name).replace("{pc_y}", pc_y_name)

    def _add_default_trace():
        fig.add_trace(go.Scatter(
            x=pc_x, y=pc_y, mode="markers",
            marker=dict(size=marker_size, color="#277da1",
                        line=dict(width=1, color="DarkSlateGrey")),
            hovertemplate=_make_hover(
                "Sample: %{text}<br>{pc_x}: %{x:.2f}<br>{pc_y}: %{y:.2f}<extra></extra>"
            ),
            text=adata.obs.index,
            name="Samples",
        ))

    if color_by is None and shape_by is None:
        _add_default_trace()

    elif color_by is not None and shape_by is None:
        if color_is_numeric:
            fig.add_trace(go.Scatter(
                x=pc_x, y=pc_y, mode="markers",
                marker=dict(
                    size=marker_size, color=color_series,
                    colorscale=colormap or "Viridis",
                    colorbar=dict(title=str(color_by)),
                    line=dict(width=1, color="DarkSlateGrey"),
                ),
                hovertemplate=_make_hover(
                    "Sample: %{text}<br>{pc_x}: %{x:.2f}<br>{pc_y}: %{y:.2f}<br>"
                    + f"{color_by}: %{{marker.color:.2f}}<extra></extra>"
                ),
                text=adata.obs.index,
                name="Samples",
            ))
        else:
            for val in sorted(color_map.keys()):  # type: ignore[union-attr]
                mask = color_series.astype(str) == val  # type: ignore[union-attr]
                if mask.sum() == 0:
                    continue
                fig.add_trace(go.Scatter(
                    x=pc_x[mask], y=pc_y[mask], mode="markers",
                    marker=dict(size=marker_size, color=color_map[val],  # type: ignore[index]
                                line=dict(width=1, color="DarkSlateGrey")),
                    hovertemplate=_make_hover(
                        "Sample: %{text}<br>{pc_x}: %{x:.2f}<br>{pc_y}: %{y:.2f}<extra></extra>"
                    ),
                    text=adata.obs.index[mask],
                    name=str(val),
                ))

    elif color_by is None and shape_by is not None:
        for val in sorted(shape_map.keys()):  # type: ignore[union-attr]
            mask = shape_series.astype(str) == val  # type: ignore[union-attr]
            if mask.sum() == 0:
                continue
            fig.add_trace(go.Scatter(
                x=pc_x[mask], y=pc_y[mask], mode="markers",
                marker=dict(size=marker_size, symbol=shape_map[val],  # type: ignore[index]
                            color="#277da1",
                            line=dict(width=1, color="DarkSlateGrey")),
                hovertemplate=_make_hover(
                    "Sample: %{text}<br>{pc_x}: %{x:.2f}<br>{pc_y}: %{y:.2f}<extra></extra>"
                ),
                text=adata.obs.index[mask],
                name=str(val),
            ))

    else:  # both color_by and shape_by
        if color_is_numeric:
            sym_array = [shape_map.get(str(v), "circle") for v in shape_series]  # type: ignore[union-attr]
            fig.add_trace(go.Scatter(
                x=pc_x, y=pc_y, mode="markers",
                marker=dict(
                    size=marker_size, color=color_series,
                    colorscale=colormap or "Viridis",
                    colorbar=dict(title=str(color_by)),
                    symbol=sym_array,
                    line=dict(width=1, color="DarkSlateGrey"),
                ),
                hovertemplate=_make_hover(
                    "Sample: %{text}<br>{pc_x}: %{x:.2f}<br>{pc_y}: %{y:.2f}<br>"
                    + f"{color_by}: %{{marker.color:.2f}}<extra></extra>"
                ),
                text=adata.obs.index,
                name="Samples",
            ))
            for sval in sorted(shape_map.keys()):  # type: ignore[union-attr]
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=marker_size, symbol=shape_map[sval],  # type: ignore[index]
                                color="gray"),
                    name=str(sval), showlegend=True,
                ))
        else:
            for cval in sorted(color_map.keys()):  # type: ignore[union-attr]
                for sval in sorted(shape_map.keys()):  # type: ignore[union-attr]
                    mask = (
                        (color_series.astype(str) == cval)  # type: ignore[union-attr]
                        & (shape_series.astype(str) == sval)  # type: ignore[union-attr]
                    )
                    if mask.sum() == 0:
                        continue
                    fig.add_trace(go.Scatter(
                        x=pc_x[mask], y=pc_y[mask], mode="markers",
                        marker=dict(
                            size=marker_size,
                            color=color_map[cval],  # type: ignore[index]
                            symbol=shape_map[sval],  # type: ignore[index]
                            line=dict(width=1, color="DarkSlateGrey"),
                        ),
                        hovertemplate=_make_hover(
                            "Sample: %{text}<br>{pc_x}: %{x:.2f}<br>{pc_y}: %{y:.2f}<extra></extra>"
                        ),
                        text=adata.obs.index[mask],
                        name=f"{cval} | {sval}",
                    ))

    # ---- layout ----
    x_title, y_title = pc_x_name, pc_y_name
    if show_variance:
        if len(evr) > x_idx:
            x_title += f" ({evr[x_idx] * 100:.1f}%)"
        if len(evr) > y_idx:
            y_title += f" ({evr[y_idx] * 100:.1f}%)"

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        width=700 if figsize[0] is None else figsize[0],
        height=700 if figsize[1] is None else figsize[1],
        legend=dict(orientation="h", yanchor="top", y=-0.15,
                    xanchor="center", x=0.5),
        **kwargs,
    )
    fig.update_xaxes(showline=True, linecolor="black", ticks="outside")
    fig.update_yaxes(showline=True, linecolor="black", ticks="outside")

    if save:
        _save_figure(fig, save, "motif_abundance_pca")

    return fig


def motif_abundance_pca_variance(
    adata: "ad.AnnData",
    n_pcs: int | None = None,
    log: bool = False,
    show_cumulative: bool = True,
    figsize: tuple[int | None, int | None] = (None, None),
    title: str | None = None,
    save: str | bool | None = None,
    **kwargs,
) -> "go.Figure":
    """Plot variance explained by each principal component.

    Reads pre-computed results from ``vp.anno.tl.motif_abundance_pca()`` stored in
    ``uns['motif_abundance_pca']['variance_ratio']``.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with PCA results.
    n_pcs : int, optional
        Number of PCs to display.  If ``None``, display all.
    log : bool, default=False
        Use log scale for the variance-ratio axis.
    show_cumulative : bool, default=True
        Overlay a cumulative-variance line on the same y-axis.
    figsize : tuple[int | None, int | None], default=(None, None) to use (700, 600)
        Figure size in pixels.
    title : str | None, default=None
        Plot title.
    **kwargs
        Additional keyword arguments passed to ``fig.update_layout()``.
    save : str | bool | None, default=None
        If ``True`` or a ``str``, save the figure. A string is appended to the default filename.
        Infer the filetype if ending on {``'.pdf'``, ``'.png'``, ``'.svg'``}.

    Returns
    -------
    go.Figure
        Bar + line plot of per-PC variance ratios.

    Examples
    --------
    >>> import vampire as vp
    >>> vp.anno.pl.set_default_plotstyle()
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.tl.motif_abundance_pca(adata)
    >>> vp.anno.pl.motif_abundance_pca_variance(adata)
    """
    import numpy as np
    import plotly.graph_objects as go

    pca_info = adata.uns.get("motif_abundance_pca")
    if pca_info is None:
        raise KeyError(
            "PCA results not found. Run vp.anno.tl.motif_abundance_pca() first."
        )

    vr = np.array(pca_info.get("variance_ratio", []))
    if len(vr) == 0:
        raise ValueError("No variance_ratio data found.")

    if n_pcs is not None:
        vr = vr[:n_pcs]

    x = [f"PC{i + 1}" for i in range(len(vr))]

    fig = go.Figure()

    # Individual variance ratio (bar)
    fig.add_trace(go.Bar(
        x=x, y=vr,
        name="Individual",
        marker_color="#277da1",
        hovertemplate="%{x}<br>Variance: %{y:.4f}<extra></extra>",
    ))

    # Cumulative variance (line)
    if show_cumulative:
        cumsum = np.cumsum(vr)
        fig.add_trace(go.Scatter(
            x=x, y=cumsum,
            name="Cumulative",
            mode="lines+markers",
            line=dict(color="#f94144", width=2),
            marker=dict(size=6),
            hovertemplate="%{x}<br>Cumulative: %{y:.4f}<extra></extra>",
        ))

    yaxis_type = "log" if log else "linear"
    layout = dict(
        title=title,
        xaxis_title="Principal Component",
        yaxis=dict(
            range=[0, 1],
            title="Explained Variance Ratio", 
            type=yaxis_type
        ),
        width=700 if figsize[0] is None else figsize[0],
        height=600 if figsize[1] is None else figsize[1],
        legend=dict(orientation="h", yanchor="top", y=-0.15,
                    xanchor="center", x=0.5),
    )

    fig.update_layout(**layout, **kwargs)
    fig.update_xaxes(showline=True, linecolor="black", ticks="outside")
    fig.update_yaxes(showline=True, linecolor="black", ticks="outside")

    if save:
        _save_figure(fig, save, "motif_abundance_pca_variance")

    return fig