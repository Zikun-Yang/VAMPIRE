from __future__ import annotations
from collections import Counter
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
from ._setting import _get_categorical_colormap
from ._setting import _COLORMAP_OPTIONS # dict[str, list[str] | dict[str, str]]
from ..pp._markdup import markdup


def _resolve_colormap_for_annotation(
    ann_name: str,
    all_labels: list[str],
    colormap: (
        dict[str, str] | list[str] | str |
        dict[str, dict[str, str] | list[str] | str] | None
    ) = None,
) -> tuple[list[str], dict[str, str]]:
    """Resolve the colormap for a single annotation dimension.

    If ``colormap`` is a nested dict containing ``ann_name``, use the
    dimension-specific value. Otherwise apply the top-level colormap.
    """
    if isinstance(colormap, dict) and ann_name in colormap:
        return _get_categorical_colormap(all_labels, colormap[ann_name])
    return _get_categorical_colormap(all_labels, colormap)


def _extract_original_name(label: str) -> str:
    """Parse the original sample name from a deduplicated row label."""
    if " ... (n=" in label:
        return label.split(" ... (n=")[0]
    return label


def _is_stacked(annotation_dict: dict[str, list[list[str]]] | None) -> bool:
    """Return True if any annotation dimension contains multi-label lists."""
    if not annotation_dict:
        return False
    return any(
        any(len(labels) > 1 for labels in values)
        for values in annotation_dict.values()
    )


def heatmap_from_matrix(
    matrix: "np.ndarray",
    *,
    is_distance: bool = False,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    standard_scale: Literal["obs", "var", "zscore_obs", "zscore_var"] | None = None,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    row_cluster_method: str = "average",
    col_cluster_method: str = "average",
    row_cluster_metric: str = "euclidean",
    col_cluster_metric: str = "euclidean",
    colormap: str | list[str] | None = None,
    colorbar_title: str = "Value",
    showticklabels: bool = True,
    vmax: float | None = None,
    vmin: float | None = None,
    hover_template: str = "Row: %{y}<br>Col: %{x}<br>Value: %{hovertext}<extra></extra>",
    row_annotation: dict[str, list[list[str]]] | None = None,
    row_annotation_colormap: (
        dict[str, str] | list[str] | str |
        dict[str, dict[str, str] | list[str] | str] | None
    ) = None,
    col_annotation: dict[str, list[list[str]]] | None = None,
    col_annotation_colormap: (
        dict[str, str] | list[str] | str |
        dict[str, dict[str, str] | list[str] | str] | None
    ) = None,
    figsize: tuple[int | None, int | None] = (None, None),
    save: str | bool | None = None,
    **kwargs,
) -> "go.Figure":
    """
    Plot a clustered heatmap from an arbitrary numeric matrix.

    This is the generic engine underlying all domain-specific heatmap
    functions.  It accepts a raw 2-D numpy array, optionally clusters
    rows and/or columns, and returns an interactive Plotly figure with
    dendrograms.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array of shape (n_rows, n_cols).
    is_distance : bool, default=False
        Whether the matrix is alreadt distance matrix.
    row_labels : list[str] | None, optional
        Labels for rows.  If ``None``, integer indices are used.
    col_labels : list[str] | None, optional
        Labels for columns.  If ``None``, integer indices are used.
    standard_scale : {"obs", "var", "zscore_obs", "zscore_var"}, optional
        Standard scaling mode:

        - ``"obs"`` — min-max scale each row to [0, 1]
        - ``"var"`` — min-max scale each column to [0, 1]
        - ``"zscore_obs"`` — z-score standardize each row
        - ``"zscore_var"`` — z-score standardize each column

    cluster_rows : bool, default=True
        Whether to hierarchically cluster rows.
    cluster_cols : bool, default=True
        Whether to hierarchically cluster columns.
    row_cluster_method : str, default="average"
        Linkage method for row clustering.
    col_cluster_method : str, default="average"
        Linkage method for column clustering.
    row_cluster_metric : str, default="euclidean"
        Distance metric for row clustering.
    col_cluster_metric : str, default="euclidean"
        Distance metric for column clustering.
    colormap : str or list, optional
        Plotly colormap.
    colorbar_title : str, default="Value"
        Title shown next to the color bar.
    showticklabels : bool, default=True
        Whether to display row and column tick labels.
    vmax : float, optional
        Upper bound for clipping the heatmap color scale.
        Values above ``vmax`` are clipped for visualization only;
        the original values are still shown on hover.
    vmin : float, optional
        Lower bound for clipping the heatmap color scale.
        Values below ``vmin`` are clipped for visualization only.
    hover_template : str, optional
        Plotly hover template for the heatmap trace.
        Use ``%{text}`` to reference the un-clipped original value.
    row_annotation : dict[str, list[list[str]]] | None, optional
        Categorical annotation(s) for each row. Keys are dimension
        names (e.g. ``"haplotype"``) and values are lists of length
        ``n_rows``. Each inner list contains the labels for that row;
        multiple labels per row are rendered as a stacked proportion
        bar. Displayed as coloured sidebars between the row dendrogram
        and the heatmap.
    row_annotation_colormap : dict[str, str] | list[str] | str |
        dict[str, dict[str, str] | list[str] | str] | None, optional
        Color specification for row annotations. If a non-nested
        value is given, it applies to all dimensions. If a nested
        dict is given, keys must match dimension names and values
        are used for that dimension only. Missing dimensions fall
        back to auto-generated Glasbey colors. If ``None``, colours
        are auto-generated from the Glasbey palette. If a string is
        provided, it must be selected from ``rainbow``, ``glasbey``,
        or ``sequential`` to use the corresponding preset palette.
    col_annotation : dict[str, list[list[str]]] | None, optional
        Categorical annotation(s) for each column. Keys are dimension
        names and values are lists of length ``n_cols``. Each inner
        list contains the labels for that column; multiple labels per
        column are rendered as a stacked proportion bar. Each dimension
        is rendered as a separate coloured bar stacked between the
        column dendrogram and the heatmap.
    col_annotation_colormap : dict[str, str] | list[str] | str |
        dict[str, dict[str, str] | list[str] | str] | None, optional
        Color specification for column annotations. Same semantics as
        ``row_annotation_colormap``: non-nested values apply to all
        dimensions; nested dict keys must match dimension names.
    figsize : tuple[int | None, int | None], default=(None, None)
        Figure size in pixels. ``(None, None)`` triggers auto-computation
        from the matrix dimensions so that heatmap cells are square and
        labels are not clipped.
    save : str | bool | None, default=None
        If ``True`` or a ``str``, save the figure. A string is appended to the default filename.
        Infer the filetype if ending on {``'.pdf'``, ``'.png'``, ``'.svg'``}.
    **kwargs
        Additional keyword arguments passed to
        ``fig.update_layout()``.

    Returns
    -------
    go.Figure
        A Plotly figure containing the clustered heatmap with
        dendrograms and optional annotation blocks.
    """
    import plotly.graph_objects as go
    from scipy.cluster.hierarchy import linkage, dendrogram

    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("matrix must be 2-D")
    n_rows, n_cols = X.shape

    if n_rows == 0 or n_cols == 0:
        return go.Figure()

    has_row_labels = row_labels is not None
    has_col_labels = col_labels is not None
    if row_labels is None:
        row_labels = [str(i) for i in range(n_rows)]
    if col_labels is None:
        col_labels = [str(i) for i in range(n_cols)]

    if len(row_labels) != n_rows:
        raise ValueError("row_labels length must match matrix row count")
    if len(col_labels) != n_cols:
        raise ValueError("col_labels length must match matrix column count")
    if row_annotation is not None:
        for ann_name, ann_values in row_annotation.items():
            if len(ann_values) != n_rows:
                raise ValueError(
                    f"row_annotation['{ann_name}'] length ({len(ann_values)}) must match matrix row count ({n_rows})"
                )
            for i, v in enumerate(ann_values):
                if not isinstance(v, list):
                    raise TypeError(
                        f"row_annotation['{ann_name}'][{i}] must be a list, got {type(v)}"
                    )
    if col_annotation is not None:
        for ann_name, ann_values in col_annotation.items():
            if len(ann_values) != n_cols:
                raise ValueError(
                    f"col_annotation['{ann_name}'] length ({len(ann_values)}) must match matrix column count ({n_cols})"
                )
            for i, v in enumerate(ann_values):
                if not isinstance(v, list):
                    raise TypeError(
                        f"col_annotation['{ann_name}'][{i}] must be a list, got {type(v)}"
                    )

    # Standard scale
    if standard_scale == "obs":
        xmin = X.min(axis=1, keepdims=True)
        xmax = X.max(axis=1, keepdims=True)
        X = (X - xmin) / (xmax - xmin + 1e-12)
    elif standard_scale == "var":
        xmin = X.min(axis=0, keepdims=True)
        xmax = X.max(axis=0, keepdims=True)
        X = (X - xmin) / (xmax - xmin + 1e-12)
    elif standard_scale == "zscore_obs":
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, ddof=0, keepdims=True)
        X = (X - mean) / (std + 1e-12)
    elif standard_scale == "zscore_var":
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, ddof=0, keepdims=True)
        X = (X - mean) / (std + 1e-12)

    # Clustering
    row_order = list(range(n_rows))
    col_order = list(range(n_cols))
    row_dendro_data = None
    col_dendro_data = None

    # Detect distance matrix: square, symmetric, hollow, non-negative
    if is_distance and (cluster_rows or cluster_cols) and n_rows > 1:
        from scipy.spatial.distance import squareform
        row_linkage = linkage(
            squareform(X, checks=False),
            method=row_cluster_method,
        )
        if cluster_rows:
            row_dendro_data = dendrogram(
                row_linkage, no_plot=True, color_threshold=0,
                above_threshold_color="#000000",
            )
        if cluster_cols:
            # For a distance matrix row/col represent the same samples,
            # so col linkage is identical to row linkage.
            col_dendro_data = dendrogram(
                row_linkage, no_plot=True, color_threshold=0,
                above_threshold_color="#000000",
            )
    else:
        if cluster_rows and n_rows > 1:
            row_linkage = linkage(X, method=row_cluster_method, metric=row_cluster_metric)
            row_dendro_data = dendrogram(
                row_linkage, no_plot=True, color_threshold=0,
                above_threshold_color="#000000",
            )

        if cluster_cols and n_cols > 1:
            col_linkage = linkage(X.T, method=col_cluster_method, metric=col_cluster_metric)
            col_dendro_data = dendrogram(
                col_linkage, no_plot=True, color_threshold=0,
                above_threshold_color="#000000",
            )

    # Get leaf order
    if row_dendro_data is not None:
        row_order = row_dendro_data["leaves"]
    if col_dendro_data is not None:
        col_order = col_dendro_data["leaves"]

    # Reorder matrix and labels
    X_reordered = X[np.ix_(row_order, col_order)]
    row_labels_reordered = [str(row_labels[i]) for i in row_order]
    col_labels_reordered = [str(col_labels[i]) for i in col_order]

    row_annotations_reordered: dict[str, list[list[str]]] = {}
    if row_annotation is not None:
        row_annotations_reordered = {
            name: [values[i] for i in row_order]
            for name, values in row_annotation.items()
        }
    col_annotations_reordered: dict[str, list[list[str]]] = {}
    if col_annotation is not None:
        col_annotations_reordered = {
            name: [values[i] for i in col_order]
            for name, values in col_annotation.items()
        }

    n_col_annots = len(col_annotations_reordered)

    # Clip for visualization, but keep original values for hover
    X_display = np.clip(X_reordered, a_min=vmin, a_max=vmax)

    # Create figure manually (no subplots) so dendrogram and heatmap can share
    # axes and zoom / pan together.
    fig = go.Figure()

    # Pre-compute max distances for dendrogram axis ranges
    row_max_dist = 1.0
    if row_dendro_data is not None and row_dendro_data["dcoord"]:
        row_max_dist = max(max(d) for d in row_dendro_data["dcoord"])
    col_max_dist = 1.0
    if col_dendro_data is not None and col_dendro_data["dcoord"]:
        col_max_dist = max(max(d) for d in col_dendro_data["dcoord"])

    # Add column dendrogram (top) — shares x-axis with heatmap
    if col_dendro_data is not None:
        for x_pos, y_dist in zip(col_dendro_data["icoord"], col_dendro_data["dcoord"]):
            x_norm = [(xi - 5) / 10 for xi in x_pos]
            fig.add_trace(go.Scatter(
                x=x_norm, y=y_dist,
                mode="lines",
                line=dict(color="black", width=1.2),
                showlegend=False,
                hoverinfo="skip",
                xaxis="x",
                yaxis="y2",
            ))

    # Add row dendrogram (left) — shares y-axis with heatmap
    if row_dendro_data is not None:
        for x_dist, y_pos in zip(row_dendro_data["dcoord"], row_dendro_data["icoord"]):
            y_norm = [(yi - 5) / 10 for yi in y_pos]
            fig.add_trace(go.Scatter(
                x=[row_max_dist - xi for xi in x_dist],  # flip: leaves on the right
                y=y_norm,
                mode="lines",
                line=dict(color="black", width=1.2),
                showlegend=False,
                hoverinfo="skip",
                xaxis="x2",
                yaxis="y",
            ))

    # Add heatmap — shares x-axis with column dendrogram, y-axis with row dendrogram
    _DEFAULT_COLORMAP = [
        [0.0, "rgb(33, 102, 172)"],
        [0.5, "rgb(255, 255, 255)"],
        [1.0, "rgb(178, 34, 34)"],
    ] if (standard_scale is not None and "zscore" in standard_scale) and (X.min() < 0) else [
        [0.0, "rgb(255, 255, 255)"],
        [1.0, "rgb(178, 34, 34)"],
    ]
    _colormap = colormap if colormap is not None else _DEFAULT_COLORMAP
    fig.add_trace(go.Heatmap(
        z=X_display.tolist(),
        hovertext=[[f"{v:.4g}" for v in row] for row in X_reordered],
        x=list(range(n_cols)),
        y=list(range(n_rows)),
        colorscale=_colormap,
        showscale=True,
        colorbar=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            thickness=15,
            len=0.4,
            title=dict(text=colorbar_title, side="bottom"),
        ),
        hovertemplate=hover_template,
        xaxis="x",
        yaxis="y",
    ))

    # Add black border around the heatmap matrix
    fig.add_shape(
        type="rect",
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        xref="x domain",
        yref="y domain",
        line=dict(
            color="black",
            width=_sizing.get_active_line_width(),
        ),
        fillcolor="rgba(0,0,0,0)",
        layer="above",
    )

    # Annotation blocks
    all_row_palettes: dict[str, dict[str, str]] = {}
    n_row_annots = len(row_annotations_reordered)
    if row_annotations_reordered:
        for idx, (ann_name, ann_values) in enumerate(row_annotations_reordered.items()):
            xaxis_name = f"x{3 + idx}" if n_row_annots > 1 else "x3"
            all_labels = sorted(set(label for labels in ann_values for label in labels))
            label_order = {l: i for i, l in enumerate(all_labels)}
            _, palette = _resolve_colormap_for_annotation(
                ann_name, all_labels, row_annotation_colormap
            )
            all_row_palettes[ann_name] = palette

            # stacked horizontal bars: one trace per unique label
            for label in all_labels:
                widths: list[float] = []
                bases: list[float] = []
                y_positions: list[int] = []
                hover_texts: list[str] = []
                for row_idx, labels in enumerate(ann_values):
                    if not labels:
                        continue
                    cnt = Counter(labels)
                    total = len(labels)
                    base = sum(
                        cnt.get(all_labels[i], 0) / total
                        for i in range(label_order[label])
                    )
                    w = cnt.get(label, 0) / total
                    if w > 0:
                        widths.append(w)
                        bases.append(base)
                        y_positions.append(row_idx)
                        hover_texts.append(
                            f"{ann_name}: {label}<br>count: {int(w * total)}/{total}"
                        )
                if widths:
                    fig.add_trace(go.Bar(
                        x=widths,
                        y=y_positions,
                        base=bases,
                        marker=dict(color=palette[label]),
                        orientation="h",
                        width=1,
                        showlegend=False,
                        hoverinfo="text",
                        hovertext=hover_texts,
                        xaxis=xaxis_name,
                        yaxis="y",
                    ))

    all_col_palettes: dict[str, dict[str, str]] = {}
    if col_annotations_reordered:
        for idx, (ann_name, ann_values) in enumerate(col_annotations_reordered.items()):
            yaxis_name = f"y{3 + idx}" if n_col_annots > 1 else "y3"
            all_labels = sorted(set(label for labels in ann_values for label in labels))
            label_order = {l: i for i, l in enumerate(all_labels)}
            _, palette = _resolve_colormap_for_annotation(
                ann_name, all_labels, col_annotation_colormap
            )
            all_col_palettes[ann_name] = palette

            # stacked vertical bars: one trace per unique label
            for label in all_labels:
                heights: list[float] = []
                bases: list[float] = []
                x_positions: list[int] = []
                hover_texts: list[str] = []
                for col_idx, labels in enumerate(ann_values):
                    if not labels:
                        continue
                    cnt = Counter(labels)
                    total = len(labels)
                    base = sum(
                        cnt.get(all_labels[i], 0) / total
                        for i in range(label_order[label])
                    )
                    h = cnt.get(label, 0) / total
                    if h > 0:
                        heights.append(h)
                        bases.append(base)
                        x_positions.append(col_idx)
                        hover_texts.append(
                            f"{ann_name}: {label}<br>count: {int(h * total)}/{total}"
                        )
                if heights:
                    fig.add_trace(go.Bar(
                        x=x_positions,
                        y=heights,
                        base=bases,
                        marker=dict(color=palette[label]),
                        orientation="v",
                        width=1,
                        showlegend=False,
                        hoverinfo="text",
                        hovertext=hover_texts,
                        xaxis="x",
                        yaxis=yaxis_name,
                    ))

    # Resolve figsize with auto-sizing
    font_size = kwargs.get("font", {}).get("size")
    if font_size is None:
        font_size = _sizing.get_active_font_size()

    max_row_label_len = max(len(l) for l in row_labels) if row_labels else 0
    max_col_label_len = max(len(l) for l in col_labels) if col_labels else 0

    _is_auto = figsize[0] is None or figsize[1] is None
    # Dynamic margins so long labels are not clipped.
    char_width = font_size * 0.7
    _l_margin = 80
    _t_margin = 100
    _r_margin = 120
    _b_margin = max(120, int(max(max_row_label_len, max_col_label_len) * char_width + 40))
    _width, _height = _sizing.resolve_figsize(
        figsize[0],
        figsize[1],
        calc_width=lambda: _sizing.heatmap_width(
            n_cols, font_size, cluster_rows=cluster_rows,
            n_row_annotations=n_row_annots,
            l_margin=_l_margin, r_margin=_r_margin
        ),
        calc_height=lambda: _sizing.heatmap_height(
            n_rows, font_size, cluster_cols=cluster_cols, n_col_annotations=n_col_annots,
            t_margin=_t_margin, b_margin=_b_margin
        ),
    )
    figsize = (_width, _height)

    # Domain layout — space allocation follows cluster_* parameters exactly.
    # When auto-sizing, dendrogram / annotation sizes are fixed pixels so the
    # heatmap area lines up precisely with n_cols * cell_px × n_rows * cell_px.
    # When the user supplies figsize, the original ratio-based behaviour is kept.
    dendro_px = _sizing.HEATMAP_DENDROGRAM_PX
    annot_px = _sizing.HEATMAP_ANNOTATION_PX
    annot_gap_px = 2

    # Domain layout
    plot_w = max(figsize[0] - _l_margin - _r_margin, 1)
    plot_h = max(figsize[1] - _t_margin - _b_margin, 1)

    gap_per_x = annot_gap_px / plot_w
    gap_per_y = annot_gap_px / plot_h
    row_stacked = _is_stacked(row_annotations_reordered)
    col_stacked = _is_stacked(col_annotations_reordered)
    annot_w_per = annot_px / plot_w * (3 if row_stacked else 1)
    annot_h_per = annot_px / plot_h * (3 if col_stacked else 1)

    x_dendro_w = dendro_px / plot_w if cluster_rows else 0.0
    x_annot_gap = gap_per_x if n_row_annots > 0 else 0.0
    x_annot_w_total = annot_w_per * n_row_annots + gap_per_x * max(0, n_row_annots - 1)
    x_heatmap_left = x_dendro_w + x_annot_w_total + 2 * x_annot_gap

    y_dendro_h = dendro_px / plot_h if cluster_cols else 0.0
    y_annot_gap = gap_per_y if n_col_annots > 0 else 0.0
    y_annot_h_total = annot_h_per * n_col_annots + gap_per_y * max(0, n_col_annots - 1)
    y_heatmap_top = 1.0 - y_dendro_h - y_annot_h_total - 2 * y_annot_gap

    # Register x-axis for each row annotation dimension
    if n_row_annots > 0:
        for idx, ann_name in enumerate(row_annotations_reordered.keys()):
            xaxis_name = f"x{3 + idx}" if n_row_annots > 1 else "x3"
            # trace references axis as "x3", but layout property name is "xaxis3"
            layout_xaxis_name = "xaxis" + xaxis_name[1:]
            domain_start = x_dendro_w + x_annot_gap + idx * (annot_w_per + gap_per_x)
            domain_end = domain_start + annot_w_per
            fig.update_layout(**{
                layout_xaxis_name: dict(
                    domain=[domain_start, domain_end],
                    range=[0, 1],
                    showticklabels=False,
                    showline=False,
                    automargin=False,
                    mirror=False,
                    showgrid=False,
                    zeroline=False,
                    ticks="",
                )
            })
            # 100% label (above annotation column) only when stacked
            if _is_stacked(row_annotations_reordered):
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=(domain_start + domain_end) / 2,
                    y=y_heatmap_top + gap_per_y,
                    text="100%",
                    showarrow=False,
                    xanchor="center",
                    yanchor="bottom",
                    font=dict(size=font_size),
                )

            # Dimension name label (vertical, below annotation column)
            fig.add_annotation(
                xref=xaxis_name,
                yref="paper",
                x=0.5,
                y=-3.0 / plot_h,
                text=ann_name,
                showarrow=False,
                textangle=-90,
                xanchor="center",
                yanchor="top",
                font=dict(size=font_size),
            )

    # Register y-axis for each column annotation dimension
    if n_col_annots > 0:
        for idx, ann_name in enumerate(col_annotations_reordered.keys()):
            yaxis_name = f"y{3 + idx}" if n_col_annots > 1 else "y3"
            layout_yaxis_name = "yaxis" + yaxis_name[1:]
            domain_start = y_heatmap_top + y_annot_gap + idx * (annot_h_per + gap_per_y)
            domain_end = domain_start + annot_h_per
            fig.update_layout(**{
                layout_yaxis_name: dict(
                    domain=[domain_start, domain_end],
                    range=[0, 1],
                    showticklabels=False,
                    showline=False,
                    automargin=False,
                    mirror=False,
                    showgrid=False,
                    zeroline=False,
                    ticks="",
                )
            })
            # 100% label (vertical, to the right of annotation row) only when stacked
            if _is_stacked(col_annotations_reordered):
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=1.0 + gap_per_x,
                    y=(domain_start + domain_end) / 2,
                    text="100%",
                    showarrow=False,
                    textangle=-90,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=font_size),
                )

            # Dimension name label (horizontal, to the right of annotation row)
            fig.add_annotation(
                xref="paper",
                yref=yaxis_name,
                x=1.0 + 3.0 / plot_w,
                y=0.5,
                text=ann_name,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=font_size),
            )

    # Add legend entries — one legend per annotation dimension
    legend_infos = []  # list of (legend_id, title, names)
    legend_idx = 0
    for ann_name, palette in all_row_palettes.items():
        legend_id = f"legend{legend_idx + 1}" if legend_idx > 0 else "legend"
        for cat, color in palette.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                showlegend=True,
                name=cat,
                legend=legend_id,
            ))
        legend_infos.append((legend_id, ann_name, list(palette.keys())))
        legend_idx += 1
    for ann_name, palette in all_col_palettes.items():
        legend_id = f"legend{legend_idx + 1}" if legend_idx > 0 else "legend"
        for cat, color in palette.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                showlegend=True,
                name=cat,
                legend=legend_id,
            ))
        legend_infos.append((legend_id, ann_name, list(palette.keys())))
        legend_idx += 1

    # ---- Legend layout estimation ----
    def _estimate_legend_layout(names, title, avail_width, font_size):
        """
        Estimate the number of wrapped legend rows and the total legend height in pixels.
        """
        if not names:
            return 0, 0

        _marker_px = 20   # marker + left padding
        _gap_px = 10      # entry spacing distance
        _char_px = font_size * 0.7

        entry_widths = [len(str(n)) * _char_px + _marker_px + _gap_px for n in names]

        lines = 0
        current_width = 0
        for w in entry_widths:
            if current_width + w > avail_width and current_width > 0:
                lines += 1
                current_width = w
            else:
                current_width += w

        if current_width > 0:
            lines += 1

        _title_h = font_size + 8 if title else 0
        _line_h = font_size + 6
        total_h = _title_h + lines * _line_h + 8  # bottom padding

        return lines, int(total_h)

    _avail_width = max(_width - 60, 100)
    _legend_gap = 30  # legend spacing distance

    plot_height = max(_height - 100 - _b_margin, 1)

    # colorbar position
    cb_y = -50 / plot_height
    fig.update_traces(
        selector=dict(type="heatmap"),
        colorbar=dict(y=cb_y, yanchor="top"),
    )

    # Build per-dimension legend layout params, stacked vertically below the plot
    legend_layouts = {}
    current_y_offset_px = 150  # first legend top distance from plot bottom
    for legend_id, title, names in legend_infos:
        lines, h = _estimate_legend_layout(names, title, _avail_width, font_size)
        legend_layouts[legend_id] = dict(
            orientation="h",
            yanchor="top",
            y=-current_y_offset_px / plot_height,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            title=dict(text=title, side="top"),
        )
        current_y_offset_px += h + _legend_gap
    fig.update_layout(
        xaxis=dict( # heatmap
            domain=[x_heatmap_left, 1],
            range=[-0.5, n_cols - 0.5],
            tickvals=list(range(n_cols)),
            ticktext=col_labels_reordered if (showticklabels and has_col_labels) else [],
            tickangle=90,
            showticklabels=showticklabels and has_col_labels,
            showline=False,
            automargin=False,
            mirror=False,
            showgrid=False,
            zeroline=False,
            ticks="outside" if (showticklabels and has_col_labels) else "",
        ),
        xaxis2=dict( # row dendrogram
            domain=[0, x_dendro_w],
            range=[0, row_max_dist],
            showticklabels=False,
            showline=False,
            automargin=False,
            mirror=False,
            showgrid=False,
            zeroline=False,
            ticks="",
        ),
        yaxis=dict( # heatmap
            domain=[0, y_heatmap_top],
            range=[-0.5, n_rows - 0.5],
            tickvals=list(range(n_rows)),
            ticktext=row_labels_reordered if (showticklabels and has_row_labels) else [],
            showticklabels=showticklabels and has_row_labels,
            side="right",
            showline=False,
            automargin=False,
            mirror=False,
            showgrid=False,
            zeroline=False,
            ticks="outside" if (showticklabels and has_row_labels) else "",
        ),
        yaxis2=dict( # column dendrogram
            domain=[y_heatmap_top + y_annot_h_total + 2 * y_annot_gap, 1],
            range=[0, col_max_dist],
            showticklabels=False,
            showline=False,
            automargin=False,
            mirror=False,
            showgrid=False,
            zeroline=False,
            ticks="",
        ),
        bargap=0,
        barmode="overlay",
        width=figsize[0],
        height=figsize[1],
        **legend_layouts,
        margin=dict(l=_l_margin, r=_r_margin, t=_t_margin, b=_b_margin),
    )

    fig.update_layout(**kwargs)

    if save:
        _save_figure(fig, save, "heatmap_from_matrix")

    return fig


def motif_abundance_heatmap(
    adata: "ad.AnnData",
    *,
    layer: str | None = None,
    standard_scale: Literal["obs", "var", "zscore_obs", "zscore_var"] | None = "obs",
    deduplicate = False,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    row_cluster_method: str = "average",
    col_cluster_method: str = "average",
    row_cluster_metric: str = "euclidean",
    col_cluster_metric: str = "euclidean",
    colormap: str | list[str] | None = None,
    showticklabels: bool = True,
    vmax: float | None = None,
    vmin: float | None = None,
    row_annotation: str | list[str] | dict[str, dict[str, str]] | None = None,
    row_annotation_colormap: (
        dict[str, str] | list[str] | str |
        dict[str, dict[str, str] | list[str] | str] | None
    ) = None,
    col_annotation: str | list[str] | dict[str, dict[str, str]] | None = None,
    col_annotation_colormap: (
        dict[str, str] | list[str] | str |
        dict[str, dict[str, str] | list[str] | str] | None
    ) = None,
    figsize: tuple[int | None, int | None] | None = (None, None),
    save: str | bool | None = None,
    **kwargs,
) -> "go.Figure":
    """
    Plot a sample × motif abundance heatmap with hierarchical clustering
    and dendrograms.

    This is a convenience wrapper around :func:`matrix_heatmap` that
    extracts the motif abundance matrix from an ``AnnData`` object.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object generated from ``pp.read_anno()``.
    layer : str, optional
        Layer in ``adata.layers`` to use.  If ``None``, uses ``adata.X``.
    standard_scale : {"obs", "var", "zscore_obs", "zscore_var"}, optional, default is ``"obs"``
        Standard scaling mode:

        - ``"obs"`` — min-max scale each row (sample) to [0, 1]
        - ``"var"`` — min-max scale each column (motif) to [0, 1]
        - ``"zscore_obs"`` — z-score standardize each row (sample)
        - ``"zscore_var"`` — z-score standardize each column (motif)

    deduplicate : bool, default=False
        If True, collapse samples with identical motif arrays into a single
        row. The track label shows the first sample name followed by
        ``... (n=X)`` where X is the number of collapsed samples. The draw
        order follows the position of the first occurrence in ``sample_order``.
    cluster_rows : bool, default=True
        Whether to hierarchically cluster rows (samples).
    cluster_cols : bool, default=True
        Whether to hierarchically cluster columns (motifs).
    row_cluster_method : str, default="average"
        Linkage method for row clustering.
    col_cluster_method : str, default="average"
        Linkage method for column clustering.
    row_cluster_metric : str, default="euclidean"
        Distance metric for row clustering.
    col_cluster_metric : str, default="euclidean"
        Distance metric for column clustering.
    colormap : str or list[str], optional
        Plotly colormap name.  Default is white to red.
    showticklabels : bool, default=True
        Whether to display row and column tick labels.
    vmax : float, optional
        Upper bound for clipping the heatmap color scale.
        Values above ``vmax`` are clipped for visualization only.
    vmin : float, optional
        Lower bound for clipping the heatmap color scale.
        Values below ``vmin`` are clipped for visualization only.
    row_annotation : str | list[str] | dict[str, dict[str, str]] | None, optional
        Categorical annotation(s) for each row.

        - ``str`` — column name in ``adata.obs``; values are read and
          wrapped into the unified dict format.
        - ``list[str]`` — multiple column names in ``adata.obs``; each
          becomes a separate annotation dimension.
        - ``dict[str, dict[str, str]]`` — keys are dimension names and
          values are ``{sample_name -> category}`` mappings. Looked up
          by sample name so the order is safe.

        When ``deduplicate=True``, inputs are aggregated per sample group
        so collapsed rows show stacked proportions.
    row_annotation_colormap : dict[str, str] | list[str] | str |
        dict[str, dict[str, str] | list[str] | str] | None, optional
        Color specification for row annotations. Non-nested values apply
        to all dimensions; nested dict keys must match dimension names.
    col_annotation : str | list[str] | dict[str, dict[str, str]] | None, optional
        Categorical annotation(s) for each column. Same semantics as
        ``row_annotation``, but reads from ``adata.var`` instead of
        ``adata.obs``.
    col_annotation_colormap : dict[str, str] | list[str] | str |
        dict[str, dict[str, str] | list[str] | str] | None, optional
        Color specification for column annotations. Non-nested values apply
        to all dimensions; nested dict keys must match dimension names.
    figsize : tuple[int | None, int | None] | None, default=(None, None)
        Figure size in pixels.  ``(None, None)`` triggers auto-computation
        from the matrix dimensions so that heatmap cells are square and
        labels are not clipped.
    save : str | bool | None, default=None
        If ``True`` or a ``str``, save the figure. A string is appended to the default filename.
        Infer the filetype if ending on {``'.pdf'``, ``'.png'``, ``'.svg'``}.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`matrix_heatmap` and ``fig.update_layout()``.

    Returns
    -------
    go.Figure
        A Plotly figure containing the clustered heatmap with
        dendrograms.

    Examples
    --------
    >>> import vampire as vp
    >>> vp.anno.pl.set_default_plotstyle()
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.pl.motif_abundance_heatmap(
    ...     adata,
    ...     cluster_rows=True,
    ...     cluster_cols=True,
    ...     standard_scale="obs",
    ... )
    """
    # Extract data matrix
    X = adata.X if layer is None else adata.layers[layer]
    if hasattr(X, "toarray"):
        X = X.toarray()

    row_labels = [str(l) for l in adata.obs.index]
    col_labels = [str(l) for l in adata.var.index]

    if deduplicate:
        if "unique_group" not in adata.obs.columns:
            logger.warning(
                "unique_group not found in adata.obs. "
                "vp.anno.pp.markdup() has not been run. Running it automatically."
            )
            adata = markdup(adata)

        obs = adata.obs.copy()
        name_to_group: dict[str, int] = obs["unique_group"].to_dict()
        group_to_names: dict[int, list[str]] = {}
        for name, group in name_to_group.items():
            if group not in group_to_names:
                group_to_names[group] = []
            group_to_names[group].append(name)

        # keep first occurrence of each group
        seen_groups = {}
        keep_idx = []
        for i, name in enumerate(obs.index.astype(str)):
            gid = obs.loc[name, "unique_group"]
            if gid not in seen_groups:
                seen_groups[gid] = i
                keep_idx.append(name)

        # subset matrix
        X = X[adata.obs.index.get_indexer(keep_idx), :]

        # update labels
        row_labels = []
        for name in keep_idx:
            gid = name_to_group[name]
            gsize = len(group_to_names[gid])
            if gsize == 1:
                row_labels.append(name)
            else:
                row_labels.append(f"{name} ... (n={gsize})")

    # resolve row_annotation into dict[str, list[list[str]]] | None
    row_annotation_dict: dict[str, list[list[str]]] | None = None
    if row_annotation is not None:
        if isinstance(row_annotation, str):
            row_annotation = [row_annotation]

        if isinstance(row_annotation, list):
            row_annotation_dict = {}
            for col_name in row_annotation:
                if col_name not in adata.obs.columns:
                    raise ValueError(
                        f"row_annotation column '{col_name}' not found in adata.obs.columns"
                    )
                if deduplicate:
                    row_annotation_dict[col_name] = [
                        [
                            str(adata.obs.loc[s, col_name])
                            for s in group_to_names[name_to_group[_extract_original_name(l)]]
                        ]
                        for l in row_labels
                    ]
                else:
                    row_annotation_dict[col_name] = [
                        [str(adata.obs.loc[i, col_name])]
                        for i in row_labels
                    ]
        elif isinstance(row_annotation, dict):
            row_annotation_dict = {}
            for name, mapping in row_annotation.items():
                if deduplicate:
                    row_annotation_dict[name] = [
                        [
                            str(mapping.get(s, ""))
                            for s in group_to_names[name_to_group[_extract_original_name(l)]]
                        ]
                        for l in row_labels
                    ]
                else:
                    row_annotation_dict[name] = [
                        [str(mapping.get(i, ""))]
                        for i in row_labels
                    ]
        else:
            raise TypeError(
                f"row_annotation must be str, list[str], dict[str, dict[str, str]] or None, "
                f"got {type(row_annotation)}"
            )

    # resolve col_annotation into dict[str, list[list[str]]] | None
    col_annotation_dict: dict[str, list[list[str]]] | None = None
    if col_annotation is not None:
        if isinstance(col_annotation, str):
            col_annotation = [col_annotation]

        if isinstance(col_annotation, list):
            col_annotation_dict = {}
            for col_name in col_annotation:
                if col_name not in adata.var.columns:
                    raise ValueError(
                        f"col_annotation column '{col_name}' not found in adata.var.columns"
                    )
                col_annotation_dict[col_name] = [
                    [str(adata.var.loc[i, col_name])]
                    for i in col_labels
                ]
        elif isinstance(col_annotation, dict):
            col_annotation_dict = {
                name: [
                    [str(mapping.get(i, ""))]
                    for i in col_labels
                ]
                for name, mapping in col_annotation.items()
            }
        else:
            raise TypeError(
                f"col_annotation must be str, list[str], dict[str, dict[str, str]] or None, "
                f"got {type(col_annotation)}"
            )

    fig = heatmap_from_matrix(
        matrix=X,
        is_distance=False,
        row_labels=row_labels,
        col_labels=col_labels,
        standard_scale=standard_scale,
        cluster_rows=cluster_rows,
        cluster_cols=cluster_cols,
        row_cluster_method=row_cluster_method,
        col_cluster_method=col_cluster_method,
        row_cluster_metric=row_cluster_metric,
        col_cluster_metric=col_cluster_metric,
        colormap=colormap,
        showticklabels=showticklabels,
        figsize=figsize,
        vmax=vmax,
        vmin=vmin,
        row_annotation=row_annotation_dict,
        row_annotation_colormap=row_annotation_colormap,
        col_annotation=col_annotation_dict,
        col_annotation_colormap=col_annotation_colormap,
        colorbar_title="Abundance",
        hover_template="Sample: %{y}<br>Motif: %{x}<br>Value: %{hovertext}<extra></extra>",
        **kwargs,
    )

    if save:
        _save_figure(fig, save, "motif_abundance_heatmap")

    return fig


def haplotype_distance_heatmap(
    adata: "ad.AnnData",
    *,
    store_key: str = "haplotype",
    metric: str = "structural",
    deduplicate: bool = False,
    reorder: bool = True,
    cluster: bool = False,
    colormap: str | list[str] | None = None,
    figsize: tuple[int | None, int | None] | None = (None, None),
    save: str | bool | None = None,
    **kwargs,
) -> "go.Figure":
    """Plot sample pairwise distance matrix from haplotype analysis.

    Visualises one of the distance matrices stored in ``obsp`` by
    ``vp.anno.tl.haplotype_neighbor()``.  Samples are annotated by their haplotype
    assignment so that the block structure is visible.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with haplotype results from ``vp.anno.tl.haplotype_neighbor()``.
    store_key : str, default="haplotype"
        Key prefix matching ``store_key`` used in ``vp.anno.tl.haplotype_neighbor()``.
    metric : str, default="structural"
        Which distance matrix to visualise.  Options:
        ``"structural"``, ``"cnv"``.
    deduplicate : bool, default=False
        If True, collapse samples with identical motif arrays into a single
        row. The track label shows the first sample name followed by
        ``... (n=X)`` where X is the number of collapsed samples. The draw
        order follows the position of the first occurrence in ``sample_order``.
    reorder : bool, default=True
        If ``True``, rows and columns are sorted by haplotype label
        so that samples from the same haplotype are adjacent.
    cluster : bool, default=False
        If ``True``, hierarchically cluster rows and columns
        (overrides ``reorder``).
    colormap : str | list[str] | None, default=None
        Plotly colormap for the heatmap. If ``None``, defaults to a red-to-white scale.
    figsize : tuple[int | None, int | None] | None, default=(None, None)
        Figure size in pixels.  ``(None, None)`` triggers auto-computation
        from the matrix dimensions so that heatmap cells are square and
        labels are not clipped.
    save : str | bool | None, default=None
        If ``True`` or a ``str``, save the figure. A string is appended to the default filename.
        Infer the filetype if ending on {``'.pdf'``, ``'.png'``, ``'.svg'``}.
    **kwargs
        Additional arguments passed to ``heatmap_from_matrix``.

    Returns
    -------
    go.Figure
        Plotly figure with the distance matrix heatmap.

    Examples
    --------
    >>> import vampire as vp
    >>> vp.anno.pl.set_default_plotstyle()
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.tl.sample_msa(adata)
    >>> vp.anno.tl.haplotype_neighbor(adata)
    >>> vp.anno.pl.haplotype_distance_heatmap(adata, metric = "structural")
    """
    import numpy as np

    metric_key = f"{store_key}_{metric}_distance"
    dist_mat = adata.obsp.get(metric_key)
    if dist_mat is None:
        raise KeyError(
            f"Distance matrix not found at obsp['{metric_key}']. "
            f"Run tl.haplotype_neighbor() first."
        )

    if store_key not in adata.obs.columns:
        raise KeyError(
            f"'{store_key}' not found at adata.obs. "
            f"Run vp.anno.tl.haplotype_neighbor() first."
        )

    if deduplicate:
        if "unique_group" not in adata.obs.columns:
            logger.warning(
                "unique_group not found in adata.obs. "
                "vp.anno.pp.markdup() has not been run. Running it automatically."
            )
            adata = markdup(adata)

        obs = adata.obs.copy()
        name_to_group: dict[str, int] = obs["unique_group"].to_dict()
        group_to_names: dict[int, list[str]] = {}
        for name, group in name_to_group.items():
            if group not in group_to_names:
                group_to_names[group] = []
            group_to_names[group].append(name)

        # keep first occurrence of each group
        seen_groups = {}
        keep_idx = []
        for i, name in enumerate(obs.index.astype(str)):
            gid = obs.loc[name, "unique_group"]
            if gid not in seen_groups:
                seen_groups[gid] = i
                keep_idx.append(name)

        # subset matrix (keep symmetric)
        keep_positions = adata.obs.index.get_indexer(keep_idx)
        dist_mat = dist_mat[np.ix_(keep_positions, keep_positions)]

        # update labels
        names: list[str] = []
        annotation: dict[str, Any] = {}
        for name in keep_idx:
            gid = name_to_group[name]
            gsize = len(group_to_names[gid])
            if gsize == 1:
                names.append(name)
                annotation[name] = adata.obs.loc[name, store_key]
            else:
                compact_name = f"{name} ... (n={gsize})"
                names.append(compact_name)
                annotation[compact_name] = adata.obs.loc[name, store_key]

        annotation_list: list[str] = [annotation[i] for i in names]
    else:
        annotation: dict[str, str] = adata.obs[store_key].to_dict()
        names: list[str] = list(adata.obs_names)
        annotation_list: list[str] = [annotation[i] for i in names]

    # sort by haplotype
    def _haplotype_sort_key(x):
        import re
        m = re.search(r"\d+", str(x))
        if m:
            return int(m.group())
        return float("inf")
    
    if reorder and not cluster:
        ###sort_idx = np.argsort(annotation_list)
        sort_idx = sorted(
            range(len(annotation_list)),
            key=lambda i: _haplotype_sort_key(annotation_list[i])
        )
        dist_mat = dist_mat[np.ix_(sort_idx, sort_idx)]
        names = [names[i] for i in sort_idx]
        annotation_list = [annotation[i] for i in names]

    _DEFAULT_COLORMAP = [
        [0.0, "rgb(178, 34, 34)"],
        [1.0, "rgb(255, 255, 255)"],
    ]

    row_annotation_dict = {"Haplotype": [[h] for h in annotation_list]}

    fig = heatmap_from_matrix(
        matrix=dist_mat,
        is_distance=True,
        row_labels=names,
        col_labels=None,
        cluster_rows=cluster,
        cluster_cols=cluster,
        colormap=colormap or _DEFAULT_COLORMAP,
        figsize=figsize,
        colorbar_title="Distance",
        row_annotation=row_annotation_dict,
        col_annotation={"Haplotype": [[v] for v in annotation_list]},
        hover_template="Sample: %{y}<br>Sample: %{x}<br>Distance: %{hovertext}<extra></extra>",
        legend2=dict(visible=False),
        **kwargs,
    )

    if save:
         _save_figure(fig, save, "haplotype_distance_heatmap")

    return fig