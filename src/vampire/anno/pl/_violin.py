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
from ._setting import _get_categorical_colormap
from ._setting import _COLORMAP_OPTIONS # dict[str, list[str] | dict[str, str]]


def copy_number_violin(
    adata: "ad.AnnData",
    *,
    group_by: str | None = None,
    group_order: Sequence[str] | None = None,
    motif: str | int | None = None,
    log: bool = False,
    show_box: bool = True,
    show_points: bool = False,
    show_counts: bool = True,
    colormap: dict[str, str] | list[str] | str | None = None,
    figsize: tuple[int | None, int | None] = (None, None),
    save: str | bool | None = None,
    **kwargs,
) -> "go.Figure":
    """
    Plot copy-number distribution across sample groups as a violin plot.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with copy-number information.
    group_by : str | none, default is None
        Column name in ``adata.obs`` used to group samples. If it is None, plot without grouping
    group_order : Sequence[str] | None, default=None
        Explicit order for the groups on the x-axis.  If ``None``, groups are sorted alphabetically.
    motif : str | int | None, default=None
        If ``None``, the total copy number per sample
        (``adata.obs["copy_number"]``) is used.
        If ``str``, the motif is looked up in ``adata.var.index`` first,
        then in ``adata.var["motif"]``, and the matching column from
        ``adata.X`` is used.
        If ``int``, ``adata.X[:, motif]`` is used directly.
    log : bool, default=False
        Whether to apply ``log1p`` transform to copy-number values before plotting.
    show_box : bool, default=True
        Whether to overlay a mini box plot inside each violin.
    show_points : bool, default=False
        Whether to overlay individual data points on each violin.
    show_counts : bool, default=True
        If ``True``, the x-axis tick label of each group shows the sample
        count on a second line (``<group><br>n={count}``).
    colormap : dict[str, str] | list[str] | str | None, default=None
        Colormap for the violins.  If a ``str``, it is looked up in the module and plotly default colormap options.  
        If a list of color strings, used directly.  
        If a dict, keys are group names and values are color strings.  
        If ``None``, the module default rainbow colormap is used.
    figsize : tuple[int | None, int | None], default=(None, None)
        Figure size in pixels.
    save : str | bool | None, default=None
        If ``True`` or a ``str``, save the figure. A string is appended to the default filename.
        Infer the filetype if ending on {``'.pdf'``, ``'.png'``, ``'.svg'``}.
    **kwargs
        Additional arguments passed to ``fig.update_layout()``.

    Returns
    -------
    go.Figure
        Plotly figure with the violin plot.

    Examples
    --------
    >>> import vampire as vp
    >>> vp.anno.pl.set_default_plotstyle()
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.pl.copy_number_violin(adata, group_by="haplotype")
    >>> vp.anno.pl.copy_number_violin(adata, group_by="ancestry", motif=0)
    """
    import plotly.graph_objects as go
    import numpy as np

    if group_by is not None and group_by not in adata.obs.columns:
        raise KeyError(f"group_by column '{group_by}' not found in adata.obs")

    # Resolve copy-number vector
    if motif is None:
        if "copy_number" not in adata.obs.columns:
            raise KeyError(
                "adata.obs['copy_number'] not found. "
                "Ensure the AnnData object has total copy-number per sample."
            )
        y = adata.obs["copy_number"].to_numpy(dtype=float)
        y_title = "Copy number"
    elif isinstance(motif, int):
        if motif < 0 or motif >= adata.n_vars:
            raise IndexError(
                f"motif index {motif} out of range for {adata.n_vars} motifs"
            )
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        y = np.ravel(np.asarray(X, dtype=float)[:, motif])
        y_title = "Motif copy number"
    elif isinstance(motif, str):
        if motif in adata.var.index:
            idx = adata.var.index.get_loc(motif)
        elif "motif" in adata.var.columns and motif in adata.var["motif"].values:
            idx = adata.var["motif"].tolist().index(motif)
        else:
            raise KeyError(
                f"motif '{motif}' not found in adata.var.index or adata.var['motif']"
            )
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        y = np.ravel(np.asarray(X, dtype=float)[:, idx])
        y_title = "Motif copy number"
    else:
        raise TypeError(f"motif must be str, int, or None, got {type(motif).__name__}")

    if group_by is None:
        groups = ["All"]
        group_labels = np.array(["All"] * adata.n_obs)
        xaxis_title = "All"
    else:
        group_labels = adata.obs[group_by].astype(str).to_numpy()
        all_groups = set(adata.obs[group_by].dropna().unique())
        if group_order is not None:
            groups = list(group_order)
            missing = set(all_groups) - set(groups)
            if missing:
                raise ValueError(
                    f"group_order contains unknown groups: {missing}"
                )

        else:
            groups = sorted(adata.obs[group_by].dropna().unique(), key=str)
        
        if len(groups) == 0:
            raise ValueError(f"No valid groups found in adata.obs['{group_by}']")
        xaxis_title = group_by
    
    # get real font size: user override > active template > fallback
    font_size = kwargs.get("font", {}).get("size")
    if font_size is None:
        font_size = _sizing.get_active_font_size()

    n_groups: int = len(groups)

    width, height = _sizing.resolve_figsize(
        figsize[0],
        figsize[1],
        calc_width=lambda: _sizing.violin_width(n_groups, font_size),
        calc_height=lambda: _sizing.violin_height(font_size),
    )

    fig = go.Figure()
    group_counts = {}
    _, colormap = _get_categorical_colormap(groups, colormap if colormap is not None else _COLORMAP_OPTIONS["rainbow"])
    for i, group in enumerate(groups):
        if group_by is None:
            mask = np.ones(adata.n_obs, dtype=bool)
        else:
            mask = (adata.obs[group_by] == group).to_numpy()
        group_counts[group] = int(mask.sum())
        group_y = y[mask]
        x_vals = [str(group)] * len(group_y)
        
        trace_kwargs = dict(
            x=x_vals,
            y=group_y,
            name=str(group),
            fillcolor=colormap[group],
            line=dict(color=colormap[group], width=1.2),
            opacity=0.8,
            width=0.6,
            spanmode="hard",
            side="both",
            box_visible=show_box,
            box=dict(fillcolor="white", line=dict(color="black", width=1.2)),
            meanline_visible=True,
            hovertemplate="%{x}<br>Count: %{y:.1f}<extra></extra>",
        )
        if show_points:
            trace_kwargs["points"] = "all"
            trace_kwargs["jitter"] = 0.2
            trace_kwargs["pointpos"] = 0
            trace_kwargs["marker"] = dict(
                color="white",
                line=dict(color="black", width=1),
                size=8,
                opacity=0.9,
            )
        else:
            trace_kwargs["points"] = False
        fig.add_trace(go.Violin(**trace_kwargs))

    layout = dict(
        xaxis_title=xaxis_title,
        yaxis_title=y_title,
        width=width,
        height=height,
        violinmode="group",
        legend=dict(orientation="h", yanchor="top", y=-0.3,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    traceorder="normal",
                    ),
        margin=dict(l=120, b=130),
    )
    fig.update_layout(**layout)
    fig.update_layout(**kwargs)
    fig.update_xaxes(showline=True, linecolor="black", ticks="outside")
    fig.update_yaxes(showline=True, linecolor="black", ticks="outside")

    if show_counts:
        fig.update_xaxes(
            ticktext=[f"{g}<br>(n={group_counts[g]})" for g in groups],
            tickvals=groups,
        )

    if log:
        fig.update_yaxes(type="log")

    if save:
        _save_figure(fig, save, "copy_number_violin")

    return fig


def copy_number_stacked_violin(
    adata: "ad.AnnData",
    *,
    group_by: str | None = None,
    group_order: Sequence[str] | None = None,
    motifs: str | Sequence[str] | None = None,
    log: bool = False,
    show_box: bool = False,
    show_points: bool = False,
    show_counts: bool = True,
    colormap: str | Sequence[str] | None = None,
    row_height: int = 50,
    figsize: tuple[int | None, int | None] = (None, None),
    save: str | bool | None = None,
    **kwargs,
) -> "go.Figure":
    """
    Plot copy-number distributions for multiple motifs as stacked violins.

    Each row corresponds to one motif; each column corresponds to a group
    defined by ``group_by``.  Useful for comparing copy-number variation
    across motifs and sample groups simultaneously.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with motif copy-number matrix in ``X``.
    group_by : str | None
        Column name in ``adata.obs`` used to group samples. If it is None, plot without grouping
    group_order : Sequence[str] | None, default=None
        Explicit order for the groups on the x-axis.  If ``None``, groups
        are sorted alphabetically.
    motifs : str | Sequence[str] | None, default=None
        Motif(s) to visualise.  If ``None``, all motifs in ``adata`` are
        used.  If more than 30 motifs are selected a warning is emitted.
        A single ``str`` or a list/sequence of motif IDs / sequences is
        accepted.
    log : bool, default=False
        Whether to apply ``log1p`` transform to copy-number values before
        plotting.
    show_box : bool, default=True
        Whether to overlay a mini box plot inside each violin.
    show_points : bool, default=False
        Whether to overlay individual data points on each violin.
    show_counts : bool, default=False
        If ``True``, the x-axis tick label of each group shows the sample
        count on a second line (``<group><br>n={count}``).
    colormap : str | Sequence[str] | None, default=None
        Colormap for the median-based violin fill.  If a ``str``, it is
        passed to ``plotly.colors.sample_colorscale`` (e.g. ``"Viridis"``,
        ``"Plasma"``).  If a sequence of hex/rgb strings, used directly.
        If ``None``, the module default sequential colormap is used.
    row_height : int, default=80
        Height in pixels allocated to each motif row.
    figsize : tuple[int | None, int | None] | None, default=(None, None)
        Figure size in pixels.
    save : str | bool | None, default=None
        If ``True`` or a ``str``, save the figure. A string is appended to the default filename.
        Infer the filetype if ending on {``'.pdf'``, ``'.png'``, ``'.svg'``}.
    **kwargs
        Additional arguments passed to ``fig.update_layout()``.

    Returns
    -------
    go.Figure
        Plotly figure with stacked violin plots.

    Examples
    --------
    >>> import vampire as vp
    >>> vp.anno.pl.set_default_plotstyle()
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.pl.copy_number_stacked_violin(adata, group_by="haplotype")
    >>> vp.anno.pl.copy_number_stacked_violin(
    ...     adata, group_by="ancestry", motifs=["0", "1", "2"]
    ... )
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    if group_by is not None and group_by not in adata.obs.columns:
        raise KeyError(f"group_by column '{group_by}' not found in adata.obs")

    # Resolve motif list
    if motifs is None:
        motif_list = list(adata.var_names)
    elif isinstance(motifs, str):
        motif_list = [motifs]
    else:
        motif_list = list(motifs)

    if len(motif_list) == 0:
        raise ValueError("No motifs selected for plotting.")
    if len(motif_list) > 30:
        logger.warning(
            "Plotting %d motifs; consider passing a smaller subset via "
            "the `motifs` argument for clarity.",
            len(motif_list),
        )

    # Resolve motif indices
    motif_indices: list[int] = []
    motif_labels: list[str] = []
    for m in motif_list:
        if isinstance(m, int):
            if m < 0 or m >= adata.n_vars:
                raise IndexError(f"motif index {m} out of range")
            idx = m
        elif isinstance(m, str):
            if m in adata.var.index:
                idx = adata.var.index.get_loc(m)
            elif "motif" in adata.var.columns and m in adata.var["motif"].values:
                idx = adata.var["motif"].tolist().index(m)
            else:
                raise KeyError(
                    f"motif '{m}' not found in adata.var.index or adata.var['motif']"
                )
        else:
            raise TypeError(f"motif entries must be str or int, got {type(m).__name__}")
        motif_indices.append(idx)
        motif_labels.append(str(m))

    # Resolve group order
    if group_by is None:
        groups = ["All"]
        group_labels = np.array(["All"] * adata.n_obs)
    else:
        group_labels = adata.obs[group_by].astype(str).to_numpy()
        all_groups = set(adata.obs[group_by].dropna().unique())

        if group_order is not None:
            groups = list(group_order)
            missing = set(all_groups) - set(groups)
            if missing:
                raise ValueError(
                    f"group_order contains unknown groups: {missing}"
                )
        else:
            groups = sorted(all_groups, key=str)

        if len(groups) == 0:
            raise ValueError(
                f"No valid groups found in adata.obs['{group_by}']"
            )

    n_motifs = len(motif_indices)
    n_groups = len(groups)

    # Pre-compute per-group sample counts
    group_counts = {}
    for group in groups:
        if group_by is None:
            group_counts[group] = adata.n_obs
        else:
            group_counts[group] = int((group_labels == str(group)).sum())

    font_size = kwargs.get("font", {}).get("size")
    if font_size is None:
        font_size = _sizing.get_active_font_size()

    width, height = _sizing.resolve_figsize(
        figsize[0],
        figsize[1],
        calc_width=lambda: _sizing.stacked_violin_width(n_groups, font_size),
        calc_height=lambda: _sizing.stacked_violin_height(n_motifs, row_height, font_size),
    )

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    if log:
        X = np.log1p(X)

    # Resolve colours
    if colormap is None:
        color_palette = ["rgb(255, 255, 255)", "rgb(178, 34, 34)"]
    elif isinstance(colormap, str):
        try:
            from plotly.colors import sample_colorscale
            color_palette = sample_colorscale(colormap, np.linspace(0, 1, 256))
        except Exception:
            color_palette = _SEQUENTIAL_COLORMAP
    else:
        color_palette = list(colormap)
    n_colors = len(color_palette)
    plotly_colormap = [[i / (n_colors - 1), color_palette[i]] for i in range(n_colors)]

    # Compute per-motif, per-group medians for colour mapping
    medians = np.zeros((n_motifs, n_groups), dtype=float)
    for row_idx, idx in enumerate(motif_indices):
        y_all = X[:, idx]
        for col_idx, group in enumerate(groups):
            if group_by is None:
                mask = np.ones(adata.n_obs, dtype=bool)
            else:
                mask = group_labels == str(group)
            medians[row_idx, col_idx] = np.median(y_all[mask])

    # Normalise medians row-wise (per motif)
    median_norm = np.zeros_like(medians)
    for row_idx in range(n_motifs):
        vmin, vmax = medians[row_idx].min(), medians[row_idx].max()
        if vmax > vmin:
            median_norm[row_idx] = (medians[row_idx] - vmin) / (vmax - vmin)
        else:
            # use darkest color if all values identical
            median_norm[row_idx] = 1.0

    fig = make_subplots(
        rows=n_motifs,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
    )

    # Add left-side motif labels aligned to each subplot's vertical centre
    for i, label in enumerate(motif_labels, start=1):
        yaxis_name = f"yaxis{i}" if i > 1 else "yaxis"
        y_domain = fig.layout[yaxis_name].domain
        y_center = (y_domain[0] + y_domain[1]) / 2

        # horizontal tick
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=-0.018,
            x1=-0.002,
            y0=y_center,
            y1=y_center,
            line=dict(width=_sizing.get_active_line_width()),
        )

        # label
        fig.add_annotation(
            x=-0.02,
            y=y_center,
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=_sizing.get_active_font_size()),
            xanchor="right",
            yanchor="middle",
            textangle=0,
        )

    for row_idx, idx in enumerate(motif_indices, start=1):
        y_all = X[:, idx]
        for col_idx, group in enumerate(groups):
            if group_by is None:
                mask = np.ones(adata.n_obs, dtype=bool)
            else:
                mask = group_labels == str(group)
            group_y = y_all[mask]
            x_vals = [str(group)] * len(group_y)
            ratio = median_norm[row_idx - 1, col_idx]
            color_idx = int(np.round(ratio * (n_colors - 1)))
            color = color_palette[min(color_idx, n_colors - 1)]
            trace_kwargs = dict(
                x=x_vals,
                y=group_y,
                name=str(group),
                showlegend=False,
                fillcolor=color,
                line=dict(color="black", width=1),
                opacity=0.8,
                width=0.7,
                spanmode="hard",
                side="both",
                box_visible=show_box,
                box=dict(fillcolor="white", line=dict(color="black", width=1.2)),
                meanline_visible=True,
                hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
            )
            if show_points:
                trace_kwargs["points"] = "all"
                trace_kwargs["jitter"] = 0.2
                trace_kwargs["pointpos"] = 0
                trace_kwargs["marker"] = dict(
                    color="white",
                    line=dict(color="black", width=1),
                    size=8,
                    opacity=0.9,
                )
            else:
                trace_kwargs["points"] = False
            fig.add_trace(go.Violin(**trace_kwargs), row=row_idx, col=1)

    # Strip all axes (ticks + lines) globally, then restore bottom x-axis only
    fig.update_xaxes(showticklabels=False, showline=False, zeroline=False, ticks="")
    fig.update_yaxes(showticklabels=False, showline=False, zeroline=False, ticks="")
    fig.update_xaxes(
        showticklabels=True,
        showline=True,
        linecolor="black",
        ticks="outside",
        row=n_motifs,
        col=1,
    )

    if show_counts:
        fig.update_xaxes(
            ticktext=[f"{g}<br>(n={group_counts[g]})" for g in groups],
            tickvals=groups,
            row=n_motifs,
            col=1,
        )

    # Global bounding box around the whole plot area
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        xref="paper",
        yref="paper",
        line=dict(color="black", width=_sizing.get_active_line_width()),
        fillcolor="rgba(0,0,0,0)",
        layer="above",
    )

    # Colour-bar legend for the median-based gradient
    # Dynamic colorbar y so its top edge stays ~30 px below the plot area
    # regardless of total figure height.
    plot_height = max(height - 60 - 100, 1)
    cb_y = - 70 / plot_height
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale=plotly_colormap,
            showscale=True,
            cmin=0,
            cmax=1,
            colorbar=dict(
                orientation="h",
                title=dict(text="Normalized median CN", side="bottom"),
                thickness=15,
                len=0.5,
                xanchor="center",
                x=0.5,
                y=cb_y,
                yanchor="top",
            ),
        ),
        showlegend=False,
        hoverinfo="skip",
    ))

    layout = dict(
        width=width,
        height=height,
        violinmode="group",
        margin=dict(b=100, t=60, l=120),
    )
    fig.update_layout(**layout, **kwargs)

    if save:
        _save_figure(fig, save, "copy_number_stacked_violin")

    return fig