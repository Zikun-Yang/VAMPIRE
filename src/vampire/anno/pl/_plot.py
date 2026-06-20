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
from ..pp._markdup import markdup


"""
#
# tracksplot function
#
"""
def tracksplot(
    tracks: list,
    region: str,
    title: str = "",
    x_title: str = "Position (bp)",
    vertical_spacing: float = 0.02,
    base_width: int = 600,
    track_name_dx: float = -0.07,
    figsize: tuple[int | None, int | None] | None = (None, None),
    save: str | bool | None = None,
    **kwargs
) -> go.Figure:
    """
    Create a multi-track genomic plot with shared x-axis.

    Each track gets its own subplot with independent y-axis, but all tracks share
    the same x-axis (genomic position). Supported track types include bedgraph,
    bed, and heatmap.

    Parameters
    ----------
    tracks : list[dict]
        list of track configuration dictionaries. Each dictionary should contain:

        Common fields for all track types:
        
        - **name** (`str`) - Track name displayed on the left side
        - **type** (`str`) - Track type, one of `"bedgraph"`, `"bed"`, or `"heatmap"`
        - **data** (`pl.DataFrame | pl.LazyFrame`) - Polars DataFrame with track data
        - **height** (`float`, optional) - Relative height of the track. Default is 1.0
        - **showlegend** (`bool`, optional) - Whether to show the legend. Default is False

        Additional options for `"bedgraph"` tracks:

        - **plot_type** (`str`, optional) - `"line"`, `"bar"` or `"density"`. Default is `"line"`
        - **max_value** (`float`, optional) - Maximum value. Default is the maximum in the data
        - **min_value** (`float`, optional) - Minimum value. Default is the minimum in the data
        - **linewidth** (`float`, optional) - Line width. Default is 1
        - **color** (`str`, optional) - Line or bar color. Default is `"#212529"`
        - **colorscale** (`list[str]` or `list[tuple[float, str]]`, optional) - Colorscale for density plot

        Additional options for `"bed"` tracks:

        - **stranded** (`bool`, optional) - Whether to show stranded arrows. Default is False
        - **arrowhead_length** (`float`, optional) - Arrowhead length compared with the region length for stranded arrows. Default is 0.03
        - **color** (`str`, optional) - color. Default is `"#212529"`
        - **draw_baseline** (`bool`, optional) - When ``True``, draws a thin black
          horizontal line across the full region before the rectangles, so gaps
          appear as breaks. Default is ``False``.

        Additional options for `"heatmap"` tracks:

        - **max_value** (`float`, optional) - Maximum value. Default is the maximum in the data
        - **min_value** (`float`, optional) - Minimum value. Default is the minimum in the data
        - **colorscale** (`list[str]` or `list[tuple[float, str]]`, optional) - Colorscale for heatmap
        - **flip_y** (`bool`, optional) - Whether to flip the y-axis. Default is False
    
    region : str
        Genomic region in the format "chrom:start-end" (e.g., "chr1:1000-2000").
    
    title : str, optional
        Title of the figure. Default is an empty string.

    x_title : str, optional
        Title of the x axis. Default is `"Position (bp)"`.
    
    vertical_spacing : float, optional
        Vertical spacing between subplots as a fraction of total height.
        Default is 0.02.

    base_width : int, default is 600
        The plotting width of the whole figure.

    track_name_dx : float, optional
        Horizontal offset applied to track name position along the x-axis,expressed as a fraction of the total width.
        Default is -0.07.
    
    figsize : tuple[int | None, int | None], optional
        Figure size in pixels. Default is (None, None).

    save: str | bool | None
        If True or a str, save the figure. A string is appended to the default filename.
        Infer the filetype if ending on {'.pdf', '.png', '.svg'}.
    
    **kwargs
        Additional keyword arguments passed to Plotly `update_layout`.
        Used to control figure-level styling (e.g. template, margin,
        background color, legend settings).

    Returns
    -------
    go.Figure
        A Plotly figure object with all tracks plotted as subplots.

    Examples
    --------
    >>> import vampire as vp
    >>> vp.anno.pl.set_default_plotstyle()
    >>> tracks = vp.datasets.chm13_cen1_tracks()
    >>> vp.anno.pl.tracksplot(
    ...     tracks,
    ...     region = "chm13_chr1:121119216-127324115",
    ...     title = "chm13_chr1:121119216-127324115",
    ...     vertical_spacing = 0.02,
    ...     track_name_dx = -0.08,
    ...     base_width = 400, # optional; adjust figure width to fit within the manual page width
    ... )
    """
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go
    import plotly.subplots as sp

    # get coordinates
    region = region.split(":")
    CHROM = region[0]
    START = int(region[1].split("-")[0])
    END = int(region[1].split("-")[1])

    # create subplots: one row per track, shared x-axis
    n_tracks = len(tracks)
    if n_tracks == 0:
        return go.Figure()

    # get real figure size
    MIN_MARGIN: int = 40
    MAX_NAME_LENGTH: int = max(len(track["name"]) for track in tracks)
    HAVE_HEATMAP: bool = any(track["type"] == "heatmap" for track in tracks)

    # Resolve figsize with auto-sizing
    font_size = kwargs.get("font", {}).get("size")
    if font_size is None:
        font_size = _sizing.get_active_font_size()

    _width, _height = _sizing.resolve_figsize(
        figsize[0] if figsize is not None else None,
        figsize[1] if figsize is not None else None,
        calc_width=lambda: _sizing.tracksplot_width(
            tracks, font_size, max_name_length=MAX_NAME_LENGTH, base_width=base_width
        ),
        calc_height=lambda: _sizing.tracksplot_height(
            tracks, font_size, vertical_spacing=vertical_spacing,
            max_name_length=MAX_NAME_LENGTH, base_width=base_width
        ),
    )
    figsize = (_width, _height)

    left_margin: int = _sizing.tracksplot_left_margin(
        tracks, font_size, max_name_length=MAX_NAME_LENGTH
    )
    real_height: float = figsize[1] - MIN_MARGIN * 2
    real_width: float = figsize[0] - left_margin - MIN_MARGIN
    heights, _ = _sizing.tracksplot_subplot_heights(
        tracks, real_width=real_width, real_height=real_height,
        vertical_spacing=vertical_spacing, font_size=font_size
    )
    
    # set subplot titles and heights
    fig = sp.make_subplots(
        rows = n_tracks,
        cols = 1,
        shared_xaxes = True,  # share x-axis across all subplots
        vertical_spacing = vertical_spacing,  # spacing between subplots
        row_heights = heights,
    )

    track_idx = 0
    for track in tracks:
        # get data
        name: str = track["name"]
        type: str = track["type"]
        data: pl.DataFrame|pl.LazyFrame = track["data"]
        match type:
            case "heatmap":
                data = data.filter((pl.col("chrom1") == CHROM) & 
                                (pl.col("chrom2") == CHROM) & 
                                (pl.col("end1") >= START) & 
                                (pl.col("start1") <= END) &
                                (pl.col("end2") >= START) & 
                                (pl.col("start2") <= END))
            case _:
                data = data.filter((pl.col("chrom") == CHROM) & 
                                (pl.col("end") >= START) & 
                                (pl.col("start") <= END))
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        logger.debug(
            "Track %d/%d: '%s' (%s) — %d items after region filter",
            track_idx + 1,
            n_tracks,
            name,
            type,
            data.height,
        )

        match type:
            case "bedgraph":
                plot_type = track.get("plot_type", "line")
                if plot_type == "line":
                    _plot_bedgraph_track_line(fig, track, data, track_idx + 1)
                elif plot_type == "bar":
                    _plot_bedgraph_track_bar(fig, track, data, track_idx + 1)
                elif plot_type == "density":
                    _plot_bedgraph_track_density(fig, track, data, track_idx + 1)
                else:
                    raise ValueError(f"Invalid plot type: {plot_type}")
            
            case "bed":
                _plot_bed_track(fig, track, data, track_idx + 1, (CHROM, START, END))
            
            case "heatmap":
                _plot_heatmap_track(fig, track, data, track_idx + 1)
                
            case _:
                raise ValueError(f"Cannot identify the track type from the columns: {track.columns}")
        track_idx += 1
    
    # update layout: only show x-axis ticks/labels on bottom subplot
    if n_tracks > 1:
        for row_idx in range(1, n_tracks):
            fig.update_xaxes(
                showticklabels = False,
                ticks = "",
                row = row_idx,
                col = 1,
            )
    fig.update_xaxes(
        range = [START, END],
        title_text = x_title,
        title_standoff = 10,
        showline = True,          # show x-axis line
        linecolor = "black",      # axis line color
        linewidth = _sizing.get_active_line_width(),          # axis line width
        tickwidth = _sizing.get_active_line_width(),
        row = n_tracks,
        col = 1,
    )
        
    # set figure size
    fig.update_layout(
        width = figsize[0],
        height = figsize[1],
        margin = dict(l=left_margin, r=MIN_MARGIN, t=MIN_MARGIN, b=MIN_MARGIN + 10 + 5),
        autosize = False
    )

    # set figure title
    left_margin = fig.layout.margin.l
    right_margin = fig.layout.margin.r
    width = fig.layout.width
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=font_size + 2),
            x=(width + left_margin - right_margin) / (2 * width),
            xanchor="center",
        )
    )

    fig.update_layout(
        **kwargs
    )

    # add annotations on the left side
    for idx, track in enumerate(tracks):
        y_domain = fig.layout[f"yaxis{idx+1}"].domain
        y_center = (y_domain[0] + y_domain[1]) / 2
        fig.add_annotation(
            text = track["name"],
            xref = "paper",
            yref = "paper",
            x = track_name_dx,
            y = y_center,
            xanchor = "right",
            yanchor = "middle",
            showarrow = False
        )

    if save:
        _save_figure(fig, save, "tracksplot")
    
    return fig

def _plot_bedgraph_track_line(
    fig: go.Figure,
    track: dict,
    data: pl.DataFrame,
    row: int
) -> None:
    """
    Plot a bedgraph track as a line plot.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure object to add the bedgraph track to.
    track : dict
        Track configuration dictionary containing plot settings (e.g., name, color,
        max_value, min_value).
    data : pl.DataFrame
        Polars DataFrame containing bedgraph data with columns: chrom, start, end,
        and value.
    row : int
        Subplot row number (1-indexed) where the bedgraph should be added.

    Returns
    -------
    None
        The function modifies the figure in-place.
    """
    import plotly.graph_objects as go

    ymax = track.get("max_value", data["value"].max())
    ymin = track.get("min_value", data["value"].min())

    # build line plot from bedgraph data
    x_coords = []
    y_coords = []
    
    # sort by start position
    data_sorted = data.sort("start")
    
    for row_data in data_sorted.iter_rows(named = True):
        x_coords.extend([row_data["start"], row_data["end"]])
        y_coords.extend([row_data["value"], row_data["value"]])
    
    # add trace with lines
    fig.add_trace(
        go.Scatter(
            x = x_coords,
            y = y_coords,
            mode = 'lines',
            name = track["name"],
            line = dict(
                width=track.get("linewidth", 1),
                color=track.get("color", "#212529"),
                shape='hv',
            ),  # step plot (horizontal-vertical)
            showlegend = track.get("showlegend", False),
            legendgroup=track["name"],
            legendgrouptitle_text=track["name"]
        ),
        row = row,
        col = 1
    )
    fig.update_yaxes(
        range = [ymin, ymax],
        showline = True,          # show x-axis line
        linecolor = "black",      # axis line color
        linewidth = 1.4,
        tickwidth = 1.4,
        row = row,
        col = 1
    )
    fig.update_xaxes(
        showline = True,          # show y-axis line
        linecolor = "black",      # axis line color
        linewidth = 1.4,
        row = row,
        col = 1
    )

def _plot_bedgraph_track_bar(
    fig: go.Figure,
    track: dict,
    data: pl.DataFrame,
    row: int
) -> None:
    """
    Plot a bedgraph track as a bar plot.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure object to add the bedgraph track to.
    track : dict
        Track configuration dictionary containing plot settings (e.g., name, color,
        max_value, min_value).
    data : pl.DataFrame
        Polars DataFrame containing bedgraph data with columns: chrom, start, end,
        and value.
    row : int
        Subplot row number (1-indexed) where the bedgraph should be added.

    Returns
    -------
    None
        The function modifies the figure in-place.
    """
    import plotly.graph_objects as go

    ymax = track.get("max_value", data["value"].max())
    ymin = track.get("min_value", data["value"].min())

    # sort by start position
    data_sorted = data.sort("start")

    # add trace with bars
    fig.add_trace(
        go.Bar(
            x = data_sorted["start"].to_list(),
            y = data_sorted["value"].to_list(),
            marker = dict(
                color=track.get("color", "#212529"),
            ),
            name = track["name"],
            showlegend = track.get("showlegend", False),
            legendgroup=track["name"],
            legendgrouptitle_text=track["name"]
        ),
        row = row,
        col = 1
    )
    fig.update_yaxes(
        range = [ymin, ymax],
        showline = True,          # show x-axis line
        linecolor = "black",      # axis line color
        linewidth = 1.4,
        tickwidth = 1.4,
        row = row,
        col = 1
    )
    fig.update_xaxes(
        showline = True,          # show y-axis line
        linecolor = "black",      # axis line color
        linewidth = 1.4,
        row = row,
        col = 1
    )

def _plot_bedgraph_track_density(
    fig: go.Figure,
    track: dict,
    data: pl.DataFrame,
    row: int
) -> None:
    """

    """
    import plotly.graph_objects as go

    if data.height == 0:
        return

    # determine value range
    ymax = track.get("max_value", data["value"].max())
    ymin = track.get("min_value", data["value"].min())

    data = data.filter((data["value"] >= ymin) & (data["value"] <= ymax))
    data_sorted = data.sort("start")

    if data_sorted.height == 0:
        return

    # x use the interval start
    x_vals = data_sorted["start"].to_list()

    # 1D heatmap
    z_vals = [data_sorted["value"].to_list()]

    all_columns = data_sorted.columns

    DEFAULT_COLORMAP: list[str] = ["#5E4FA2", "#3288BD", "#66C2A5", "#ABDDA4", "#E6F598", "#FFFFBF", "#FEE08B", "#FDAE61", "#F46D43", "#D53E4F", "#9E0142"]
    colorscale = track.get("colorscale", DEFAULT_COLORMAP)
    if all(isinstance(x, str) for x in colorscale):
        values = data_sorted["value"].to_numpy()
        colorscale = _get_colorscale(values, colorscale)
    elif all(_is_float_str_tuple(x) for x in colorscale):
        # nothing to do
        pass
    else:
        raise ValueError(f"""
        Invalid colorscale: {colorscale}, give a list of colors or a list of tuples with breaks and colors\n
        Example: ['#5E4FA2', '#3288BD', '#66C2A5']\n
        Example: [(0, '#5E4FA2'), (0.5, '#3288BD'), (1, '#66C2A5')]
        """)

    # customdata must be 2D
    customdata: list[list[list[Any]]] = [
        [
            [row_data[col] for col in all_columns]
            for row_data in data_sorted.iter_rows(named=True)
        ]
    ]

    fig.add_trace(
        go.Heatmap(
            x = x_vals,
            y = [0],                     # single row
            z = z_vals,
            colorscale = colorscale,
            zmin = ymin,
            zmax = ymax,
            showscale = track.get("showlegend", False),
            customdata = customdata,
            meta = all_columns,
            hovertemplate = (
                "<br>".join(
                    f"{col}: %{{customdata[{i}]}}"
                    for i, col in enumerate(all_columns)
                )
                + "<extra></extra>"
            ),
            colorbar = dict(
                title=track["name"],
                orientation="h",   # horizontal
                x=0.5,
                xanchor="center",
                y=-0.1 - 0.1 * row,
                len=1
            ),
            legendgroup=track["name"],
            legendgrouptitle_text=track["name"]
        ),
        row = row,
        col = 1
    )

    fig.update_yaxes(
        showticklabels = False,
        ticks = "",
        row = row,
        col = 1
    )

def _plot_bed_track(
    fig: go.Figure,
    track: dict,
    data: pl.DataFrame,
    row: int,
    region: tuple[str, int, int]
) -> None:
    """
    Plot a bed track as rectangles or arrows.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure object to add the bed track to.
    track : dict
        Track configuration dictionary containing plot settings (e.g., name,
        stranded). If stranded is True, arrows are drawn; otherwise rectangles.

        - **draw_baseline** (`bool`, optional) — When ``True``, a thin black
          horizontal line is drawn across the full region *before* the
          rectangles, so gaps (positions with no data) appear as breaks in the
          line. Default is ``False``.
    data : pl.DataFrame
        Polars DataFrame containing bed data with columns: chrom, start, end, and
        optionally itemRgb (for colors) and strand (if stranded is True).
    row : int
        Subplot row number (1-indexed) where the bed track should be added.
    region : tuple[str, int, int]
        Genomic region in the format (chrom, start, end).
    
    Returns
    -------
    None
        The function modifies the figure in-place.
    """
    import numpy as np
    import plotly.graph_objects as go

    CHROM, START, END = region
    data_sorted = data.sort("start")

    # Background baseline — drawn as a shape with layer="below" so it is
    # guaranteed to sit underneath all traces regardless of trace type.
    if track.get("draw_baseline", False):
        fig.add_shape(
            type="line",
            x0=START, y0=0.5, x1=END, y1=0.5,
            line=dict(color="black", width=1),
            layer="below",
            row=row, col=1,
        )

    # prepare data for batch plotting
    bases = []  # x starting positions
    widths = []  # widths (end - start)
    colors = []  # colors for each rectangle
    y_positions = []  # y positions (all same for one track)
    custom_data_list = []  # custom data for each rectangle
    
    # rectangle height
    rect_height = 1.0
    y_center = 0.5  # center at 0.5 for each subplot
    
    # check if color column exists
    has_itemRgb = "itemRgb" in data.columns
    
    # get all column names
    all_columns = data.columns
    
    for row_data in data_sorted.iter_rows(named=True):
        bases.append(row_data["start"])
        widths.append(row_data["end"] - row_data["start"])
        y_positions.append(y_center)
        
        # determine color
        color = track.get("color", "#212529")
        if has_itemRgb and "color" not in track.keys(): # if color is specified in the track, use it
            item_rgb = row_data.get("itemRgb")
            if item_rgb and item_rgb != ".":
                # format: r,g,b
                if ',' in item_rgb:
                    rgb = item_rgb.split(",")
                    if len(rgb) == 3:
                        color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
                elif item_rgb.startswith("#"):
                    color = item_rgb
        colors.append(color)
        
        # prepare custom data
        custom_data_list.append([row_data.get(col) for col in all_columns])
    
    # add single trace with all rectangles (batch plotting for efficiency)
    if bases:  # only add if there are rectangles
        if not track.get("stranded", False):
            fig.add_trace(
                go.Bar(
                    x = widths,  # bar lengths (widths)
                    y = y_positions,  # y positions
                    base = bases,  # x starting positions
                    orientation = 'h',  # horizontal bars
                    marker = dict(
                        color=colors
                    ),
                    name = track["name"],
                    width = rect_height,  # height of bars in y direction
                    showlegend = track.get("showlegend", False),
                    customdata = custom_data_list,  # add custom data for hover and interaction
                    meta = all_columns,  # store column names for reference
                    hovertemplate = (
                        "<br>".join(
                            f"{col}: %{{customdata[{i}]}}"
                            for i, col in enumerate(all_columns)
                        )
                        + "<extra></extra>"
                    ),
                    legendgroup=track["name"],
                    legendgrouptitle_text=track["name"]
                ),
                row = row,
                col = 1
            )
        else:
            ARROWHEAD_RATIO: float = track.get("arrowhead_length", 0.03)
            BODY_WIDTH: float = 0.4
            ARROWHEAD_LENGTH: float = ARROWHEAD_RATIO * (END - START)
            for idx, row_data in enumerate(data_sorted.iter_rows(named=True)):
                dx = row_data["end"] - row_data["start"] if row_data["strand"] == "+" else row_data["start"] - row_data["end"]
                pts = _make_solid_arrow(row_data["start"], y_center, dx, body_width=BODY_WIDTH, arrowhead_length=ARROWHEAD_LENGTH)
                cd = np.tile(custom_data_list[idx], (len(pts), 1)).tolist()
                fig.add_trace(
                    go.Scatter(
                        x = pts[:,0],
                        y = pts[:,1],
                        fill = 'toself',
                        line = dict(
                            width=1,
                            color="white",
                        ),
                        mode='lines',
                        fillcolor = colors[idx],
                        hoverinfo="skip",
                        showlegend = track.get("showlegend", False),
                        legendgroup=track["name"],
                        legendgrouptitle_text=track["name"]
                    ),
                    row = row,
                    col = 1
                )
                fig.add_trace(
                    go.Bar(
                        x = [widths[idx]],
                        y = [y_positions[idx]],
                        base = [bases[idx]],
                        marker = dict(
                            color = colors[idx]
                        ),
                        orientation = 'h',
                        opacity=0, # pseudo-trace to show hover
                        showlegend = False,
                        customdata = cd,
                        meta = all_columns,
                        hovertemplate = (
                            "<br>".join(
                                f"{col}: %{{customdata[{i}]}}"
                                for i, col in enumerate(all_columns)
                            )
                            + "<extra></extra>"
                        )
                    ),
                    row = row,
                    col = 1
                )
        
        # set y-axis
        fig.update_yaxes(
            range = [0, 1],
            showticklabels = False,
            ticks = "",
            row = row,
            col = 1
        )

def _plot_heatmap_track(
    fig: go.Figure,
    track: dict,
    data: pl.DataFrame,
    row: int
) -> None:
    """
    Plot a triangular heatmap track on the figure.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure object to add the heatmap to.
    track : dict
        Track configuration dictionary containing plot settings (e.g., max_value,
        min_value, colorscale, flip_y).
    data : pl.DataFrame
        Polars DataFrame containing heatmap data with columns: chrom1, start1, end1,
        chrom2, start2, end2, and value.
    row : int
        Subplot row number (1-indexed) where the heatmap should be added.

    Returns
    -------
    None
        The function modifies the figure in-place.
    """
    import plotly.graph_objects as go

    # compute zmin/zmax
    zmax = track.get("max_value", data["value"].max())
    zmin = track.get("min_value", data["value"].min())
    data = data.filter((data["value"] >= zmin) & (data["value"] <= zmax))
    if data.height == 0:
        return

    # prepare colorscale
    DEFAULT_COLORMAP: list[str] = ["#5E4FA2", "#3288BD", "#66C2A5", "#ABDDA4", "#E6F598", "#FFFFBF", "#FEE08B", "#FDAE61", "#F46D43", "#D53E4F", "#9E0142"]
    colorscale = track.get("colorscale", DEFAULT_COLORMAP)
    if all(isinstance(x, str) for x in colorscale):
        # assign breaks to colors
        values = data["value"].to_numpy()
        colorscale = _get_colorscale(values, colorscale)
    elif all(_is_float_str_tuple(x) for x in colorscale):
        # nothing to do
        pass
    else:
        raise ValueError(f"""
            Invalid colorscale: {colorscale}, give a list of colors or a list of tuples with breaks and colors\n
            Example: ['#5E4FA2', '#3288BD', '#66C2A5']\n
            Example: [(0, '#5E4FA2'), (0.5, '#3288BD'), (1, '#66C2A5')]
        """)

    # build bins
    bins = (
        pl.concat([
            data.select([pl.col("start1").alias("start"), pl.col("end1").alias("end")]),
            data.select([pl.col("start2").alias("start"), pl.col("end2").alias("end")])
        ])
        .unique()
        .sort("start")
        .with_row_index("idx")
    )

    bin_starts = bins["start"].to_numpy()
    bin_ends = bins["end"].to_numpy()
    bin_centers = (bin_starts + bin_ends) / 2

    # join i/j
    data = (
        data
        .join(bins.rename({"start": "start1", "idx": "i"}), on="start1", how="left")
        .join(bins.rename({"start": "start2", "idx": "j"}), on="start2", how="left")
    )

    i = data["i"].to_numpy()
    j = data["j"].to_numpy()
    v = data["value"].to_numpy()

    x1 = bin_centers[i]
    x2 = bin_centers[j]

    # compute rotated triangle coordinates
    xp = (x1 + x2) / 2
    yp = (x2 - x1) / 2

    mask = yp >= 0
    xp = xp[mask]
    yp = yp[mask]
    v = v[mask]

    if len(xp) == 0:
        return None, None, None

    # compute resolution
    resolution = int(np.median(np.diff(np.sort(bin_centers))))
    if resolution <= 0:
        resolution = 1

    # build grid
    x_min, x_max = xp.min(), xp.max()
    y_min, y_max = 0, yp.max()
    x_bins = np.arange(x_min, x_max + resolution, resolution)
    y_bins = np.arange(y_min, y_max + resolution, resolution)

    z_vals = np.zeros((len(y_bins), len(x_bins)), dtype=np.float32)
    counts = np.zeros_like(z_vals, dtype=np.int32)

    # compute floor index
    x_idx = np.floor((xp - x_min) / resolution).astype(int)
    y_idx = np.floor((yp - y_min) / resolution).astype(int)

    valid = (
        (x_idx >= 0) & (x_idx < len(x_bins)) &
        (y_idx >= 0) & (y_idx < len(y_bins))
    )

    # accumulate and count
    np.add.at(z_vals, (y_idx[valid], x_idx[valid]), v[valid])
    np.add.at(counts, (y_idx[valid], x_idx[valid]), 1)

    # compute average value and NaN
    mask_nonzero = counts > 0
    z_vals[mask_nonzero] /= counts[mask_nonzero]
    
    # output frequency of values with 2 or more counts (normal situation)
    if np.any(counts[mask_nonzero] > 1):
        frequency = np.sum(counts[mask_nonzero] > 1) / len(counts[mask_nonzero])
        logger.debug(f"Some cells have 2 or more values: {frequency:.3%}")
    
    z_vals[~mask_nonzero] = np.nan

    # add trace
    fig.add_trace(
        go.Heatmap(
            x = x_bins,
            y = y_bins,
            z = z_vals,
            name = track["name"],
            colorscale = colorscale,
            zmin = zmin,
            zmax = zmax,
            showscale = track.get("showlegend", False),
            colorbar = dict(
                title=track["name"],
                orientation="h",   # horizontal
                x=0.5,
                xanchor="center",
                y=-0.1 - 0.1 * row,
                len=1
            ),
            legendgroup=track["name"],
            legendgrouptitle_text=track["name"]
        ),
        row = row,
        col = 1
    )

    fig.update_yaxes(
        range = [0, (x_max - x_min) / 2] if not track.get("flip_y", False) else [(x_max - x_min) / 2, 0],
        scaleanchor = f"x{row}", # bind to the x-axis of the current row
        constrain="domain",
        showticklabels = False,  # hide tick labels (numbers)
        ticks = "",
        row = row,
        col = 1
    )

def _make_solid_arrow(
    x0: float, 
    y0: float, 
    dx: float, 
    body_width: float,
    arrowhead_length: float
) -> np.ndarray:
    """
    Generate coordinates for a solid arrow shape.

    Parameters
    ----------
    x0 : float
        Arrow start point x-coordinate.
    y0 : float
        Arrow start point y-coordinate.
    dx : float
        Arrow vector (positive for positive strand, negative for negative strand).
    body_width : float
        Arrow body relative width.
    arrowhead_length : float
        Arrow length.
    
    Returns
    -------
    np.ndarray
        Array of shape (n_points, 2) containing the arrow point coordinates.
    """
    # define 7 points of the arrow shape (relative coordinates)
    # assume the arrow is in the positive x-axis direction, and the tail is at (0,0)
    #
    #            |\    y4 
    #   |--------| \   y3
    #   |--------| /   y2
    #            |/    y1
    #   x1(0)    x2 x3(1)
    import numpy as np

    ARROW_WIDTH: float = 1.0
    Y_CENTER: float = 0.0
    BODY_LENGTH: float = (abs(dx) - arrowhead_length) / abs(dx)

    if BODY_LENGTH <= 0:
        y1, y4 = Y_CENTER - ARROW_WIDTH / 2, Y_CENTER + ARROW_WIDTH / 2
        x1, x3 = 0, 1
        pts = np.array([
            [x1, y1],
            [x3, Y_CENTER],
            [x1, y4],
            [x1, y1]
        ])
    else:
        y1, y2, y3, y4 = Y_CENTER - ARROW_WIDTH / 2, Y_CENTER - ARROW_WIDTH * body_width / 2, Y_CENTER + ARROW_WIDTH * body_width / 2, Y_CENTER + ARROW_WIDTH / 2
        x1, x2, x3 = 0, BODY_LENGTH, 1
        pts = np.array([
            [x1, y2],
            [x2, y2],
            [x2, y1],
            [x3, Y_CENTER],
            [x2, y4],
            [x2, y3],
            [x1, y3],
            [x1, y2]
        ])

    # adjust strand
    if dx < 0:
        pts[:,0] = 1 - pts[:,0]

    # scale x coordinates to match dx, dy vector
    pts[:,0] *= abs(dx)

    pts[:,0] += x0
    pts[:,1] += y0

    return pts

def _is_float_str_tuple(x):
    """
    Check if x is a float-string tuple.

    Parameters
    ----------
    x : Any
        The value to check.

    Returns
    -------
    bool
        True if x is a float-string tuple, False otherwise.
    """
    return (
        isinstance(x, tuple)
        and len(x) == 2
        and isinstance(x[0], (float, int))  # sometimes you may give int
        and isinstance(x[1], str)
    )

def _get_colorscale(values: np.ndarray, color_list: list[str]) -> np.ndarray:
    """
    Get numeric color break boundaries for given values.

    Parameters
    ----------
    values : np.ndarray
        Array of values.
    color_list : list[str]
        list of colors.

    Returns
    -------
    list[tuple[float, str]]
        list of color scale tuples with lower/upper boundaries and colors.
    """
    import numpy as np

    values = np.asarray(values)
    ncolors = len(color_list)
    zmin = values.min()
    zmax = values.max()

    # corner case: empty
    if values.size == 0:
        return []

    # corner case: all values identical
    if np.all(values == values[0]):
        return np.array([values[0], values[0]])

    # equal-quantile breaks
    probs = np.linspace(0, 1, ncolors + 1)
    breaks = np.quantile(values, probs)

    # remove duplicate breakpoints
    breaks = np.unique(breaks)

    # still only one value after unique
    if len(breaks) == 1:
        breaks = np.array([breaks[0], breaks[0]])

    breaks = (breaks - zmin) / (zmax - zmin)
    breaks[0] = 0.0
    breaks[-1] = 1.0
    colorscale = color_list[:len(breaks)-1]
    # construct colorscale with double points
    colorscale = [
        (float(pos), color)
        for i, color in enumerate(colorscale)
        for pos in (breaks[i], breaks[i + 1])
    ]   

    return colorscale

"""
#
# waterfall function
#
"""
def waterfall(
    adata: ad.AnnData,
    feature: str = "motif",
    sample_order: list[str] | None = None,
    color: str = "id",
    colormap: dict[str, str] | list | str = "rainbow",
    deduplicate: bool = False,
    row_annotation: str | list[str] | dict[str, str] | dict[str, dict[str, str]] | None = None,
    row_annotation_colormap: (
        dict[str, str] | str |
        dict[str, dict[str, str] | str] | None
    ) = None,
    figsize: tuple[int | None, int | None] = (None, None),
    track_name_dx: float = -0.01,
    save: str | bool | None = None,
    **kwargs
) -> go.Figure:
    """
    Create a waterfall plot for motif composition across samples.

    The waterfall plot visualizes motif variation across samples in a
    stacked or ordered layout, where each sample is represented along the
    y-axis.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object generated from `pp.read_anno()`.

    feature : str, default="motif"
        Key prefix for the feature arrays stored in ``adata.uns``.
        The function looks up ``uns[f"{feature}_array"]`` and
        ``uns[f"{feature.replace('motif', 'orientation')}_array"]``.
        Common values: ``"motif"`` (raw arrays), ``"aligned_motif"``
        (alignment output from ``vp.anno.tl.sample_msa()``).

    sample_order : list of str, optional
        Ordered list of sample identifiers defining the x-axis order.
        If None, samples are ordered based on the default order in `adata.obs`.

    color : str, default="id"
        Column name in `adata.var` used to assign motif coloring.

    colormap : dict | list | str
        Color mapping for features. Default is `rainbow`.

        - dict: explicit mapping {feature -> color}
        - list: sequential color assignment following input order
        - str: use preset colormap: `rainbow`, `glasbey`, `sequential`

    deduplicate : bool, default=False
        If True, collapse samples with identical motif arrays into a single
        track. The track label shows the first sample name followed by
        ``... (n=X)`` where X is the number of collapsed samples. The draw
        order follows the position of the first occurrence in ``sample_order``.

    row_annotation : str | list[str] | dict[str, str] | dict[str, dict[str, str]] | None, optional
        Sample-level categorical annotation displayed as colored block(s)
        between the track label and the main plot.

        - ``str`` — column name in ``adata.obs`` to read categories from.
        - ``list[str]`` — list of column names in ``adata.obs``; each column
          becomes an independent annotation dimension.
        - ``dict[str, str]`` — explicit ``{sample_name -> category}``.
        - ``dict[str, dict[str, str]]`` — multiple dimensions, e.g.
          ``{"haplotype": {sample: label, ...}, "batch": {...}}``.
          Each dimension is rendered as an independent annotation column.
        - ``None`` — no annotation drawn.

    row_annotation_colormap : dict[str, str] | str | dict[str, dict[str, str] | str] | None, optional
        Color mapping for ``row_annotation`` categories.

        - Non-nested values apply to **all** dimensions.
        - Nested ``dict[str, ...]`` keys must match dimension names.
        - ``str``: preset colormap name (``"rainbow"``, ``"glasbey"``, ``"sequential"``).
        - ``None``: auto-generate from preset.

    figsize : tuple[int | None, int | None], optional
        Figure size as (width, height) in pixels. Default is (None, None).

        - (None, None): auto-compute both dimensions from data.
        - (w, None): fixed width, auto-compute height from sample count.
        - (None, h): fixed height, auto-compute width from motif/kmer count.
        - (w, h): use user-specified size.

        width is proportional to the maximum sequence length (max_x) and
        font size to prevent horizontal crowding. height is proportional
        to the number of samples (n_tracks) and font size to keep track
        labels readable and avoid vertical overlap or excessive sparsity.

    track_name_dx: float, optional
        Horizontal offset applied to track name position along the x-axis,expressed as a fraction of the total width.
        Default is -0.01.

    save: str | bool | None
        If True or a str, save the figure. A string is appended to the default filename.
        Infer the filetype if ending on {'.pdf', '.png', '.svg'}.

    **kwargs
        Additional keyword arguments passed to Plotly `update_layout`.
        Used to control figure-level styling (e.g. template, margin,
        background color, legend settings).

    Returns
    -------
    fig : go.Figure
        Plotly figure object representing the waterfall visualization.

    Examples
    --------
    >>> import vampire as vp
    >>> vp.anno.pl.set_default_plotstyle()
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.pl.waterfall(
    ...     adata,
    ...     colormap = "rainbow",
    ... )
    """
    import polars as pl
    import anndata as ad
    import plotly.graph_objects as go
    
    track_list: list[dict] = []
    max_x: int = 0

    # check sample_order
    if sample_order is None:
        sample_order: list[str] = list(adata.obs.index)
    all_sample_list: list[str] = list(adata.obs.index)
    missing = set(all_sample_list) - set(sample_order)
    if missing:
        raise KeyError(f"Missing samples in sample_order: {missing}")

    # resolve row_annotation into dict[str, dict[str, str]]
    row_annotations: dict[str, dict[str, str]] = {}
    if row_annotation is not None:
        if isinstance(row_annotation, str):
            col = row_annotation
            if col not in adata.obs.columns:
                raise ValueError(
                    f"row_annotation='{col}' not found in adata.obs.columns"
                )
            name = col
            row_annotations = {name: dict(adata.obs[col])}
        elif isinstance(row_annotation, list):
            if (
                len(row_annotation) > 0
                and all(isinstance(x, str) for x in row_annotation)
                and all(x in adata.obs.columns for x in row_annotation)
            ):
                for col in row_annotation:
                    row_annotations[col] = dict(adata.obs[col])
            else:
                raise ValueError(
                    "row_annotation as list must be a list of column names present in adata.obs.columns"
                )
        elif isinstance(row_annotation, dict):
            if not row_annotation:
                row_annotations = {}
            else:
                first_val = next(iter(row_annotation.values()))
                if isinstance(first_val, dict):
                    # multi-dimension: {dim_name: {sample: label}}
                    for dim_name, mapping in row_annotation.items():
                        missing = set(all_sample_list) - set(mapping.keys())
                        if missing:
                            raise KeyError(
                                f"row_annotation['{dim_name}'] is missing samples: {missing}"
                            )
                    row_annotations = row_annotation
                else:
                    # single-dimension: {sample: label}
                    missing = set(all_sample_list) - set(row_annotation.keys())
                    if missing:
                        raise KeyError(
                            f"row_annotation dict is missing samples: {missing}"
                        )
                    name = "annotation"
                    row_annotations = {name: row_annotation}
        else:
            raise TypeError(
                f"row_annotation must be str, list of column names, dict or None, got {type(row_annotation)}"
            )

    # build motif colormap using _get_categorical_colormap
    all_id_list = list(adata.var.index)
    if color == "id":
        id2element = {m: m for m in all_id_list}
    else:
        if color not in adata.var.columns:
            raise ValueError(
                f"color = '{color}' not found in adata.var.columns: {list(adata.var.columns)}"
            )
        id2element = dict(zip(adata.var.index, adata.var[color]))
    all_element_list = list(dict.fromkeys(id2element.values()))
    _, mapped_colormap = _get_categorical_colormap(all_element_list, colormap)

    motif_array_dict: dict[str, list[str]] = adata.uns[f"{feature}_array"]
    orientation_name: str = feature.replace("motif", "orientation")
    orientation_array_dict: dict[str, list[str]] = adata.uns[f"{orientation_name}_array"]

    # deduplicate identical motif/orientation arrays
    if deduplicate:
        if "unique_group" not in adata.obs.columns:
            logger.warning(
                "unique_group not found in adata.obs. "
                "vp.anno.pp.markdup() has not been run. Running it automatically."
            )
            adata = markdup(adata)
        
        # ensure ordering matches sample_order
        obs = adata.obs.loc[sample_order].copy()

        # deduplicate using unique_group
        seen: dict[tuple, dict] = {}
        for idx, sample in enumerate(sample_order):
            gid = obs.loc[sample, "unique_group"]

            if gid not in seen:
                seen[gid] = {
                    "idx": idx,
                    "first_sample": sample,
                    "count": 0,
                    "samples": [],
                }

            seen[gid]["samples"].append(sample)
            seen[gid]["count"] += 1

        _ordered = sorted(seen.values(), key=lambda x: x["idx"])

        draw_items = [
            (x["first_sample"], x["first_sample"], x["count"], x["samples"])
            for x in _ordered
        ]
    else:
        draw_items = [(s, s, 1, [s]) for s in sample_order]

    for sample, first_sample, count, samples in draw_items:
        # get data
        motif_array: list[str] = motif_array_dict[sample]
        orientation_array: list[str] = orientation_array_dict[sample]
        array_len: int = len(motif_array)

        # skip gaps ("-") but keep original positions so alignment is preserved
        start_array: list[float] = []
        end_array: list[float] = []
        motif_filtered: list[str] = []
        ori_filtered: list[str] = []
        color_filtered: list[str] = []

        block_cn_list = adata.uns.get("block_copy_number", {}).get(sample, [])
        if not isinstance(block_cn_list, list):
            block_cn_list = list(block_cn_list)
        cn_idx: int = 0
        for pos, (m, o) in enumerate(zip(motif_array, orientation_array)):
            if m == "-":
                continue
            start_array.append(float(pos))
            if cn_idx < len(block_cn_list):
                end_array.append(float(pos + block_cn_list[cn_idx]))
            else:
                end_array.append(float(pos + 1))
            motif_filtered.append(m)
            ori_filtered.append(o)
            color_filtered.append(mapped_colormap[m])
            cn_idx += 1

        # Old behavior: only the last block could be fractional. Kept as a
        # fallback when per-block copy numbers are unavailable.
        if end_array and not block_cn_list:
            total_cn: float = adata.obs.loc[adata.obs.index == sample, "copy_number"].iloc[0]
            end_array[-1] = start_array[-1] + total_cn - int(total_cn)

        max_x = max(max_x, array_len)
        track_data: pl.DataFrame = pl.DataFrame({
            "chrom": ["seq"] * len(motif_filtered),
            "start": start_array,
            "end": end_array,
            "motif": motif_filtered,
            "strand": ori_filtered,
            "itemRgb": color_filtered,
        }, schema={
            "chrom": pl.Utf8,
            "start": pl.Float64,
            "end": pl.Float64,
            "motif": pl.Utf8,
            "strand": pl.Utf8,
            "itemRgb": pl.Utf8,
        })
        track_name = first_sample if count == 1 else f"{first_sample} ... (n={count})"
        track_dict = {
            "name": track_name,
            "type": "bed",
            "data": track_data,
        }
        track_list.append(track_dict)

    # auto-compute figsize to avoid crowding or excessive sparsity
    n_tracks = len(track_list)

    # get real font size: user override > active template > fallback
    font_size = kwargs.get("font", {}).get("size")
    if font_size is None:
        font_size = _sizing.get_active_font_size()

    # Reserve space for the longest sample name so tracksplot does not
    # squeeze the content area.
    max_name_length = max(len(td["name"]) for td in track_list)

    # ---- sizing: account for optional annotation column(s) ----
    n_anno_dims = len(row_annotations)
    anno_width_px_for_sizing = 0
    if n_anno_dims > 0:
        _, base_height = _sizing.waterfall_height(n_tracks, font_size)
        real_height_approx = base_height - _sizing.WATERFALL_TOP_MARGIN - _sizing.WATERFALL_BOTTOM_MARGIN
        track_height_approx = real_height_approx / n_tracks if n_tracks > 0 else 0
        dim_width = int(track_height_approx) if not deduplicate else int(track_height_approx * 3)
        gap_between = 3  # px gap between dimension columns
        anno_width_px_for_sizing = dim_width * n_anno_dims + gap_between * max(0, n_anno_dims - 1)

    plot_width, total_width = _sizing.waterfall_width(
        max_x, font_size, max_name_length, annotation_width_px=anno_width_px_for_sizing
    )
    plot_height, total_height = _sizing.waterfall_height(n_tracks, font_size)
    width, height = _sizing.resolve_figsize(
        figsize[0],
        figsize[1],
        calc_width=lambda: total_width,
        calc_height=lambda: total_height,
    )
    actual_figsize = (width, height)

    # Detect whether any sample contains gaps ("-") — only draw baselines for
    # aligned data where gaps need to be visualised as breaks in the line.
    has_gap = any(
        any(m == "-" for m in motif_array_dict[s])
        for s, _, _, _ in draw_items
    )
    if has_gap:
        for td in track_list:
            td["draw_baseline"] = True

    fig: go.Figure = tracksplot(
        tracks=track_list,
        region=f"seq:0-{max_x}",
        title="",
        x_title="Copy index",
        figsize=actual_figsize,
        vertical_spacing=0.00,
        track_name_dx=track_name_dx,
        **kwargs
    )

    def _estimate_legend_layout(names, title, avail_width, font_size):
        """
        Estimate the number of wrapped legend rows and the total legend height in pixels.

        Parameters
        ----------
        names : list[str]
            List of legend entry labels, in the same order they are displayed
            horizontally in Plotly.
        title : str | None
            Legend title text.
        avail_width : int
            Available width for the legend, in pixels.
        font_size : int
            Font size used for legend text.

        Returns
        -------
        tuple[int, int]
            Estimated number of rows and total height of the legend in pixels,
            as ``(n_rows, total_height)``.
        """
        if not names:
            return 0, 0

        _marker_px = 20   # marker + left padding
        _gap_px = 10      # entry spacing distance
        _char_px = font_size * 0.7

        entry_widths = [len(str(n)) * _char_px + _marker_px + _gap_px for n in names]

        # simulate the layout logic of Plotly orientation="h"
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

    # ---- Re-place track names and optionally add annotation overlay ----
    # Remove tracksplot's default annotations so we control exact placement.
    fig.layout.annotations = []

    # ---- Annotation overlay ----
    if row_annotations:
        from collections import Counter

        # Collect per-dimension, per-track annotation labels
        track_anno_by_dim: dict[str, list[list[str]]] = {}
        dim_palettes: dict[str, dict[str, str]] = {}
        for dim_name, mapping in row_annotations.items():
            track_anno_by_dim[dim_name] = [
                [mapping.get(s, "Unknown") for s in samples]
                for _, _, _, samples in draw_items
            ]
            all_cats = sorted(set(
                a for annos in track_anno_by_dim[dim_name] for a in annos
            ))
            if isinstance(row_annotation_colormap, dict):
                cur_colormap = row_annotation_colormap[dim_name]
            else:
                cur_colormap = row_annotation_colormap
            _, palette = _get_categorical_colormap(all_cats, cur_colormap)
            dim_palettes[dim_name] = palette

        # Compute exact track height from the rendered figure
        y_domain = fig.layout.yaxis1.domain
        track_height_paper = y_domain[1] - y_domain[0]
        track_height_px = track_height_paper * fig.layout.height

        # Fixed-pixel annotation width per dimension
        dim_width_px = track_height_px * 3 if deduplicate else track_height_px
        gap_px = 3
        plot_width_px = fig.layout.width - fig.layout.margin.l - fig.layout.margin.r
        dim_width_paper = dim_width_px / plot_width_px if plot_width_px > 0 else 0
        gap_paper = gap_px / plot_width_px if plot_width_px > 0 else 0

        n_dims = len(row_annotations)
        total_anno_width_paper = dim_width_paper * n_dims + gap_paper * max(0, n_dims - 1)

        # Rightmost annotation edge just left of plot area
        rightmost_anno_x1 = -gap_paper
        leftmost_anno_x0 = rightmost_anno_x1 - total_anno_width_paper
        name_x = leftmost_anno_x0 - gap_paper

        # Ensure left margin accommodates shifted track names + all annotation columns
        px_per_char = _sizing._scale(_sizing.TRACKSPLOT_NAME_PX_PER_CHAR, font_size)
        text_width_px = max_name_length * px_per_char
        required_left_px = int(abs(name_x) * plot_width_px + text_width_px + gap_px)
        required_left_px = max(required_left_px, int(abs(leftmost_anno_x0) * plot_width_px + gap_px))
        current_left_px = fig.layout.margin.l
        if required_left_px > current_left_px:
            extra_px = required_left_px - current_left_px
            fig.update_layout(
                margin=dict(l=required_left_px),
                width=fig.layout.width + extra_px,
            )
    else:
        name_x = track_name_dx

    # Re-add track-name annotations at computed positions
    for idx, track in enumerate(track_list):
        y_domain = fig.layout[f"yaxis{idx+1}"].domain
        y_center = (y_domain[0] + y_domain[1]) / 2
        fig.add_annotation(
            text=track["name"],
            xref="paper",
            yref="paper",
            x=name_x,
            y=y_center,
            xanchor="right",
            yanchor="middle",
            showarrow=False,
        )

    # ---- Draw annotation shapes per dimension (horizontal side-by-side) ----
    plot_area_height = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b
    if row_annotations:
        min_idx = 0
        _legend_infos: list[tuple[str, str, list[str]]] = []  # (legend_id, title, names)

        for dim_idx, (dim_name, track_annos) in enumerate(track_anno_by_dim.items()):
            palette = dim_palettes[dim_name]
            # Compute x range for this dimension column
            dim_anno_x1 = rightmost_anno_x1 - dim_idx * (dim_width_paper + gap_paper)
            dim_anno_x0 = dim_anno_x1 - dim_width_paper

            # Build ordered category list for this dimension
            dim_cats_ordered: list[str] = []
            for annos in track_annos:
                for a in annos:
                    if a not in dim_cats_ordered:
                        dim_cats_ordered.append(a)

            # Draw shapes for each track
            for t_idx, annos in enumerate(track_annos):
                y_domain = fig.layout[f"yaxis{t_idx+1}"].domain
                y0 = y_domain[0]
                y1 = y_domain[1]

                if len(annos) == 1:
                    fig.add_shape(
                        type="rect",
                        xref="paper", yref="paper",
                        x0=dim_anno_x0, x1=dim_anno_x1,
                        y0=y0, y1=y1,
                        fillcolor=palette[annos[0]],
                        line=dict(width=0),
                        layer="above",
                    )
                else:
                    cat_counts = Counter(annos)
                    x_start = dim_anno_x0
                    total = len(annos)
                    for cat in dim_cats_ordered:
                        if cat not in cat_counts:
                            continue
                        x_end = x_start + (dim_anno_x1 - dim_anno_x0) * cat_counts[cat] / total
                        fig.add_shape(
                            type="rect",
                            xref="paper", yref="paper",
                            x0=x_start, x1=x_end,
                            y0=y0, y1=y1,
                            fillcolor=palette[cat],
                            line=dict(width=0),
                            layer="above",
                        )
                        x_start = x_end

                if deduplicate and t_idx == min_idx:
                    fig.add_annotation(
                        x=(dim_anno_x0 + dim_anno_x1) / 2,
                        y=y1 + gap_paper,
                        xref="paper",
                        yref="paper",
                        text="100%",
                        showarrow=False,
                        xanchor="center",
                        yanchor="bottom",
                        font=dict(size=font_size),
                    )

            # Dimension name label (vertical, below annotation column)
            last_y_domain = fig.layout[f"yaxis{len(track_list)}"].domain
            bottom_y = last_y_domain[0]
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=(dim_anno_x0 + dim_anno_x1) / 2,
                y=bottom_y - 3.0 / plot_area_height,
                text=dim_name,
                showarrow=False,
                textangle=-90,
                xanchor="center",
                yanchor="top",
                font=dict(size=font_size),
            )

            # Legend entries for this dimension
            legend_id = f"legend{dim_idx + 2}"
            for cat in dim_cats_ordered:
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None],
                        mode="markers",
                        marker=dict(size=10, color=palette[cat], symbol="square"),
                        name=str(cat),
                        showlegend=True,
                        legend=legend_id,
                    ),
                )
            _legend_infos.append((legend_id, dim_name, dim_cats_ordered))

        # ---- Layout per-dimension legends stacked vertically ----
        avail_width = fig.layout.width - fig.layout.margin.l - fig.layout.margin.r
        plot_area_height = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b
        _legend_gap_px = 30

        current_y_offset_px = 50  # first legend top distance from plot bottom
        total_legend_height = 0
        for legend_id, title, names in _legend_infos:
            lines, h = _estimate_legend_layout(names, title, avail_width, font_size)
            fig.update_layout(**{
                legend_id: dict(
                    orientation="h",
                    yanchor="top",
                    y=-current_y_offset_px / plot_area_height,
                    x=0.5,
                    xanchor="center",
                    bgcolor="rgba(0,0,0,0)",
                    title=dict(text=title, side="top"),
                )
            })
            current_y_offset_px += h + _legend_gap_px
            total_legend_height += h + _legend_gap_px

        # Increase figure height & bottom margin to fit all legends
        if total_legend_height > 0:
            current_bottom = fig.layout.margin.b
            required_bottom = current_bottom + total_legend_height + 10
            extra_height = required_bottom - current_bottom
            fig.update_layout(
                height=fig.layout.height + extra_height,
                margin=dict(b=required_bottom),
            )

    fig.update_layout(**kwargs)

    if save:
        _save_figure(fig, save, "waterfall")

    return fig

def waterfall_legend(
    adata: ad.AnnData,
    feature: str = "motif",
    sample_order: list[str] | None = None,
    color: str = "id",
    colormap: dict | list | str = "rainbow",
    figsize: tuple[int | None, int | None] = (None, None),
    track_name_dx: float = -0.01,
    save: str | bool | None = None,
    **kwargs
) -> go.Figure:
    """
    Create a legend figure for the waterfall plot.

    Displays colored squares alongside their corresponding motif sequences
    (or color-column values) in a separate figure. The order and coloring
    are consistent with `vp.anno.pl.waterfall()`.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object generated from `pp.read_anno()`.

    feature : str, default="motif"
        Key prefix for the feature arrays stored in ``adata.uns``.
        The function looks up ``uns[f"{feature}_array"]`` and
        ``uns[f"{feature.replace('motif', 'orientation')}_array"]``.
        Common values: ``"motif"`` (raw arrays), ``"aligned_motif"``
        (alignment output from ``vp.anno.tl.sample_msa()``).

    sample_order : list of str, optional
        Unused in legend, kept for API consistency with `waterfall()`.

    color : str, default="id"
        Column name in `adata.var` used to assign coloring. When ``color="id"``,
        legend labels show motif ids; otherwise labels show values from the
        specified column.

    colormap : dict | list | str
        Color mapping specification. Must match the colormap used in the
        corresponding `waterfall()` call for consistent coloring.

    figsize : tuple[int | None, int | None], optional
        Figure size as (width, height) in pixels. Default is (None, None).

        - (None, None): auto-compute both dimensions from data.
        - (w, None): fixed width, auto-compute height from element count.
        - (None, h): fixed height, auto-compute width from label length.
        - (w, h): use user-specified size.

    track_name_dx: float, optional
        Unused in legend, kept for API consistency with `waterfall()`.

    save : str | bool | None, default=None
        If True or a str, save the figure. A string is appended to the default filename.
        Infer the filetype if ending on {'.pdf', '.png', '.svg'}.

    **kwargs
        Additional keyword arguments passed to Plotly `update_layout`.

    Returns
    -------
    fig : go.Figure
        Plotly figure object with colored squares and their labels.

    Examples
    --------
    >>> import vampire as vp
    >>> vp.anno.pl.set_default_plotstyle()
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.pl.waterfall_legend(
    ...     adata,
    ...     color = "motif",
    ...     colormap = "rainbow",
    ... )
    """
    import plotly.graph_objects as go

    # build motif colormap using _get_categorical_colormap
    all_id_list = list(adata.var.index)
    if color == "id":
        id2element = {m: m for m in all_id_list}
    else:
        if color not in adata.var.columns:
            raise ValueError(
                f"color = '{color}' not found in adata.var.columns: {list(adata.var.columns)}"
            )
        id2element = dict(zip(adata.var.index, adata.var[color]))
    all_element_list = list(dict.fromkeys(id2element.values()))
    _, mapped_colormap = _get_categorical_colormap(all_element_list, colormap)

    fig = go.Figure()

    n_items = len(mapped_colormap)
    if n_items == 0:
        return fig

    font_size = kwargs.get("font", {}).get("size")
    if font_size is None:
        font_size = _sizing.get_active_font_size()

    max_label_len = max(len(str(k)) for k in mapped_colormap.keys())

    gap_length: float = 0.1
    y_pos_list: list[float] = []
    for i, (element, color_val) in enumerate(mapped_colormap.items()):
        y_pos = n_items - 1 - i - i * gap_length  # top-to-bottom order
        y_pos_list.append(y_pos)

        # colored square (no border), width=1 height=1 for 1:1 aspect
        fig.add_trace(go.Scatter(
            x=[0, 1, 1, 0, 0],
            y=[y_pos, y_pos, y_pos + 1, y_pos + 1, y_pos],
            fill="toself",
            fillcolor=color_val,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            mode="lines",
        ))

        # label text, left-aligned
        fig.add_annotation(
            x=1.5,
            y=y_pos + 0.5,
            text=str(element),
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=font_size),
        )

    ymax = max(y_pos_list) + 1
    ymin = min(y_pos_list)

    # auto-compute figsize and margins so long legend labels stay visible
    width, height = _sizing.resolve_figsize(
        figsize[0],
        figsize[1],
        calc_width=lambda: _sizing.waterfall_legend_width(max_label_len, font_size),
        calc_height=lambda: _sizing.waterfall_legend_height(n_items, font_size),
    )

    fig.update_layout(
        xaxis=dict(
            range=[0, (ymax - ymin) / float(height) * float(width)],
            zeroline=False,
            showticklabels=False,
            showline=False,
            ticks="",
        ),
        yaxis=dict(
            range=[ymin, ymax],
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
            ticks="",
        ),
        width=width,
        height=height,
        margin=_sizing.waterfall_legend_margin(max_label_len, font_size),
    )

    fig.update_layout(**kwargs)

    if save:
        _save_figure(fig, save, "waterfall_legend")
    
    return fig


"""
#
# haplotype clustering evaluation plot
#
"""
def haplotype_leiden_res_scan(
    adata: ad.AnnData,
    *,
    store_key: str = "haplotype",
    title: str | None = None,
    figsize: tuple[int | None, int | None] = (None, None),
    save: str | bool | None = None,
) -> go.Figure:
    """
    Plot cluster evaluation curve based on resolution-based evaluation
    (from ``vp.anno.tl.haplotype_leiden_res_scan()``).

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with evaluation results.
    store_key : str, default="haplotype"
        Key prefix matching the ``store_key`` used in the tool function.
    figsize : tuple[int | None, int | None], default=(None, None)
        Figure size in pixels.
    title : str | None
        Plot title.
    save : str | bool | None, default=None
        If ``True`` or a ``str``, save the figure. A string is appended to
        the default filename. Infer the filetype if ending on
        {``'.pdf'``, ``'.png'``, ``'.svg'``}.

    Returns
    -------
    go.Figure
        Plotly figure with the evaluation curve (and cluster-count bars for
        resolution-based data).

    Examples
    --------
    >>> import vampire as vp
    >>> vp.anno.pl.set_default_plotstyle()
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.tl.haplotype_neighbor(adata, metrics=["structural", "composition"])
    >>> best_res = vp.anno.tl.haplotype_leiden_res_scan(adata)
    >>> vp.anno.pl.haplotype_leiden_res_scan(adata)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    eval_data = adata.uns.get(f"{store_key}_evaluation")
    if eval_data is None:
        raise KeyError(
            f"Evaluation data not found at uns['{store_key}_evaluation']. "
            f"Run a haplotype evaluation function first."
        )

    # Detect data format: legacy k-based or new resolution-based
    is_resolution_based = "resolution_range" in eval_data
    line_width = _sizing.get_active_line_width()

    resolutions = eval_data["resolution_range"]
    k_values = eval_data["n_clusters"]
    metric_data = eval_data.get("metric", {})
    scores = metric_data.get("scores", [])
    best_res = metric_data.get("best_resolution")
    best_score = metric_data.get("best_score")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Modularity line (left Y)
    fig.add_trace(
        go.Scatter(
            x=resolutions,
            y=scores,
            mode="lines+markers",
            name="Modularity",
            line=dict(color="#212529", width=line_width),
            marker=dict(size=8),
        ),
        secondary_y=False,
    )

    # Mark best point
    if best_res is not None and best_res in resolutions:
        best_idx = int(np.where(np.array(resolutions) == best_res)[0][0])
        fig.add_trace(
            go.Scatter(
                x=[best_res],
                y=[scores[best_idx]],
                mode="markers",
                name=f"Best res={best_res:.2f}",
                marker=dict(color="#f94144", size=14, symbol="star"),
            ),
            secondary_y=False,
        )

    # k bar chart (right Y)
    fig.add_trace(
        go.Bar(
            x=resolutions,
            y=k_values,
            name="k (clusters)",
            opacity=0.25,
            marker_color="#6c757d",
            showlegend=True,
        ),
        secondary_y=True,
    )

    # X-axis: resolution with k on second line
    ticktext = [
        ###f"{r}<br>(k={k})" for r, k in zip(resolutions, k_values)
        r for r in resolutions
    ]
    fig.update_xaxes(
        title_text="Resolution",
        ticktext=ticktext,
        tickvals=resolutions,
        showline=True,
        linecolor="black",
        linewidth=line_width,
        ticks="outside",
        tickwidth=line_width,
    )
    fig.update_yaxes(
        title_text="Modularity",
        showline=True,
        linecolor="black",
        linewidth=line_width,
        ticks="outside",
        tickwidth=line_width,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Number of clusters (k)",
        showline=True,
        linecolor="black",
        linewidth=line_width,
        ticks="outside",
        tickwidth=line_width,
        secondary_y=True,
    )
    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    fig.update_layout(
        width=700 if figsize[0] is None else figsize[0],
        height=600 if figsize[1] is None else figsize[1],
    )

    if save:
        _save_figure(fig, save, "haplotype_evaluation")

    return fig