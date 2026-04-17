from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List, Tuple, Dict, Any, Literal, Optional

if TYPE_CHECKING:
    import anndata as ad
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go
    import plotly.subplots as sp

import logging
logger = logging.getLogger(__name__)


"""
#
# trackplot function
#
"""
def trackplot(
    tracks: List,
    region: str,
    title: str = "",
    x_title: str = "Position (bp)",
    figsize: Tuple[int, int] = (800, 400),
    vertical_spacing: float = 0.02,
    track_name_dx: float = -0.7,
    **kwargs
) -> go.Figure:
    """
    Create a multi-track genomic plot with shared x-axis.

    Each track gets its own subplot with independent y-axis, but all tracks share
    the same x-axis (genomic position). Supported track types include bedgraph,
    bed, and heatmap.

    Parameters
    ----------
    tracks : List[Dict]
        List of track configuration dictionaries. Each dictionary should contain:

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
        - **colorscale** (`List[str]` or `List[Tuple[float, str]]`, optional) - Colorscale for density plot

        Additional options for `"bed"` tracks:

        - **stranded** (`bool`, optional) - Whether to show stranded arrows. Default is False
        - **arrowhead_length** (`float`, optional) - Arrowhead length compared with the region length for stranded arrows. Default is 0.03
        - **color** (`str`, optional) - color. Default is `"#212529"`

        Additional options for `"heatmap"` tracks:

        - **max_value** (`float`, optional) - Maximum value. Default is the maximum in the data
        - **min_value** (`float`, optional) - Minimum value. Default is the minimum in the data
        - **colorscale** (`List[str]` or `List[Tuple[float, str]]`, optional) - Colorscale for heatmap
        - **flip_y** (`bool`, optional) - Whether to flip the y-axis. Default is False
    
    region : str
        Genomic region in the format "chrom:start-end" (e.g., "chr1:1000-2000").
    
    title : str, optional
        Title of the figure. Default is an empty string.

    x_title : str, optional
        Title of the x axis. Default is `"Position (bp)"`.
    
    figsize : Tuple[int, int], optional
        Figure size as (width, height) in pixels. Default is (800, 400).
    
    vertical_spacing : float, optional
        Vertical spacing between subplots as a fraction of total height.
        Default is 0.02.

    track_name_dx: float, optional
        Horizontal offset applied to track name position along the x-axis,expressed as a fraction of the total width.
        Default is -0.7.

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
    >>> tracks = [
    ...     {"name": "Coverage", "type": "bedgraph", "data": df1},
    ...     {"name": "Genes", "type": "bed", "data": df2}
    ... ]
    >>> fig = vp.anno.pl.trackplot(tracks, "chr1:1000-5000", figsize=(1000, 300))
    """
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go
    import plotly.subplots as sp

    VERTICAL_SPACING: float = vertical_spacing
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
    real_height: float = figsize[1] - MIN_MARGIN * 2
    real_width: float = figsize[0] - MAX_NAME_LENGTH * 8 - MIN_MARGIN

    # assign subplot heights
    HAVE_HEATMAP: bool = any(track["type"] == "heatmap" for track in tracks)
    HEATMAP_HEIGHT: float = real_width / 2  # height is half of the figure width
    total_height: float = real_height * (1.0 - VERTICAL_SPACING * (n_tracks - 1))
    assignable_height: float = total_height
    total_ratio: float = sum(track.get("height", 1.0) for track in tracks)
    heights: np.ndarray = np.zeros(n_tracks, dtype=np.float32)
    if HAVE_HEATMAP:
        for idx, track in enumerate(tracks):
            if track["type"] == "heatmap":
                heights[idx] = HEATMAP_HEIGHT
                assignable_height -= HEATMAP_HEIGHT
                total_ratio -= track.get("height", 1.0)
    if assignable_height <= 0:
        raise ValueError("The figure height is too small to fit the heatmaps")
    for idx, track in enumerate(tracks):
        if track["type"] != "heatmap":
            heights[idx] = assignable_height / total_ratio * track.get("height", 1.0)
    heights: np.ndarray = heights / heights.sum()
    heights: List[float] = heights.tolist()
    
    # set subplot titles and heights
    fig = sp.make_subplots(
        rows = n_tracks,
        cols = 1,
        shared_xaxes = True,  # share x-axis across all subplots
        vertical_spacing = VERTICAL_SPACING,  # spacing between subplots
        subplot_titles = [""] * n_tracks,  # no titles, use annotations instead
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
        linewidth = 1.4,          # axis line width
        tickwidth = 1.4,
        row = n_tracks,
        col = 1,
    )
        
    # set figure size
    fig.update_layout(
        width = figsize[0],
        height = figsize[1],
        margin = dict(l=MAX_NAME_LENGTH * 8, r=MIN_MARGIN, t=MIN_MARGIN, b=MIN_MARGIN + 10 + 5),
        autosize = False
    )

    # set figure title
    fig.update_layout(
        title_text = title,
        title_font_size = 16,
        title_x = (fig.layout.margin.l + (fig.layout.width - fig.layout.margin.l - fig.layout.margin.r)/2) / fig.layout.width
    )

    fig.update_layout(
        **kwargs
    )

    # add annotations on the left side
    for idx, track in enumerate(tracks):
        y_domain = fig.layout[f"yaxis{idx+1}"].domain
        y_center = (y_domain[0] + y_domain[1]) / 2

        fig.add_annotation(
            text=track["name"],
            xref="paper",
            yref="paper",
            x=track_name_dx,
            y=y_center,
            xanchor="right",
            yanchor="middle",
            showarrow=False
        )
    
    return fig

def _plot_bedgraph_track_line(
    fig: go.Figure,
    track: Dict,
    data: pl.DataFrame,
    row: int
) -> None:
    """
    Plot a bedgraph track as a line plot.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure object to add the bedgraph track to.
    track : Dict
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
    track: Dict,
    data: pl.DataFrame,
    row: int
) -> None:
    """
    Plot a bedgraph track as a bar plot.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure object to add the bedgraph track to.
    track : Dict
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
    track: Dict,
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

    DEFAULT_COLORMAP: List[str] = ["#5E4FA2", "#3288BD", "#66C2A5", "#ABDDA4", "#E6F598", "#FFFFBF", "#FEE08B", "#FDAE61", "#F46D43", "#D53E4F", "#9E0142"]
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
    customdata: List[List[List[Any]]] = [
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
    track: Dict,
    data: pl.DataFrame,
    row: int,
    region: Tuple[str, int, int]
) -> None:
    """
    Plot a bed track as rectangles or arrows.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure object to add the bed track to.
    track : Dict
        Track configuration dictionary containing plot settings (e.g., name,
        stranded). If stranded is True, arrows are drawn; otherwise rectangles.
    data : pl.DataFrame
        Polars DataFrame containing bed data with columns: chrom, start, end, and
        optionally itemRgb (for colors) and strand (if stranded is True).
    row : int
        Subplot row number (1-indexed) where the bed track should be added.
    region : Tuple[str, int, int]
        Genomic region in the format (chrom, start, end).
    
    Returns
    -------
    None
        The function modifies the figure in-place.
    """
    import plotly.graph_objects as go

    CHROM, START, END = region
    data_sorted = data.sort("start")

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
    track: Dict,
    data: pl.DataFrame,
    row: int
) -> None:
    """
    Plot a triangular heatmap track on the figure.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure object to add the heatmap to.
    track : Dict
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
    DEFAULT_COLORMAP: List[str] = ["#5E4FA2", "#3288BD", "#66C2A5", "#ABDDA4", "#E6F598", "#FFFFBF", "#FEE08B", "#FDAE61", "#F46D43", "#D53E4F", "#9E0142"]
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

def _get_colorscale(values: np.ndarray, color_list: List[str]) -> np.ndarray:
    """
    Get numeric color break boundaries for given values.

    Parameters
    ----------
    values : np.ndarray
        Array of values.
    color_list : List[str]
        List of colors.

    Returns
    -------
    List[Tuple[float, str]]
        List of color scale tuples with lower/upper boundaries and colors.
    """

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
    feature: Literal["motif", "kmer"] = "motif",
    sample_order: Optional[List[str]] = None,
    ksize: Optional[int] = None,
    color: str = "id",
    colormap: dict | List | str = "rainbow",
    figsize: Tuple[int, int] = (600, 1000),
    track_name_dx: float = -0.01,
    **kwargs
) -> go.Figure:
    """
    Create a waterfall plot for motif or k-mer composition across samples.

    The waterfall plot visualizes motif variation across samples in a
    stacked or ordered layout, where each sample is represented along the
    y-axis.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object generated from `pp.read_anno()`.

    feature : {"motif", "kmer"}, default="motif"
        Type of feature to visualize.

        - "motif": uses precomputed motif units from decomposition
        - "kmer": uses k-mer features

    sample_order : list of str, optional
        Ordered list of sample identifiers defining the x-axis order.
        If None, samples are ordered based on the default order in `adata.obs`.

    ksize : int, optional
        k-mer size used when `feature="kmer"`.
        If None, the k-mer size is inferred from the most frequent motif length.

    color : str, default="id"
        Column name in `adata.var` used to assign motif coloring.

    colormap : dict | list | str
        Color mapping for features. Default is `rainbow`.

        - dict: explicit mapping {feature -> color}
        - list: sequential color assignment following input order
        - str: use preset colormap: `rainbow`, `glasbey`, `sequential`

    figsize : Tuple[int, int], optional
        Figure size as (width, height) in pixels. Default is (600, 1000).

    track_name_dx: float, optional
        Horizontal offset applied to track name position along the x-axis,expressed as a fraction of the total width.
        Default is -0.01.

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
    >>> vp.anno.pl.waterfall(
    ...     adata,
    ...     colormap = "rainbow", # rainbow / glasbey / sequential
    ...     figsize = (600, 800),
    ...     sample_order = sample_order,
    ...     track_name_dx = -0.01
    ...     # kwargs for vp.anno.pl.trackplot()
    ...     font = dict(size=8),
    ...     margin = dict(l=120),
    ... )

    >>> vp.anno.pl.waterfall(
    ...     adata,
    ...     feature = "kmer",
    ...     ksize = 5
    ... )
    """
    import polars as pl
    import anndata as ad
    import plotly.graph_objects as go
    
    track_list: List[Dict] = []
    max_x: int = 0

    # check sample_order
    if sample_order is None:
        sample_order: List[str] = list(adata.obs.index)
    all_sample_list: List[str] = list(adata.obs.index)
    missing = set(all_sample_list) - set(sample_order)
    if missing:
        raise KeyError(f"Missing samples in sample_order: {missing}")

    # check color
    match feature:
        case "motif":
            all_id_list: List[str] = list(adata.var.index)
            if color == "id":
                id2element: Dict[str, Any] = {m: m for m in all_id_list}
            else:
                if color not in adata.var.columns:
                    raise ValueError(f"color = '{color}' not found in adata.var.columns: {adata.var.columns}")
                id2element: Dict[str, Any] = Dict(zip(adata.var.index, adata.var[color]))
            all_element_list: List[Any] = list(dict.fromkeys(id2element.values()))

        case "kmer":
            # get ksize
            if ksize is None:
                ksize: int = len(adata.var["motif"][0])
            # get k-mers
            key: str = f"kmer_array_k={ksize}"
            if key not in adata.uns:
                _get_kmer_array(adata, ksize) # TODO
            kmer_array_dict: Dict[str, List[str]] = adata.uns[f"kmer_array_k={ksize}"]
            all_element_list: List[Any] = list(dict.fromkeys(kmer for arr in kmer_array_dict.values() for kmer in arr))

        case _:
            raise ValueError(f"Invalid feature: {feature}, feature must be 'motif' or 'kmer'")

    # assign color
    element_num: int = len(all_element_list)

    RAINBOW_COLORMAP: List[str] = [
        "#f94144", "#f8961e", "#f9c74f", "#90be6d", "#43aa8b", "#4d908e", "#577590", "#277da1", "#5983f2", "#898bf1", 
        "#8945dc"
    ]
    GLASBEY_COLORMAP: List[str] = [
        "#5983f2", '#87db96', '#eef248', '#f25b76', '#f29dd3', '#38abbd', '#f2b668', '#a570f2', '#f2b668', '#288126',
        '#3e3ed1', '#48f0d1', '#987b74', '#c5c2f2', '#96ab83', '#b639b1', '#63586e', '#722222', '#428acd', '#f1eba8',
        '#154715', '#877928', '#f2948c', '#e2bebe', '#627767', '#afe0f2', '#e04392', '#a386af', '#6a2e9b', '#3bbc38',
        '#965e2d', '#945361', '#c0bc39', '#768190', '#31a68d', '#9fa3f2', '#583f41', '#361010', '#f0c8f2', '#c0dbc1',
        '#d0863f', '#96aead', '#e07ff2', '#445c7b', '#ba7b97', '#545d31', '#bba183', '#297c80', '#adf248', '#7e4288',
        '#91c0f2', '#7c6af2', '#879a2f', '#f14acf', '#b9a6bb', '#bb7266', '#705c4a', '#1e6249', '#3ccbca', '#d9ddf2',
        '#661e53', '#7877a7', '#b99c37', '#d5c8a7', '#d9acf2', '#c44af0', '#8a6080', '#982d7a', '#669364', '#c8999b',
        '#38412d', '#9ba5b9', '#868667', '#f27748', '#cdf2ee', '#bc91f2', '#84c5aa', '#c7f2a5', '#8f6aa8', '#714632',
        '#f2b399', '#a03044', '#cc3d3d', '#5a2632', '#6f8f8d', '#f287ab', '#53656e', '#719bc1', '#a17b51', '#49d2f1',
        '#37ba7f', '#5c3e59', '#9cc95b', '#bbca8f', '#645c99', '#4a2d1e', '#755f23', '#f2d175', '#923cc4', '#7bf0f2',
        '#82274f', '#745960', '#2c7995', '#a18795', '#7e7387', '#898bf1', '#b95a92', '#c25569', '#f2afc1', '#b975b3',
        '#33621e', '#28866b', '#66793b', '#9190b5', '#a4d1cc', '#c38f75', '#c7b578', '#5f1f68', '#916358', '#b3c2d0',
        '#e9f2d4', '#43e044', '#c293b8', '#6638bb', '#7d9581', '#6c6b4f', '#aaf2c4', '#376eb7', '#b6a5e0', '#589b3d',
        '#634b79', '#154837', '#8eb6c8', '#90472b', '#456861', '#ae3470', '#f29d71', '#f2d4e3', '#794460', '#24310e',
        '#986c77', '#68aba8', '#97975b', '#45754b', '#b86237', '#c6dc6b', '#8945dc', '#84b26a', '#a5bba8', '#cf7587',
        '#cd9c62', '#ef9bf2', '#4e5e4d', '#d1c4df', '#734146', '#618e9c', '#77a3f2', '#a437ba', '#deafd0', '#f2c598',
        '#434414', '#b4cdf2', '#a3aede', '#5e6f8b', '#db80b8', '#9b92a7', '#534e35', '#b17ac9', '#9988d5', '#b78082',
        '#d840d7', '#afae91', '#362e10', '#d794ad', '#a45851', '#a5b167', '#79ab8a', '#9ec79b', '#a78f60', '#266171',
        '#2b9086', '#55231a', '#607779', '#395332', '#d98468', '#5a4536', '#8266b2', '#505565', '#c490d1', '#944d8c',
        '#c9dde5', '#46b4df', '#9e7696', '#dbb641', '#88f28a', '#e5d844', '#714f22', '#5b3346', '#7f8ec0', '#829ba7',
        '#4bddae', '#afebd8', '#7d2574', '#867059', '#d76b60', '#af657c', '#77577f', '#f2d2c1', '#6b76bd', '#d6b7a1',
        '#f27be6', '#d2e3b2', '#b5764f', '#cca9b5', '#b38835', '#8d7ca9', '#f268b3', '#f2e3bd', '#3a534b', '#ae9185',
        '#6e805d', '#b89ec7', '#5ae1f1', '#985579', '#9c659f', '#732b90', '#826a78', '#636284', '#eca6a3', '#dbd094',
        '#6a6720', '#195354', '#5761f2', '#3dcd76', '#edb3f0', '#7b8e52', '#534019', '#92c9d1'
    ]
    SEQUENTIAL_COLORMAP: List[str] = [
        "#FED976", "#FDBA9B", "#F7958D", "#ED96C9", "#ec57e5", "#a4cae4", "#7bd1ca", "#bfde9f", "#58d581",
        "#FEBD0B", "#FC8D59", "#EF3B2C", "#DD3497", "#af14a8", "#4292C6", "#35978F", "#7FBC41", "#238B45",
        "#DEA402", "#d24504", "#91150b", "#7d1552", "#3d073a", "#204c69", "#143936", "#3f5d20", "#092512"
    ]
    COLORMAP_OPTIONS: Dict[str, List[str]] = {
        "rainbow": RAINBOW_COLORMAP,
        "glasbey": GLASBEY_COLORMAP, 
        "sequential": SEQUENTIAL_COLORMAP,
    }

    match colormap:
        case str():
            if colormap not in COLORMAP_OPTIONS:
                raise ValueError(f"colormap {colormap} is not found! please select from {COLORMAP_OPTIONS.keys()}")
            DEFAULT_COLORMAP: List[str] = COLORMAP_OPTIONS[colormap]
            
            if element_num > len(DEFAULT_COLORMAP):
                logger.warning(f"Number of {color} is larger then number of colors in default colormap, using black to represent remaining motifs")
                DEFAULT_COLORMAP += ["#1a1a1a"] * (element_num - len(DEFAULT_COLORMAP))
            mapped_colormap: Dict[str, str] = dict(zip(all_element_list, DEFAULT_COLORMAP[:element_num]))

        case list():
            if not all(isinstance(x, str) for x in colormap):
                raise TypeError("List colormap must be List[str]")
            
            if element_num > len(colormap):
                logger.warning(f"Number of {color} is larger then number of colors in given colormap, using black to represent remaining motifs")
                colormap += ["#1a1a1a"] * (element_num - len(colormap))
            mapped_colormap: Dict[str, str] = dict(zip(all_element_list, colormap[:element_num]))

        case dict():
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in colormap.items()):
                raise TypeError("Dict colormap must be Dict[str, str]")

            missing = set(all_element_list) - set(colormap.keys())
            if missing:
                raise KeyError(f"Missing {color} in colormap: {missing}")
            
            mapped_colormap: Dict[str, str] = colormap

    match feature:
        case "motif":
            motif_array_dict: Dict[str, List[str]] = adata.uns["motif_array"]
            orientation_array_dict: Dict[str, List[str]] = adata.uns["orientation_array"]
            for sample in sample_order:
                motif_array: List[str] = motif_array_dict[sample]
                orientation_array: List[str] = orientation_array_dict[sample]
                color_array: List[str] = [mapped_colormap[m] for m in motif_array]
                array_len: int = len(motif_array)
                max_x: int = max(max_x, array_len)
                track_data: pl.DataFrame = pl.DataFrame({
                    "chrom": ["seq"] * array_len,
                    "start": [i for i in range(array_len)],
                    "end": [i + 1 for i in range(array_len)],
                    "motif": motif_array,
                    "strand": orientation_array,
                    "itemRgb": color_array,
                })
                track_dict = {
                    "name": sample,
                    "type": "bed",
                    "data": track_data,
                }
                track_list.append(track_dict)

        case "kmer":
            kmer_array_dict: Dict[str, List[str]] = adata.uns[f"kmer_array_k={ksize}"]
            for sample in sample_order:
                kmer_array: List[str] = kmer_array_dict[sample]
                orientation_array: List[str] = ["+"] * len(kmer_array)
                color_array: List[str] = [mapped_colormap[m] for m in kmer_array]
                array_len: int = len(kmer_array)
                max_x: int = max(max_x, array_len)
                track_data: pl.DataFrame = pl.DataFrame({
                    "chrom": ["seq"] * array_len,
                    "start": [i for i in range(array_len)],
                    "end": [i + 1 for i in range(array_len)],
                    "kmer": kmer_array,
                    "strand": orientation_array,
                    "itemRgb": color_array,
                })
                track_dict = {
                    "name": sample,
                    "type": "bed",
                    "data": track_data,
                }
                track_list.append(track_dict)

        case _:
            raise ValueError(f"Invalid feature: {feature}, feature must be 'motif' or 'kmer'")

    fig: go.Figure = trackplot(
        tracks = track_list,
        region = f"seq:0-{max_x}",
        title = "",
        x_title = "Copy index" if feature == "motif" else "Kmer index",
        figsize = figsize,
        vertical_spacing = 0.00,
        track_name_dx = track_name_dx,
        **kwargs
    )
    return fig

def _get_kmer_array(
    adata: ad.AnnData,
    ksize: int
) -> None:
    """
    add kmer array to adata.uns[f'kmer_array_k={`ksize`}'] in-place

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object generated from `pp.read_anno()`.

    ksize : int
        length of kmer.

    Returns
    -------
    None
    """
    seq_array: Dict[str, str] = adata.uns["sequence"]
    kmer_array: Dict[str, List[str]] = {
        sample: [seq[i: i + ksize] for i in range(len(seq) - ksize + 1)]
        for sample, seq in seq_array.items()
    }
    
    adata.uns[f"kmer_array_k={ksize}"] = kmer_array


"""
#
# DNA logo function
#
"""
def logo(
    adata: ad.AnnData, 
    reference_motif: Optional[str] = None,
    feature: Literal["count", "probability", "information"] = "information",
    colormap: dict = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728", "-": "#403d39"},
    conserved_color: Optional[str] = "#cccccc",
    title: str = "",
    figsize: Tuple[int, int] = (1200, 300),
    **kwargs
) -> go.Figure:
    """
    Plot the logo plot from anndata object
    
    Parameters
    ----------
    adata: ad.AnnData
        The AnnData object.

    reference_motif: Optional[str]
        The reference motif used as alignment reference. Default is None (use the most frequent motif).
    
    feature: Literal["count", "probability", "information"]
        The feature to use. Default is "information".
    
    colormap: dict
        The colors of the bases. Default is `{"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728", "-": "#403d39"}`.
    
    conserved_color: Optional[str]
        Override color for conserved sites (non-variant positions). Default is "#cccccc".
        If set to None, conserved sites will use the general base color instead.

    title: str
        The title of the plot. Default is empty.

    figsize : Tuple[int, int], optional
        Figure size as (width, height) in pixels. Default is (1200, 300).
    
    **kwargs
        Additional keyword arguments passed to Plotly `update_layout`.
        Used to control figure-level styling (e.g. template, margin,
        background color, legend settings).
    
    Returns
    -------
    go.Figure
        The logo figure.

    Examples
    --------
    >>> import vampire as vp
    >>> vp.anno.pl.logo(
    ...     adata,
    ...     feature = "information",
    ...     figsize = (1200, 300)
    ... )
    """
    import parasail
    import numpy as np
    import anndata as ad
    import plotly.graph_objects as go

    # config
    LETTERS: List[str] = ["A", "C", "G", "T", "-"]
    LETTER_TO_IDX: Dict[str, int] = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
        "-": 4,
    }

    # get reference motif
    if reference_motif is None:
        if len(adata.var) == 0:
            raise ValueError("adata.var is empty")
        reference_motif = (
            adata.var
            .sort_values("copy_number", ascending=False)
            ["motif"]
            .iloc[0]
        )
    
    # align motifs
    count: np.ndarray = np.zeros((len(reference_motif), len(LETTERS)))
    MATRIX = parasail.matrix_create("ACGT", 2, -1)
    for _, row in adata.var.iterrows():
        motif = row["motif"]
        cn = row["copy_number"]

        result = parasail.nw_trace_striped_16(
            motif,
            reference_motif,
            5,
            1,
            MATRIX
        )

        ref_aln = result.traceback.ref
        seq_aln = result.traceback.query
        ref_pos = 0

        for r, s in zip(ref_aln, seq_aln):
            if r == "-":
                continue
            if s in LETTER_TO_IDX:
                count[ref_pos, LETTER_TO_IDX[s]] += cn
            ref_pos += 1
        
        assert ref_pos == len(reference_motif) # check if the alignment is correct

    match feature:
        case "count":
            matrix = count
        case "probability":
            matrix = count / count.sum(axis = 1, keepdims=True)   
        case "information":
            matrix = count / count.sum(axis = 1, keepdims=True)
            matrix = _compute_information_content(matrix)
        case _:
            raise ValueError(f"Invalid feature: {feature}")

    # plot
    fig: go.Figure = logo_from_matrix(
        matrix = matrix,
        letters = LETTERS,
        feature = feature,
        colormap = colormap,
        conserved_color = conserved_color,
        title = title,
        figsize = figsize,
        **kwargs
    )

    return fig

def logo_from_matrix(
    matrix: np.ndarray,
    letters: List[str],
    feature: Literal["count", "probability", "information"] = "information",
    colormap: dict = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728", "-": "#403d39"},
    conserved_color: Optional[str] = "#cccccc",
    title: str = "",
    figsize: Tuple[int, int] = (1200, 300),
    **kwargs
) -> go.Figure:
    """
    Plot the logo plot from 2D matrix, such as count matrix, frequency matrix and position weight matrix (PWM).
    
    Parameters
    ----------
    matrix: np.ndarray
        The 2D position-by-symbol matrix used to construct a sequence logo. Shape is (L, K).
        L = number of positions (x-axis, sequence length)
        K = number of symbols (defined by `letters`)

        matrix[i, j] gives the contribution (count/probability/information)
        of symbol `letters[j]` at position i.

    letters : List[str]
        Symbols corresponding to matrix columns, e.g. ["A", "C", "G", "T", "-"].
    
    feature: Literal["count", "probability", "information"]
        The feature to use. Default is "information".
    
    colormap: dict
        The colors of the bases. Default is `{"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728", "-": "#403d39"}`.
    
    conserved_color: Optional[str]
        Override color for conserved sites (non-variant positions). Default is "#cccccc".
        If set to None, conserved sites will use the general base color instead.

    title: str
        The title of the plot. Default is empty.

    figsize : Tuple[int, int], optional
        Figure size as (width, height) in pixels. Default is (1200, 300).
    
    **kwargs
        Additional keyword arguments passed to Plotly `update_layout`.
        Used to control figure-level styling (e.g. template, margin,
        background color, legend settings).
    
    Returns
    -------
    go.Figure
        The logo figure.

    Examples
    --------
    >>> import numpy as np
    >>> import vampire as vp
    >>> vp.anno.pl.logo_from_matrix(
    ...     np.array([
    ...         [8, 1, 0, 1, 0, 0, 0],
    ...         [0, 7, 0, 0, 1, 2, 0],
    ...         [0, 0, 9, 0, 0, 0, 1],
    ...         [0, 2, 0, 7, 0, 1, 0],
    ...         [1, 1, 1, 0, 7, 0, 0],
    ...         [0, 1, 0, 1, 0, 8, 0],
    ...         [1, 0, 0, 1, 0, 0, 8],
    ...     ]),
    ...     feature = "count",
    ...     letters = ["V", "A", "M", "P", "I", "R", "E"],
    ...     colormap = {"V": "#f64021", "A": "#f98016", "M": "#ffff00", "P": "#00cc66", "I": "#496ddb", "R": "#7209b7", "E": "#a01a7d"},
    ...     figsize = (800, 300)
    ... )
    """
    import re
    import numpy as np
    import plotly.graph_objects as go

    LETTERS: List[str] = letters
    LETTER_PATHS: Dict[str, str] = _get_letter_paths(letters = LETTERS)
    LETTER_WIDTH: float = 0.9 # width per position, 0-1

    # check colormap
    missing = set(LETTERS) - set(colormap.keys())
    if missing:
        raise ValueError(f"Letters {missing} are not covered in colormap!")

    fig: go.Figure = go.Figure()

    for pos, row in enumerate(matrix):
        order = np.argsort(row)
        y_offset = 0
        # get conservation
        is_conserved: np.ndarray = np.count_nonzero(row > 1e-6) == 1

        for idx in order:
            letter = LETTERS[idx]
            height = row[idx]

            if height <= 1e-6:
                continue

            glyph = LETTER_PATHS[letter]
            verts = glyph["vertices"].copy()
            codes = glyph["codes"]

            # normalize glyph
            min_x, max_x = verts[:, 0].min(), verts[:, 0].max()
            min_y, max_y = verts[:, 1].min(), verts[:, 1].max()

            sx = 1.0 / (max_x - min_x)
            sy = 1.0 / (max_y - min_y)

            verts[:, 0] = (verts[:, 0] - min_x) * sx
            verts[:, 1] = (verts[:, 1] - min_y) * sy

            # scale to final layout
            verts[:, 0] = verts[:, 0] * LETTER_WIDTH + pos
            verts[:, 1] = verts[:, 1] * height + y_offset

            # build SVG path string (only here!)
            path_parts = []
            i = 0

            while i < len(verts):
                c = codes[i]

                if c == 1:  # MOVETO
                    x, y = verts[i]
                    path_parts.append(f"M {x} {y}")
                    i += 1

                elif c == 2:  # LINETO
                    x, y = verts[i]
                    path_parts.append(f"L {x} {y}")
                    i += 1

                elif c == 3:  # CURVE3 → Q
                    x1, y1 = verts[i]
                    x2, y2 = verts[i + 1]
                    path_parts.append(f"Q {x1} {y1} {x2} {y2}")
                    i += 2

                elif c == 4:  # CURVE4 → C
                    x1, y1 = verts[i]
                    x2, y2 = verts[i + 1]
                    x3, y3 = verts[i + 2]
                    path_parts.append(f"C {x1} {y1} {x2} {y2} {x3} {y3}")
                    i += 3

                elif c == 79:  # CLOSEPOLY
                    path_parts.append("Z")
                    i += 1

            final_path = " ".join(path_parts)

            x, y = _path_to_xy(final_path)

            fillcolor: str = (
                conserved_color
                if (conserved_color is not None and is_conserved)
                else colormap[letter]
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    fill="toself",
                    fillcolor=fillcolor,
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False
                )
            )

            # hover support (invisible bar)
            fig.add_trace(go.Bar(
                x=[pos + LETTER_WIDTH / 2],
                y=[height],
                base=y_offset,
                width=LETTER_WIDTH,
                marker=dict(color=colormap[letter], opacity=0),
                hovertemplate=f"{letter}<br>{feature}={height:.3f}<extra></extra>",
                showlegend=False
            ))

            y_offset += height

    if feature == "probability":
        fig.update_yaxes(range=[0, 1])

    if feature == "information":
        fig.update_yaxes(range=[0, 2]) 

    fig.update_layout(
        xaxis = dict(
            range=[0, len(matrix)],
            tickformat="d",
            title="Motif (bp)"
        ),
        yaxis = dict(
            title=feature
        ),
        title = title,
        template="simple_white",
        width = figsize[0],
        height = figsize[1]
    )

    fig.update_layout(
        **kwargs
    )

    return fig

def _get_letter_paths(
    letters: List[str] = ["A", "C", "G", "T", "-"],
    fontsize: int = 1, 
    fontfamily: str = "DejaVu Sans Mono", 
    weight: str = "bold"
) -> Dict[str, str]:
    """
    Get the letter paths

    Parameters
    ----------
    letters: List[str]
        List of letters. Default is ["A", "C", "G", "T"].
    fontsize: int
        Font size. Default is 1.
    fontfamily: str
        Font family. Default is "DejaVu Sans".
    weight: str
        Font weight. Default is "bold".

    Returns
    -------
    Dict[str, str]
        Dictionary of letter paths.
    """
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties

    fp = FontProperties(family=fontfamily, weight=weight)
    paths = {}

    for letter in letters:
        tp = TextPath((0, 0), letter, size=fontsize, prop=fp)

        paths[letter] = {
            "vertices": tp.vertices.copy(),
            "codes": tp.codes.copy()
        }

    return paths

def _path_to_xy(
    path: str, 
    n_samples=30
) -> Tuple[List[float], List[float]]:
    """
    Convert an SVG path string into x/y coordinate arrays for polygon rendering.

    This function parses a subset of SVG path commands and converts them into
    discrete (x, y) points that can be used for plotting (e.g. with Plotly
    `Scatter` and `fill="toself"`).

    Supported commands:
    - M: Move to (start a new subpath)
    - L: Line to
    - Q: Quadratic Bézier curve (approximated by sampling)
    - Z: Close path

    Parameters
    ----------
    path : str
        SVG path string consisting of commands (M, L, Q, Z) and numeric coordinates.
        Example: "M x0 y0 L x1 y1 Q cx cy x2 y2 Z"

    n_samples : int, optional
        Number of sample points used to approximate each quadratic Bézier curve (Q).
        Higher values result in smoother curves but increase computational cost.
        Default is 30.

    Returns
    -------
    x : List[float]
        List of x-coordinates representing the polygon vertices.

    y : List[float]
        List of y-coordinates representing the polygon vertices.

    Notes
    -----
    - Bézier curves (Q) are converted into line segments via uniform sampling.
    - The returned coordinates form a closed polygon when a "Z" command is present.
    - This function does not support cubic Bézier curves (C) or other SVG commands.

    Examples
    --------
    >>> path = "M 0 0 L 1 0 Q 1.5 0.5 1 1 Z"
    >>> x, y = _path_to_xy(path)
    >>> len(x)  # includes sampled points along the curve
    """
    import numpy as np

    tokens = path.replace(",", " ").split()
    
    x, y = [], []
    i = 0
    
    while i < len(tokens):
        cmd = tokens[i]
        
        if cmd == "M" or cmd == "L":
            xi = float(tokens[i+1])
            yi = float(tokens[i+2])
            x.append(xi)
            y.append(yi)
            i += 3
        
        elif cmd == "Q":
            x0, y0 = x[-1], y[-1]
            cx = float(tokens[i+1])
            cy = float(tokens[i+2])
            x1 = float(tokens[i+3])
            y1 = float(tokens[i+4])
            
            # Bézier sampling
            for t in np.linspace(0, 1, n_samples):
                xt = (1-t)**2 * x0 + 2*(1-t)*t*cx + t**2 * x1
                yt = (1-t)**2 * y0 + 2*(1-t)*t*cy + t**2 * y1
                x.append(xt)
                y.append(yt)
            
            i += 5
        
        elif cmd == "Z":
            x.append(x[0])
            y.append(y[0])
            i += 1
        
        else:
            raise ValueError(f"Unsupported path command: {cmd}")
    
    return x, y

def _compute_information_content(
    mat: np.ndarray,
) -> np.ndarray:
    """
    Compute the information content

    Parameters
    ----------
    mat: np.ndarray
        The 2D frequency/probability matrix.

    Returns
    -------
    pwm: np.ndarray
        The position weight matrix (PWM) matrix.
    """
    import numpy as np

    ic = []
    for row in mat:
        H = -np.sum(row * np.log2(row + 1e-12))
        R = max(0, 2 - H)
        ic.append(row * R)
    
    return np.array(ic)