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
    figsize: Tuple[int, int] = (800, 400),
    vertical_spacing: float = 0.02
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
    figsize : Tuple[int, int], optional
        Figure size as (width, height) in pixels. Default is (800, 400).
    vertical_spacing : float, optional
        Vertical spacing between subplots as a fraction of total height.
        Default is 0.02.

    Returns
    -------
    go.Figure
        A Plotly figure object with all tracks plotted as subplots.

    Examples
    --------
    >>> tracks = [
    ...     {"name": "Coverage", "type": "bedgraph", "data": df1},
    ...     {"name": "Genes", "type": "bed", "data": df2}
    ... ]
    >>> fig = trackplot(tracks, "chr1:1000-5000", figsize=(1000, 300))
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
        title_text = "Position (bp)",
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

    # add annotations on the left side
    for idx, track in enumerate(tracks):
        y_domain = fig.layout[f"yaxis{idx+1}"].domain
        y_center = (y_domain[0] + y_domain[1]) / 2

        fig.add_annotation(
            text=track["name"],
            xref="paper",
            yref="paper",
            x=-0.07,
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
    sample_order: Option[List[str]] = None,
    ksize: Option[int] = None,
    color: str = "id",
    palette: dict | List | None = None,
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

    palette : dict | list | None, optional
        Color mapping for features.

        - dict: explicit mapping {feature -> color}
        - list: sequential color assignment following input order
        - None: default internal colormap will be used

    Returns
    -------
    fig : go.Figure
        Plotly figure object representing the waterfall visualization.

    Examples
    --------
    >>> fig = vp.pl.waterfall(adata)

    >>> fig = vp.pl.waterfall(
    ...     adata,
    ...     feature="kmer",
    ...     ksize=5
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
        raise KeyError(f"Missing motifs in sample_order: {missing}") # TODO

    """COLORS = {
        "warm": [
            "FEE8C8", "FDD49E", "FDBB84", "FFEDA0", "FED976", "FEB24C", "FC8D59"
        ],
        "green": [
            "C7E9C0", "A1D99B", "B8E186", "7FBC41", "74C476",
            "41AB5D", "238B45", "006D2C", "276419", "4D9221"
        ],
        "blue": [
            "80CDC1", "35978F", "01665E", "6BAED6", "4292C6", "2171B5"
        ],
        "purple": [
            "88419D", "810F7C", "F768A1", "DD3497", "AE017E", "7A0177"
        ],
        "red": [
            "FCBBA1", "FC9272", "FB6A4A", "EF3B2C", "CB181D"
        ],
        "grey": [
            "grey"
        ]
    }"""

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
                _calculate_kmer_array(adata, ksize) # TODO
            kmer_array_dict: Dict[str, List[str]] = adata.uns[f"kmer_array_k={ksize}"]
            all_element_list: List[Any] = list(dict.fromkeys(kmer for arr in kmer_array_dict.values() for kmer in arr))

        case _:
            raise ValueError(f"Invalid feature: {feature}, feature must be 'motif' or 'kmer'")

    # assign color
    element_num: int = len(all_element_list)
    DEFAULT_COLORMAP: List[str] = [
        "#FEE8C8", "#FDD49E", "#FDBB84", "#FC8D59", "#FFEDA0", "#FED976", "#FEB24C",
        "#C7E9C0", "#A1D99B", "#74C476", "#41AB5D", "#238B45", "#006D2C", "#276419",
        "#4D9221", "#7FBC41", "#B8E186", "#80CDC1", "#35978F", "#01665E", "#6BAED6",
        "#4292C6", "#2171B5", "#FCBBA1", "#FC9272", "#FB6A4A", "#EF3B2C", "#CB181D",
        "#88419D", "#810F7C", "#F768A1", "#DD3497", "#AE017E", "#7A0177"
    ]
    match palette:
        case None:
            if element_num > len(DEFAULT_COLORMAP):
                logger.warning(f"Number of {color} is larger then number of colors in default colormap, using grey to represent remaining motifs")
                DEFAULT_COLORMAP += ["grey"] * (element_num - len(DEFAULT_COLORMAP))
            colormap = dict(zip(all_element_list, DEFAULT_COLORMAP[:element_num]))

        case list():
            if not all(isinstance(x, str) for x in palette):
                raise TypeError("List palette must be List[str]")
            
            if element_num > len(palette):
                logger.warning(f"Number of {color} is larger then number of colors in given colormap, using grey to represent remaining motifs")
                palette += ["grey"] * (element_num - len(palette))
            colormap = dict(zip(all_element_list, palette[:element_num]))

        case dict():
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in palette.items()):
                raise TypeError("Dict palette must be Dict[str, str]")

            missing = set(all_element_list) - set(palette.keys())
            if missing:
                raise KeyError(f"Missing {color} in palette: {missing}")
            
            colormap = palette

    match feature:
        case "motif":
            motif_array_dict: Dict[str, List[str]] = adata.uns["motif_array"]
            orientation_array_dict: Dict[str, List[str]] = adata.uns["orientation_array"]
            for sample in sample_order:
                motif_array: List[str] = motif_array_dict[sample]
                orientation_array: List[str] = orientation_array_dict[sample]
                color_array: List[str] = [colormap[m] for m in motif_array]
                array_len: int = len(motif_array)
                max_x: int = max(max_x, array_len)
                track_data: pl.DataFrame({
                    "chrom": ["seq"] * array_len,
                    "start": list(range(array_len)),
                    "end": list(range(array_len) + 1),
                    "motif": motif_array,
                    "strand": orientation_array,
                    "color": color_array,
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
                color_array: List[str] = [colormap[m] for m in kmer_array]
                array_len: int = len(kmer_array)
                max_x: int = max(max_x, array_len)
                track_data: pl.DataFrame({
                    "chrom": ["seq"] * array_len,
                    "start": list(range(array_len)),
                    "end": list(range(array_len) + 1),
                    "kmer": kmer_array,
                    "strand": orientation_array,
                    "color": color_array,
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
        figsize = (800, 400),
        vertical_spacing = 0.02
    )
    return fig

    ################

    """
    # give seq index (y-axis)
    seq_df = (
        df.select("seq")
        .unique()
        .with_row_count("y")   # seq → y index
    )

    df = df.join(seq_df, on="seq")

    height = 0.8

    # compute hover data (fully vectorized)
    df = df.with_columns([
        ((pl.col("start") + pl.col("end")) / 2).alias("hover_x"),
        (pl.col("y") + height / 2).alias("hover_y"),
        pl.format(
            "seq: {}<br>motif: {}<br>start: {}<br>end: {}<br>rep: {}<br>motif_len: {}",
            pl.col("seq"),
            pl.col("motif"),
            pl.col("start"),
            pl.col("end"),
            pl.col("rep"),
            pl.col("motif_len")
        ).alias("hover_text")
    ])

    # construct shapes (unavoidable to convert to Python)
    shapes = [
        dict(
            type="rect",
            x0=row["start"],
            x1=row["end"],
            y0=row["y"],
            y1=row["y"] + height,
            fillcolor=colormap.get(row["motif"], "gray"),
            line=dict(width=0),
        )
        for row in df.select(["start", "end", "y", "motif"]).to_dicts()
    ]

    # construct hover scatter (fully vectorized)
    fig = go.Figure()

    fig.update_layout(shapes=shapes)

    fig.add_trace(go.Scatter(
        x=df["hover_x"].to_list(),
        y=df["hover_y"].to_list(),
        mode="markers",
        marker=dict(size=6, opacity=0),  # transparent
        text=df["hover_text"].to_list(),
        hoverinfo="text"
    ))

    # update y-axis labels
    fig.update_yaxes(
        tickmode='array',
        tickvals=seq_df["y"].to_list(),
        ticktext=seq_df["seq"].to_list()
    )

    fig.update_layout(
        title="Repeat Structure",
        xaxis_title="Copy Number / Position",
        yaxis_title="Sample",
        template="simple_white",
        height=600 + df["seq"].n_unique() * 20
    )

    return fig
    """

"""
#
# DNA logo function
#
"""
def logo(
    adata: ad.AnnData, 
    reference_motif: Optional[str] = None,
    yaxis: Literal["count", "probability", "information"] = "information",
    color: dict = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728"},
    title: str = ""
) -> go.Figure:
    """
    Plot the logo plot
    
    Parameters
    ----------
    adata: ad.AnnData
        The AnnData object.

    reference_motif: Optional[str]
        The reference motif used as alignment reference. Default is None (use the most frequent motif).
    
    yaxis: Literal["count", "probability", "information"]
        The y-axis to use. Default is "information".
    
    color: dict
        The colors of the bases. Default is `{"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728"}`.
    
    title: str
        The title of the plot. Default is empty.
    
    Returns
    -------
    go.Figure
        The logo figure.
    """
    import re
    import parasail
    import numpy as np
    import anndata as ad

    # config
    LETTERS: List[str] = ["A", "C", "G", "T", "-"]
    LETTER_PATHS: Dict[str, str] = _get_letter_paths(letters = LETTERS)
    LETTER_TO_IDX: Dict[str, int] = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
        "-": 4,
    }
    LETTER_WIDTH: float = 0.8 # width per position, 0-1

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
            reference_motif,
            motif,
            5,
            1,
            MATRIX
        )

        ref_aln = result.traceback.ref
        seq_aln = result.traceback.query
        ref_pos = 0

        for r, s in zip(ref_aln, seq_aln):
            if r != "-":
                if s in LETTER_TO_IDX:
                    count[ref_pos, LETTER_TO_IDX[s]] += cn
                ref_pos += 1

        assert ref_pos == len(reference_motif) # check if the alignment is correct

    match yaxis:
        case "count":
            pwm = count
        case "probability":
            pwm = count / count.sum(axis = 0)   
        case "information":
            pwm = count / count.sum(axis = 0)
            pwm = _compute_information_content(pwm)
        case _:
            raise ValueError(f"Invalid y-axis: {yaxis}")

    # plot
    fig: go.Figure = go.Figure()
    for pos, row in enumerate(pwm):
        order: List[int] = np.argsort(row)  # from small to large
        y_offset: float = 0

        for idx in order:
            letter: str = LETTERS[idx]
            height: float = row[idx]
            # skip if height is too small
            if height <= 1e-6:  
                continue
            raw_path: str = LETTER_PATHS[letter]

            # normalize path because matplotlib font coordinates are not 0-1, need to standardize
            coords = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", raw_path)]
            xs = coords[::2]
            ys = coords[1::2]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            norm_sx = 1.0 / (max_x - min_x)
            norm_sy = 1.0 / (max_y - min_y)

            normalized_path = _transform_path(
                raw_path,
                sx = norm_sx,
                sy = norm_sy,
                tx = - min_x * norm_sx,
                ty = - min_y * norm_sy
            )

            # apply final transformation
            final_path = _transform_path(
                normalized_path,
                sx = LETTER_WIDTH, # width per position
                sy = height, # height
                tx = pos,
                ty = y_offset
            )

            fig.add_shape(
                type = "path",
                path = final_path,
                fillcolor = color[letter],
                line = dict(width=0)
            )

            # hover support (invisible bar)
            fig.add_trace(go.Bar(
                x = [pos + LETTER_WIDTH / 2],
                y=[height],
                base = y_offset,
                width = LETTER_WIDTH, # width
                marker=dict(color=color, opacity = 0, line=dict(width=0)),
                hovertemplate = f"{letter}<br>{yaxis}={height:.3f}<extra></extra>",
                showlegend = False
            ))

            y_offset += height

    fig.update_layout(
        xaxis = dict(
            range=[0, len(pwm)],
            title="Position"
        ),
        yaxis = dict(
            title=yaxis
        ),
        title = title,
        template="simple_white",
        height = 600 # TODO how to decide 
    )
    
    if yaxis in ["probability", "information"]:
        fig.update_yaxes(range=[0, 1])

    return fig

def _get_letter_paths(
    letters: List[str] = ["A", "C", "G", "T", "-"],
    fontsize: int = 1, 
    fontfamily: str = "DejaVu Sans", 
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
        verts = tp.vertices
        codes = tp.codes

        path_str = []
        for (x, y), code in zip(verts, codes):
            if code == 1:  # MOVETO
                path_str.append(f"M {x} {y}")
            elif code == 2:  # LINETO
                path_str.append(f"L {x} {y}")
            elif code == 79:  # CLOSEPOLY
                path_str.append("Z")

        paths[letter] = " ".join(path_str)

    return paths

def _transform_path(
    path: str, 
    sx: float, 
    sy: float, 
    tx: float, 
    ty: float
) -> str:
    """
    Transform the path

    Parameters
    ----------
    path: str
        The path to transform.
    sx: float
        The scale factor for the x-axis.
    sy: float
        The scale factor for the y-axis.
    tx: float
        The translation factor for the x-axis.
    ty: float
        The translation factor for the y-axis.

    Returns
    -------
    str
        The transformed path.
    """
    import re

    tokens = re.split(r'([MLZ])', path)
    result = []

    for token in tokens:
        if token in ["M", "L", "Z"]:
            result.append(token)
        elif token.strip():
            coords = token.strip().split()
            new_coords = []
            for i in range(0, len(coords), 2):
                x = float(coords[i])
                y = float(coords[i + 1])
                x_new = x * sx + tx
                y_new = y * sy + ty
                new_coords.append(f"{x_new},{y_new}")
            result.append(" ".join(new_coords))

    return " ".join(result)

def _compute_information_content(
    pwm: np.ndarray,
) -> np.ndarray:
    """
    Compute the information content

    Parameters
    ----------
    pwm: np.ndarray
        The PWM matrix.

    Returns
    -------
    np.ndarray
        pwm: np.ndarray
        The PWM matrix.
    """

    ic = []
    for row in pwm:
        H = -np.sum([p * np.log2(p) if p > 0 else 0 for p in row])
        R = 2 - H
        ic.append(row * R)
    return np.array(ic)