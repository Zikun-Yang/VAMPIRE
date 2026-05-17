from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List, Tuple, Dict, Any, Literal, Optional
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

import plotly.io as pio
template = pio.templates[pio.templates.default]
DEFAULT_FONT_SIZE = (
    template.layout.font.size
    if hasattr(template, "layout") and template.layout.font
    else 12
)
DEFAULT_LINE_WIDTH = (
    template.layout.xaxis.linewidth
    if hasattr(template, "layout") and template.layout.xaxis and template.layout.xaxis.linewidth
    else 1.5
)

# Module-level colormap constants
_RAINBOW_COLORMAP: List[str] = [
    "#f94144", "#f8961e", "#f9c74f", "#90be6d", "#43aa8b", "#4d908e", "#577590", "#277da1", "#5983f2", "#898bf1",
    "#8945dc"
]
_GLASBEY_COLORMAP: List[str] = [
    "#5983f2", "#87db96", "#eef248", "#f25b76", "#f29dd3", "#38abbd", "#f2b668", "#a570f2", "#f2b668", "#288126",
    "#3e3ed1", "#48f0d1", "#987b74", "#c5c2f2", "#96ab83", "#b639b1", "#63586e", "#722222", "#428acd", "#f1eba8",
    "#154715", "#877928", "#f2948c", "#e2bebe", "#627767", "#afe0f2", "#e04392", "#a386af", "#6a2e9b", "#3bbc38",
    "#965e2d", "#945361", "#c0bc39", "#768190", "#31a68d", "#9fa3f2", "#583f41", "#361010", "#f0c8f2", "#c0dbc1",
    "#d0863f", "#96aead", "#e07ff2", "#445c7b", "#ba7b97", "#545d31", "#bba183", "#297c80", "#adf248", "#7e4288",
    "#91c0f2", "#7c6af2", "#879a2f", "#f14acf", "#b9a6bb", "#bb7266", "#705c4a", "#1e6249", "#3ccbca", "#d9ddf2",
    "#661e53", "#7877a7", "#b99c37", "#d5c8a7", "#d9acf2", "#c44af0", "#8a6080", "#982d7a", "#669364", "#c8999b",
    "#38412d", "#9ba5b9", "#868667", "#f27748", "#cdf2ee", "#bc91f2", "#84c5aa", "#c7f2a5", "#8f6aa8", "#714632",
    "#f2b399", "#a03044", "#cc3d3d", "#5a2632", "#6f8f8d", "#f287ab", "#53656e", "#719bc1", "#a17b51", "#49d2f1",
    "#37ba7f", "#5c3e59", "#9cc95b", "#bbca8f", "#645c99", "#4a2d1e", "#755f23", "#f2d175", "#923cc4", "#7bf0f2",
    "#82274f", "#745960", "#2c7995", "#a18795", "#7e7387", "#898bf1", "#b95a92", "#c25569", "#f2afc1", "#b975b3",
    "#33621e", "#28866b", "#66793b", "#9190b5", "#a4d1cc", "#c38f75", "#c7b578", "#5f1f68", "#916358", "#b3c2d0",
    "#e9f2d4", "#43e044", "#c293b8", "#6638bb", "#7d9581", "#6c6b4f", "#aaf2c4", "#376eb7", "#b6a5e0", "#589b3d",
    "#634b79", "#154837", "#8eb6c8", "#90472b", "#456861", "#ae3470", "#f29d71", "#f2d4e3", "#794460", "#24310e",
    "#986c77", "#68aba8", "#97975b", "#45754b", "#b86237", "#c6dc6b", "#8945dc", "#84b26a", "#a5bba8", "#cf7587",
    "#cd9c62", "#ef9bf2", "#4e5e4d", "#d1c4df", "#734146", "#618e9c", "#77a3f2", "#a437ba", "#deafd0", "#f2c598",
    "#434414", "#b4cdf2", "#a3aede", "#5e6f8b", "#db80b8", "#9b92a7", "#534e35", "#b17ac9", "#9988d5", "#b78082",
    "#d840d7", "#afae91", "#362e10", "#d794ad", "#a45851", "#a5b167", "#79ab8a", "#9ec79b", "#a78f60", "#266171",
    "#2b9086", "#55231a", "#607779", "#395332", "#d98468", "#5a4536", "#8266b2", "#505565", "#c490d1", "#944d8c",
    "#c9dde5", "#46b4df", "#9e7696", "#dbb641", "#88f28a", "#e5d844", "#714f22", "#5b3346", "#7f8ec0", "#829ba7",
    "#4bddae", "#afebd8", "#7d2574", "#867059", "#d76b60", "#af657c", "#77577f", "#f2d2c1", "#6b76bd", "#d6b7a1",
    "#f27be6", "#d2e3b2", "#b5764f", "#cca9b5", "#b38835", "#8d7ca9", "#f268b3", "#f2e3bd", "#3a534b", "#ae9185",
    "#6e805d", "#b89ec7", "#5ae1f1", "#985579", "#9c659f", "#732b90", "#826a78", "#636284", "#eca6a3", "#dbd094",
    "#6a6720", "#195354", "#5761f2", "#3dcd76", "#edb3f0", "#7b8e52", "#534019", "#92c9d1"
]
_SEQUENTIAL_COLORMAP: List[str] = [
    "#FED976", "#FDBA9B", "#F7958D", "#ED96C9", "#ec57e5", "#a4cae4", "#7bd1ca", "#bfde9f", "#58d581",
    "#FEBD0B", "#FC8D59", "#EF3B2C", "#DD3497", "#af14a8", "#4292C6", "#35978F", "#7FBC41", "#238B45",
    "#DEA402", "#d24504", "#91150b", "#7d1552", "#3d073a", "#204c69", "#143936", "#3f5d20", "#092512"
]
_COLORMAP_OPTIONS: Dict[str, List[str]] = {
    "rainbow": _RAINBOW_COLORMAP,
    "glasbey": _GLASBEY_COLORMAP,
    "sequential": _SEQUENTIAL_COLORMAP,
}

def _build_element_colormap(
    adata: ad.AnnData,
    feature: str = "motif",
    color: str = "id",
    colormap: dict | List | str = "rainbow",
) -> Dict[str, str]:
    """
    Build element-to-color mapping for waterfall visualization.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object.
    feature : str, default="motif"
        Type of feature.
    color : str, default="id"
        Column name in `adata.var` used to assign coloring.
    colormap : dict | list | str
        Color mapping specification.

    Returns
    -------
    Dict[str, str]
        Mapping from element (motif id or color column value) to color hex string.
    """
    all_id_list: List[str] = list(adata.var.index)
    if color == "id":
        id2element: Dict[str, Any] = {m: m for m in all_id_list}
    else:
        if color not in adata.var.columns:
            raise ValueError(f"color = '{color}' not found in adata.var.columns: {adata.var.columns}")
        id2element = dict(zip(adata.var.index, adata.var[color]))
    all_element_list: List[Any] = list(dict.fromkeys(id2element.values()))

    element_num: int = len(all_element_list)

    match colormap:
        case str():
            if colormap not in _COLORMAP_OPTIONS:
                raise ValueError(f"colormap {colormap} is not found! please select from {_COLORMAP_OPTIONS.keys()}")
            default_colormap: List[str] = _COLORMAP_OPTIONS[colormap]

            if element_num > len(default_colormap):
                logger.warning(f"Number of {color} is larger then number of colors in default colormap, using black to represent remaining motifs")
                default_colormap += ["#1a1a1a"] * (element_num - len(default_colormap))
            mapped_colormap: Dict[str, str] = dict(zip(all_element_list, default_colormap[:element_num]))

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

    return mapped_colormap


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
    track_name_dx: float = -0.07,
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
        - **draw_baseline** (`bool`, optional) - When ``True``, draws a thin black
          horizontal line across the full region before the rectangles, so gaps
          appear as breaks. Default is ``False``.

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

    track_name_dx : float, optional
        Horizontal offset applied to track name position along the x-axis,expressed as a fraction of the total width.
        Default is -0.07.

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

    .. raw:: html

        <iframe src="/_static/plots/anno/pl/trackplot.html"
                width="100%"
                height="650"
                style="border:0;">
        </iframe>
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
        linewidth = DEFAULT_LINE_WIDTH,          # axis line width
        tickwidth = DEFAULT_LINE_WIDTH,
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
            text = track["name"],
            xref = "paper",
            yref = "paper",
            x = track_name_dx,
            y = y_center,
            xanchor = "right",
            yanchor = "middle",
            showarrow = False
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

        - **draw_baseline** (`bool`, optional) — When ``True``, a thin black
          horizontal line is drawn across the full region *before* the
          rectangles, so gaps (positions with no data) appear as breaks in the
          line. Default is ``False``.
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
    sample_order: Optional[List[str]] = None,
    color: str = "id",
    colormap: dict | List | str = "rainbow",
    figsize: Tuple[Optional[int], Optional[int]] = (None, None),
    track_name_dx: float = -0.01,
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
        (alignment output from ``tl.align()``).

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

    figsize : Tuple[Optional[int], Optional[int]], optional
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
    ...     margin = dict(l=120), # show
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

    # build colormap
    mapped_colormap = _build_element_colormap(
        adata, feature=feature, color=color, colormap=colormap
    )

    motif_array_dict: Dict[str, List[str]] = adata.uns[f"{feature}_array"]
    orientation_name: str = feature.replace("motif", "orientation")
    orientation_array_dict: Dict[str, List[str]] = adata.uns[f"{orientation_name}_array"]
    for sample in sample_order:
        # get data
        motif_array: List[str] = motif_array_dict[sample]
        orientation_array: List[str] = orientation_array_dict[sample]
        array_len: int = len(motif_array)

        # skip gaps ("-") but keep original positions so alignment is preserved
        start_array: List[float] = []
        end_array: List[float] = []
        motif_filtered: List[str] = []
        ori_filtered: List[str] = []
        color_filtered: List[str] = []

        for pos, (m, o) in enumerate(zip(motif_array, orientation_array)):
            if m == "-":
                continue
            start_array.append(float(pos))
            end_array.append(float(pos + 1))
            motif_filtered.append(m)
            ori_filtered.append(o)
            color_filtered.append(mapped_colormap[m])

        if end_array:
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
        track_dict = {
            "name": sample,
            "type": "bed",
            "data": track_data,
        }
        track_list.append(track_dict)

    # auto-compute figsize to avoid crowding or excessive sparsity
    n_tracks = len(track_list)

    # get real font size: user override > vampire template > plotly default
    font_size = kwargs.get("font", {}).get("size", DEFAULT_FONT_SIZE) if isinstance(kwargs.get("font"), dict) else DEFAULT_FONT_SIZE

    # width scales with max_x and font size
    if figsize[0] is None:
        # each motif/kmer needs enough px to be distinguishable;
        # larger font needs proportionally wider items
        px_per_item = font_size * 0.35
        # account for margins: left (max name length * 8, ~80) + right (40) + padding
        min_margin = 120
        width = int(max_x * px_per_item + min_margin)
        width = max(width, 500)
        logger.debug(f"automatically adjust figure width to {width}")
    else:
        width = figsize[0]

    # height scales with n_tracks and font size
    if figsize[1] is None:
        # each track needs enough vertical space for bed rectangles and annotations;
        # tighter than before to avoid excessive sparsity
        min_track_height = font_size * 1.6
        # margins in trackplot: top=40, bottom=55
        height = int(n_tracks * min_track_height + 40 + 55)
        height = max(height, 300)
        logger.debug(f"automatically adjust figure height to {height}")
    else:
        height = figsize[1]

    actual_figsize = (width, height)

    # Detect whether any sample contains gaps ("-") — only draw baselines for
    # aligned data where gaps need to be visualised as breaks in the line.
    has_gap = any(
        any(m == "-" for m in motif_array_dict[s])
        for s in sample_order
    )
    if has_gap:
        for td in track_list:
            td["draw_baseline"] = True

    fig: go.Figure = trackplot(
        tracks = track_list,
        region = f"seq:0-{max_x}",
        title = "",
        x_title = "Copy index",
        figsize = actual_figsize,
        vertical_spacing = 0.00,
        track_name_dx = track_name_dx,
        **kwargs
    )

    return fig

def waterfall_legend(
    adata: ad.AnnData,
    feature: str = "motif",
    sample_order: Optional[List[str]] = None,
    color: str = "id",
    colormap: dict | List | str = "rainbow",
    figsize: Tuple[Optional[int], Optional[int]] = (None, None),
    track_name_dx: float = -0.01,
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
        Key prefix passed to `_build_element_colormap`. Must match the
        ``feature`` argument used in the corresponding `waterfall()` call.

    sample_order : list of str, optional
        Unused in legend, kept for API consistency with `waterfall()`.

    color : str, default="id"
        Column name in `adata.var` used to assign coloring. When ``color="id"``,
        legend labels show motif ids; otherwise labels show values from the
        specified column.

    colormap : dict | list | str
        Color mapping specification. Must match the colormap used in the
        corresponding `waterfall()` call for consistent coloring.

    figsize : Tuple[Optional[int], Optional[int]], optional
        Figure size as (width, height) in pixels. Default is (None, None).

        - (None, None): auto-compute both dimensions from data.
        - (w, None): fixed width, auto-compute height from element count.
        - (None, h): fixed height, auto-compute width from label length.
        - (w, h): use user-specified size.

    track_name_dx: float, optional
        Unused in legend, kept for API consistency with `waterfall()`.

    **kwargs
        Additional keyword arguments passed to Plotly `update_layout`.

    Returns
    -------
    fig : go.Figure
        Plotly figure object with colored squares and their labels.

    Examples
    --------
    >>> vp.anno.pl.waterfall_legend(
    ...     adata,
    ...     color = "id",
    ...     colormap = "rainbow",
    ...     figsize = (300, 400),
    ... )
    """
    import plotly.graph_objects as go

    # build colormap (same logic as waterfall)
    mapped_colormap = _build_element_colormap(
        adata, feature=feature, color=color, colormap=colormap
    )

    fig = go.Figure()

    n_items = len(mapped_colormap)
    if n_items == 0:
        return fig

    max_label_len = max(len(str(k)) for k in mapped_colormap.keys())

    gap_length: float = 0.1
    y_pos_list: List[float] = []
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
        fig.add_trace(go.Scatter(
            x=[1.5],
            y=[y_pos + 0.5],
            text=[str(element)],
            mode="text",
            textposition="middle right",
            textfont=dict(size=14),
            showlegend=False,
            hoverinfo="skip",
        ))

    ymax = max(y_pos_list) + 1
    ymin = min(y_pos_list)

    # auto-compute figsize
    if figsize[0] is None:
        width = max(200, 60 + max_label_len * 9)
    else:
        width = figsize[0]

    if figsize[1] is None:
        height = max(160, n_items * 40 + 30)
    else:
        height = figsize[1]

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
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_layout(**kwargs)

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
        width = figsize[0],
        height = figsize[1]
    )

    fig.update_xaxes(showline=True, linecolor="black", ticks="outside")
    fig.update_yaxes(showline=True, linecolor="black", ticks="outside")

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

"""
#
# heatmap function, including motif_abundance_heatmap
#
"""
def heatmap_from_matrix(
    matrix: "np.ndarray",
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    standard_scale: Optional[Literal["obs", "var", "zscore_obs", "zscore_var"]] = None,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    row_cluster_method: str = "average",
    col_cluster_method: str = "average",
    row_cluster_metric: str = "euclidean",
    col_cluster_metric: str = "euclidean",
    dendrogram_ratio: float = 0.15,
    colorscale: str | List | None = None,
    showticklabels: bool = True,
    figsize: Tuple[int, int] = (800, 600),
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
    colorbar_title: str = "Value",
    hover_template: str = "Row: %{y}<br>Col: %{x}<br>Value: %{hovertext}<extra></extra>",
    row_annotation: Optional[List[str]] = None,
    col_annotation: Optional[List[str]] = None,
    annotation_palette: Optional[Dict[str, str]] = None,
    annotation_ratio: float = 0.03,
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
    row_labels : list of str, optional
        Labels for rows.  If ``None``, integer indices are used.
    col_labels : list of str, optional
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
    dendrogram_ratio : float, default=0.15
        Fraction of the figure allocated to dendrograms.
    colorscale : str or list, optional
        Plotly colorscale.  Default is ``"Plasma"``.
    showticklabels : bool, default=True
        Whether to display row and column tick labels.
    vmax : float, optional
        Upper bound for clipping the heatmap color scale.
        Values above ``vmax`` are clipped for visualization only;
        the original values are still shown on hover.
    vmin : float, optional
        Lower bound for clipping the heatmap color scale.
        Values below ``vmin`` are clipped for visualization only.
    figsize : Tuple[int, int], default=(800, 600)
        Figure size in pixels.
    colorbar_title : str, default="Value"
        Title shown next to the color bar.
    hover_template : str, optional
        Plotly hover template for the heatmap trace.
        Use ``%{text}`` to reference the un-clipped original value.
    row_annotation : list of str, optional
        Categorical label for each row.  Displayed as a coloured
        sidebar between the row dendrogram and the heatmap.
    col_annotation : list of str, optional
        Categorical label for each column.  Displayed as a coloured
        bar between the column dendrogram and the heatmap.
    annotation_palette : dict, optional
        Mapping from annotation category to colour hex string.
        If ``None``, colours are auto-generated from the Glasbey
        palette.
    annotation_ratio : float, default=0.03
        Fraction of the figure width/height allocated to each
        annotation block.
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
    if row_annotation is not None and len(row_annotation) != n_rows:
        raise ValueError("row_annotation length must match matrix row count")
    if col_annotation is not None and len(col_annotation) != n_cols:
        raise ValueError("col_annotation length must match matrix column count")

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
    _is_dist = (
        n_rows == n_cols
        and np.allclose(np.diag(X), 0)
        and np.allclose(X, X.T)
        and np.all(X >= 0)
    )

    if _is_dist and (cluster_rows or cluster_cols) and n_rows > 1:
        from scipy.spatial.distance import squareform
        row_linkage = linkage(
            squareform(X, checks=False),
            method=row_cluster_method,
        )
        row_dendro_data = dendrogram(
            row_linkage, no_plot=True, color_threshold=0,
            above_threshold_color="#000000",
        )
        col_dendro_data = row_dendro_data
        row_order = row_dendro_data["leaves"]
        col_order = row_order
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

    row_annotation_reordered = None
    if row_annotation is not None:
        row_annotation_reordered = [row_annotation[i] for i in row_order]
    col_annotation_reordered = None
    if col_annotation is not None:
        col_annotation_reordered = [col_annotation[i] for i in col_order]

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
    _DEFAULT_COLORSCALE = [
        [0.0, "rgb(33, 102, 172)"],
        [0.5, "rgb(255, 255, 255)"],
        [1.0, "rgb(178, 34, 34)"],
    ] if (standard_scale is not None and "zscore" in standard_scale) and (X.min() < 0) else [
        [0.0, "rgb(255, 255, 255)"],
        [1.0, "rgb(178, 34, 34)"],
    ]
    _colorscale = colorscale if colorscale is not None else _DEFAULT_COLORSCALE
    fig.add_trace(go.Heatmap(
        z=X_display.tolist(),
        hovertext=[[f"{v:.4g}" for v in row] for row in X_reordered],
        x=list(range(n_cols)),
        y=list(range(n_rows)),
        colorscale=_colorscale,
        showscale=True,
        colorbar=dict(
            orientation="h",
            y=-0.05,
            yanchor="top",
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
            width=DEFAULT_LINE_WIDTH,
        ),
        fillcolor="rgba(0,0,0,0)",
        layer="above",
    )

    # Annotation blocks
    def _get_annotation_palette(labels, palette=None):
        if palette is None:
            unique = sorted(set(labels))
            palette = {cat: _GLASBEY_COLORMAP[i % len(_GLASBEY_COLORMAP)] for i, cat in enumerate(unique)}
        return [palette.get(l, "#cccccc") for l in labels], palette

    row_palette = {}
    col_palette = {}

    if row_annotation_reordered is not None:
        row_colors, row_palette = _get_annotation_palette(row_annotation_reordered, annotation_palette)
        fig.add_trace(go.Bar(
            x=[1] * n_rows,
            y=list(range(n_rows)),
            marker=dict(color=row_colors),
            orientation="h",
            width=1,
            showlegend=False,
            hoverinfo="skip",
            xaxis="x3",
            yaxis="y",
        ))

    if col_annotation_reordered is not None:
        col_colors, col_palette = _get_annotation_palette(col_annotation_reordered, annotation_palette)
        fig.add_trace(go.Bar(
            x=list(range(n_cols)),
            y=[1] * n_cols,
            marker=dict(color=col_colors),
            orientation="v",
            width=1,
            showlegend=False,
            hoverinfo="skip",
            xaxis="x",
            yaxis="y3",
        ))

    # Domain layout — space allocation follows cluster_* parameters exactly.
    # Both dendrogram panels share the same pixel size so their canvas heights
    # are visually identical.
    _panel_px = dendrogram_ratio * min(figsize[0], figsize[1])
    annot_px = annotation_ratio * min(figsize[0], figsize[1])

    x_dendro_w = _panel_px / figsize[0] if cluster_rows else 0.0
    x_annot_w = annot_px / figsize[0] if row_annotation_reordered is not None else 0.0
    x_heatmap_left = x_dendro_w + x_annot_w

    y_dendro_h = _panel_px / figsize[1] if cluster_cols else 0.0
    y_annot_h = annot_px / figsize[1] if col_annotation_reordered is not None else 0.0
    y_heatmap_top = 1.0 - y_dendro_h - y_annot_h

    # Add legend entries for annotation categories (invisible scatter traces)
    if row_palette or col_palette:
        merged_palette = {**row_palette, **col_palette}
        for cat, color in merged_palette.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                legendgroup=cat,
                showlegend=True,
                name=cat,
            ))

    fig.update_layout(
        xaxis=dict(
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
        xaxis2=dict(
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
        xaxis3=dict(
            domain=[x_dendro_w, x_heatmap_left],
            range=[0, 1],
            showticklabels=False,
            showline=False,
            automargin=False,
            mirror=False,
            showgrid=False,
            zeroline=False,
            ticks="",
        ),
        yaxis=dict(
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
        yaxis2=dict(
            domain=[y_heatmap_top + y_annot_h, 1],
            range=[0, col_max_dist],
            showticklabels=False,
            showline=False,
            automargin=False,
            mirror=False,
            showgrid=False,
            zeroline=False,
            ticks="",
        ),
        yaxis3=dict(
            domain=[y_heatmap_top, y_heatmap_top + y_annot_h],
            range=[0, 1],
            showticklabels=False,
            showline=False,
            automargin=False,
            mirror=False,
            showgrid=False,
            zeroline=False,
            ticks="",
        ),
        bargap=0,
        width=figsize[0],
        height=figsize[1],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            x=0.5,
            xanchor="center",
        ),
        margin=dict(l=80, r=120, t=100, b=120),
        **kwargs,
    )

    return fig

def motif_abundance_heatmap(
    adata: "ad.AnnData",
    layer: Optional[str] = None,
    standard_scale: Optional[Literal["obs", "var", "zscore_obs", "zscore_var"]] = "obs",
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    row_cluster_method: str = "average",
    col_cluster_method: str = "average",
    row_cluster_metric: str = "euclidean",
    col_cluster_metric: str = "euclidean",
    dendrogram_ratio: float = 0.15,
    colorscale: str | List | None = None,
    showticklabels: bool = True,
    figsize: Tuple[int, int] = (800, 600),
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
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
    dendrogram_ratio : float, default=0.15
        Fraction of the figure allocated to dendrograms.
    colorscale : str or list, optional
        Plotly colorscale name.  Default is ``"Plasma"``.
    showticklabels : bool, default=True
        Whether to display row and column tick labels.
    figsize : Tuple[int, int], default=(800, 600)
        Figure size in pixels.
    vmax : float, optional
        Upper bound for clipping the heatmap color scale.
        Values above ``vmax`` are clipped for visualization only.
    vmin : float, optional
        Lower bound for clipping the heatmap color scale.
        Values below ``vmin`` are clipped for visualization only.
    **kwargs
        Additional keyword arguments passed to
        ``fig.update_layout()``.

    Returns
    -------
    go.Figure
        A Plotly figure containing the clustered heatmap with
        dendrograms.

    Examples
    --------
    >>> import vampire as vp
    >>> adata = vp.anno.pp.read_anno("results.annotation.tsv")
    >>> fig = vp.anno.pl.motif_abundance_heatmap(
    ...     adata,
    ...     cluster_rows=True,
    ...     cluster_cols=True,
    ...     standard_scale="obs",
    ...     figsize=(1000, 800),
    ... )
    """
    # Extract data matrix
    X = adata.X if layer is None else adata.layers[layer]
    if hasattr(X, "toarray"):
        X = X.toarray()

    row_labels = [str(l) for l in adata.obs.index]
    col_labels = [str(l) for l in adata.var.index]

    return heatmap_from_matrix(
        matrix=X,
        row_labels=row_labels,
        col_labels=col_labels,
        standard_scale=standard_scale,
        cluster_rows=cluster_rows,
        cluster_cols=cluster_cols,
        row_cluster_method=row_cluster_method,
        col_cluster_method=col_cluster_method,
        row_cluster_metric=row_cluster_metric,
        col_cluster_metric=col_cluster_metric,
        dendrogram_ratio=dendrogram_ratio,
        colorscale=colorscale,
        showticklabels=showticklabels,
        figsize=figsize,
        vmax=vmax,
        vmin=vmin,
        colorbar_title="Abundance",
        hover_template="Sample: %{y}<br>Motif: %{x}<br>Value: %{hovertext}<extra></extra>",
        **kwargs,
    )

"""
#
# haplotype clustering evaluation plot
#
"""
def haplotype_evaluation(
    adata: ad.AnnData,
    store_key: str = "haplotype",
) -> go.Figure:
    """
    Plot silhouette score curve for haplotype cluster evaluation.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with evaluation results from ``tl.haplotype()``
        (run with ``n_clusters=None``).
    store_key : str, default="haplotype"
        Key prefix matching the ``store_key`` used in ``tl.haplotype()``.

    Returns
    -------
    go.Figure
        Plotly figure with the silhouette score curve.

    Examples
    --------
    >>> import vampire as vp
    >>> adata = vp.anno.tl.haplotype(adata)
    >>> fig = vp.anno.pl.haplotype_evaluation(adata)
    """
    import plotly.graph_objects as go

    eval_data = adata.uns.get(f"{store_key}_evaluation")
    if eval_data is None:
        raise KeyError(
            f"Evaluation data not found at uns['{store_key}_evaluation']. "
            f"Run haplotype() with n_clusters=None first."
        )

    k_range = eval_data["k_range"]
    scores = eval_data["silhouette"]
    best_k = eval_data["best_k"]
    best_score = eval_data["best_score"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=k_range,
            y=scores,
            mode="lines+markers",
            name="Silhouette Score",
            line=dict(color="#212529", width=2),
            marker=dict(size=8),
        )
    )

    if best_k in k_range:
        best_idx = k_range.index(best_k)
        fig.add_trace(
            go.Scatter(
                x=[best_k],
                y=[scores[best_idx]],
                mode="markers",
                name=f"Best k={best_k}",
                marker=dict(color="#f94144", size=14, symbol="star"),
            )
        )

    fig.add_hline(
        y=0.25,
        line_dash="dash",
        line_color="gray",
        annotation_text="threshold (0.25)",
        annotation_position="top right",
    )

    title_text = "Haplotype Cluster Evaluation"
    if best_k == 1:
        title_text += f" <br><span style='font-size:12px;color:gray'>"
        title_text += f"Weak structure (best silhouette {best_score:.3f}) — assigned to 1 cluster"
        title_text += "</span>"

    fig.update_layout(
        title=title_text,
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        showlegend=True,
    )

    return fig

def haplotype_distance_heatmap(
    adata: "ad.AnnData",
    store_key: str = "haplotype",
    metric: str = "structural",
    reorder: bool = True,
    cluster: bool = False,
    figsize: Tuple[int, int] = (800, 700),
    colorscale: str | List | None = None,
    **kwargs,
) -> "go.Figure":
    """Plot sample pairwise distance matrix from haplotype analysis.

    Visualises one of the distance matrices stored in ``obsp`` by
    ``tl.haplotype_neighbor()``.  Samples are annotated by their haplotype
    assignment so that the block structure is visible.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with haplotype results from ``tl.haplotype_neighbor()``.
    store_key : str, default="haplotype"
        Key prefix matching ``store_key`` used in ``tl.haplotype_neighbor()``.
    metric : str, default="structural"
        Which distance matrix to visualise.  Options:
        ``"structural"``, ``"cnv"``.
    reorder : bool, default=True
        If ``True``, rows and columns are sorted by haplotype label
        so that samples from the same haplotype are adjacent.
    cluster : bool, default=False
        If ``True``, hierarchically cluster rows and columns
        (overrides ``reorder``).
    figsize : Tuple[int, int], default=(800, 700)
        Figure size in pixels.
    colorscale : str | List | None, default=None
        Plotly colorscale for the heatmap. If ``None``, defaults to a red-to-white scale.
    **kwargs
        Additional arguments passed to ``heatmap_from_matrix``.

      Returns
      -------
      go.Figure
          Plotly figure with the distance matrix heatmap.

      Examples
      --------
      >>> import vampire as vp
      >>> adata = vp.anno.tl.haplotype_neighbor(adata)
      >>> fig = vp.anno.pl.haplotype_distance_heatmap(adata)
      """
    import numpy as np

    metric_key = f"{store_key}_{metric}_distance"
    dist_mat = adata.obsp.get(metric_key)
    if dist_mat is None:
        raise KeyError(
            f"Distance matrix not found at obsp['{metric_key}']. "
            f"Run tl.haplotype_neighbor() first."
        )

    labels = adata.obs[store_key]
    names = list(adata.obs_names)

    if reorder and not cluster:
        sort_idx = np.argsort(labels.astype(str))
        dist_mat = dist_mat[np.ix_(sort_idx, sort_idx)]
        names = [names[i] for i in sort_idx]
        labels = labels.iloc[sort_idx] if hasattr(labels, "iloc") else labels[sort_idx]

    _DEFAULT_COLORSCALE = [
        [0.0, "rgb(178, 34, 34)"],
        [1.0, "rgb(255, 255, 255)"],
    ]

    return heatmap_from_matrix(
        matrix=dist_mat,
        row_labels=names,
        col_labels=None,
        cluster_rows=cluster,
        cluster_cols=cluster,
        colorscale=colorscale or _DEFAULT_COLORSCALE,
        figsize=figsize,
        colorbar_title="Distance",
        row_annotation=list(labels),
        hover_template="Sample: %{y}<br>Sample: %{x}<br>Distance: %{hovertext}<extra></extra>",
        **kwargs,
    )


"""
#
# motif abundance PCA plot
#
"""
def motif_abundance_pca(
    adata: "ad.AnnData",
    color_by: Optional[str] = None,
    shape_by: Optional[str] = None,
    components: Tuple[int, int] = (1, 2),
    figsize: Tuple[int, int] = (600, 600),
    title: Optional[str] = None,
    marker_size: int = 10,
    colorscale: Optional[str] = None,
    show_variance: bool = True,
    **kwargs,
) -> "go.Figure":
    """Plot pairwise principal components from motif abundance PCA.

    Reads pre-computed PCA results stored by ``tl.motif_abundance_pca()``.
    Color and marker shape can be mapped to columns in ``adata.obs``.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with PCA results from ``tl.motif_abundance_pca()``.
    color_by : str, optional
        Column in ``adata.obs`` for marker color.  Categorical columns use
        a discrete palette; numeric columns use a continuous colorscale.
    shape_by : str, optional
        Column in ``adata.obs`` for marker shape.  Must be categorical.
    components : Tuple[int, int], default=(1, 2)
        Which two PCs to plot.  1-based indexing, e.g. ``(1, 2)`` for PC1
        vs PC2, ``(2, 3)`` for PC2 vs PC3.
    figsize : Tuple[int, int], default=(600, 600)
        Figure size in pixels.
    title : Optional[str], default=None
        Plot title.
    marker_size : int, default=10
        Marker size.
    colorscale : str, optional
        Plotly colorscale name for numeric ``color_by``.  Defaults to
        ``"Viridis"``.
    show_variance : bool, default=True
        Append explained-variance percentages to axis titles.
    **kwargs
        Additional keyword arguments passed to ``fig.update_layout()``.

    Returns
    -------
    go.Figure
        Plotly scatter figure of the chosen PCs.

    Examples
    --------
    >>> import vampire as vp
    >>> adata = vp.anno.tl.motif_abundance_pca(adata)
    >>> fig = vp.anno.pl.motif_abundance_pca(adata, shape_by="ancestry", color_by="copy_number")
    >>> fig = vp.anno.pl.motif_abundance_pca(adata, components=(1, 2))
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
    color_map: Optional[Dict[str, str]] = None
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
    shape_map: Optional[Dict[str, str]] = None
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
                    colorscale=colorscale or "Viridis",
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
                    colorscale=colorscale or "Viridis",
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
        width=figsize[0],
        height=figsize[1],
        legend=dict(orientation="h", yanchor="top", y=-0.15,
                    xanchor="center", x=0.5),
        **kwargs,
    )
    fig.update_xaxes(showline=True, linecolor="black", ticks="outside")
    fig.update_yaxes(showline=True, linecolor="black", ticks="outside")
    return fig


def motif_abundance_pca_variance(
    adata: "ad.AnnData",
    n_pcs: Optional[int] = None,
    log: bool = False,
    show_cumulative: bool = True,
    figsize: Tuple[int, int] = (700, 600),
    title: Optional[str] = None,
    **kwargs,
) -> "go.Figure":
    """Plot variance explained by each principal component.

    Reads pre-computed results from ``tl.motif_abundance_pca()`` stored in
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
    figsize : Tuple[int, int], default=(700, 500)
        Figure size in pixels.
    title : Optional[str], default=None
        Plot title.
    **kwargs
        Additional keyword arguments passed to ``fig.update_layout()``.

    Returns
    -------
    go.Figure
        Bar + line plot of per-PC variance ratios.

    Examples
    --------
    >>> import vampire as vp
    >>> fig = vp.anno.pl.motif_abundance_pca_variance(adata)
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
        yaxis=dict(title="Explained Variance Ratio", type=yaxis_type),
        width=figsize[0],
        height=figsize[1],
        legend=dict(orientation="h", yanchor="top", y=-0.15,
                    xanchor="center", x=0.5),
    )

    fig.update_layout(**layout, **kwargs)
    fig.update_xaxes(showline=True, linecolor="black", ticks="outside")
    fig.update_yaxes(showline=True, linecolor="black", ticks="outside")
    return fig


"""
#
# copy number distribution violin plot
#
"""
def copy_number_violin(
    adata: "ad.AnnData",
    group_by: str,
    motif: str | int | None = None,
    show_box: bool = True,
    show_points: bool = False,
    figsize: Tuple[int, int] = (500, 500),
    **kwargs,
) -> "go.Figure":
    """Plot copy-number distribution across sample groups as a violin plot.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with copy-number information.
    group_by : str
        Column name in ``adata.obs`` used to group samples.
    motif : str | int | None, default=None
        If ``None``, the total copy number per sample
        (``adata.obs["copy_number"]``) is used.
        If ``str``, the motif is looked up in ``adata.var.index`` first,
        then in ``adata.var["motif"]``, and the matching column from
        ``adata.X`` is used.
        If ``int``, ``adata.X[:, motif]`` is used directly.
    show_box : bool, default=True
        Whether to overlay a mini box plot inside each violin.
    show_points : bool, default=False
        Whether to overlay individual data points on each violin.
    figsize : Tuple[int, int], default=(600, 400)
        Figure size in pixels.
    **kwargs
        Additional arguments passed to ``fig.update_layout()``.

    Returns
    -------
    go.Figure
        Plotly figure with the violin plot.

    Examples
    --------
    >>> import vampire as vp
    >>> fig = vp.anno.pl.copy_number_violin(adata, group_by="haplotype")
    >>> fig = vp.anno.pl.copy_number_violin(adata, group_by="ancestry", motif="ACGT")
    """
    import plotly.graph_objects as go
    import numpy as np

    if group_by not in adata.obs.columns:
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

    groups = sorted(adata.obs[group_by].dropna().unique(), key=str)
    if len(groups) == 0:
        raise ValueError(f"No valid groups found in adata.obs['{group_by}']")

    fig = go.Figure()
    for i, group in enumerate(groups):
        mask = (adata.obs[group_by] == group).to_numpy()
        group_y = y[mask]
        x_vals = [str(group)] * len(group_y)
        color = _RAINBOW_COLORMAP[i % len(_RAINBOW_COLORMAP)]
        trace_kwargs = dict(
            x=x_vals,
            y=group_y,
            name=str(group),
            fillcolor=color,
            line=dict(color=color, width=1.2),
            opacity=0.8,
            width=0.7,
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
        xaxis_title=group_by,
        yaxis_title=y_title,
        width=figsize[0],
        height=figsize[1],
        violinmode="group",
        legend=dict(orientation="h", yanchor="top", y=-0.2,
                    xanchor="center", x=0.5),
        margin=dict(b=100),
    )
    fig.update_layout(**layout, **kwargs)
    fig.update_xaxes(showline=True, linecolor="black", ticks="outside")
    fig.update_yaxes(showline=True, linecolor="black", ticks="outside")
    return fig


"""
#
# copy number stacked violin plot
#
"""
def copy_number_stacked_violin(
    adata: "ad.AnnData",
    group_by: str,
    motifs: str | Sequence[str] | None = None,
    log: bool = False,
    categories_order: Sequence[str] | None = None,
    colorscale: str | Sequence[str] | None = None,
    show_box: bool = False,
    show_points: bool = False,
    row_height: int = 80,
    figsize: Tuple[int, int] | None = None,
    **kwargs,
) -> "go.Figure":
    """Plot copy-number distributions for multiple motifs as stacked violins.

    Each row corresponds to one motif; each column corresponds to a group
    defined by ``group_by``.  Useful for comparing copy-number variation
    across motifs and sample groups simultaneously.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with motif copy-number matrix in ``X``.
    group_by : str
        Column name in ``adata.obs`` used to group samples.
    motifs : str | Sequence[str] | None, default=None
        Motif(s) to visualise.  If ``None``, all motifs in ``adata`` are
        used.  If more than 30 motifs are selected a warning is emitted.
        A single ``str`` or a list/sequence of motif IDs / sequences is
        accepted.
    log : bool, default=False
        Whether to apply ``log1p`` transform to copy-number values before
        plotting.
    categories_order : Sequence[str] | None, default=None
        Explicit order for the groups on the x-axis.  If ``None``, groups
        are sorted alphabetically.
    colorscale : str | Sequence[str] | None, default=None
        Colormap for the median-based violin fill.  If a ``str``, it is
        passed to ``plotly.colors.sample_colorscale`` (e.g. ``"Viridis"``,
        ``"Plasma"``).  If a sequence of hex/rgb strings, used directly.
        If ``None``, the module default sequential colormap is used.
    show_box : bool, default=True
        Whether to overlay a mini box plot inside each violin.
    show_points : bool, default=False
        Whether to overlay individual data points on each violin.
    row_height : int, default=80
        Height in pixels allocated to each motif row.
    figsize : Tuple[int, int] | None, default=None
        Figure size ``(width, height)`` in pixels.  If ``None``, computed
        as ``(800, n_motifs * row_height + 120)``.
    **kwargs
        Additional arguments passed to ``fig.update_layout()``.

    Returns
    -------
    go.Figure
        Plotly figure with stacked violin plots.

    Examples
    --------
    >>> import vampire as vp
    >>> fig = vp.anno.pl.copy_number_stacked_violin(adata, group_by="haplotype")
    >>> fig = vp.anno.pl.copy_number_stacked_violin(
    ...     adata, group_by="ancestry", motifs=["ACGT", "TGCA"]
    ... )
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    if group_by not in adata.obs.columns:
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
    motif_indices: List[int] = []
    motif_labels: List[str] = []
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
    all_groups = set(adata.obs[group_by].dropna().unique())
    if categories_order is not None:
        groups = list(categories_order)
        missing = set(groups) - all_groups
        if missing:
            raise ValueError(f"categories_order contains unknown groups: {missing}")
    else:
        groups = sorted(all_groups, key=str)
    if len(groups) == 0:
        raise ValueError(f"No valid groups found in adata.obs['{group_by}']")

    n_motifs = len(motif_indices)
    n_groups = len(groups)

    if figsize is None:
        figsize = (700, n_motifs * row_height + 120)

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    if log:
        X = np.log1p(X)

    # Resolve colours
    if colorscale is None:
        color_palette = _SEQUENTIAL_COLORMAP
    elif isinstance(colorscale, str):
        try:
            from plotly.colors import sample_colorscale
            color_palette = sample_colorscale(colorscale, np.linspace(0, 1, 256))
        except Exception:
            color_palette = _SEQUENTIAL_COLORMAP
    else:
        color_palette = list(colorscale)
    n_colors = len(color_palette)
    plotly_colorscale = [[i / (n_colors - 1), color_palette[i]] for i in range(n_colors)]

    # Compute per-motif, per-group medians for colour mapping
    medians = np.zeros((n_motifs, n_groups), dtype=float)
    for row_idx, idx in enumerate(motif_indices):
        y_all = X[:, idx]
        for col_idx, group in enumerate(groups):
            mask = (adata.obs[group_by] == group).to_numpy()
            medians[row_idx, col_idx] = np.median(y_all[mask])

    # Normalise medians row-wise (per motif)
    median_norm = np.zeros_like(medians)
    for row_idx in range(n_motifs):
        vmin, vmax = medians[row_idx].min(), medians[row_idx].max()
        if vmax > vmin:
            median_norm[row_idx] = (medians[row_idx] - vmin) / (vmax - vmin)

    fig = make_subplots(
        rows=n_motifs,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
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
            line=dict(width=1),
        )

        # label
        fig.add_annotation(
            x=-0.02,
            y=y_center,
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=DEFAULT_FONT_SIZE),
            xanchor="right",
            yanchor="middle",
            textangle=0,
        )

    for row_idx, idx in enumerate(motif_indices, start=1):
        y_all = X[:, idx]
        for col_idx, group in enumerate(groups):
            mask = (adata.obs[group_by] == group).to_numpy()
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

    # Global bounding box around the whole plot area
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        xref="paper",
        yref="paper",
        line=dict(color="black", width=DEFAULT_LINE_WIDTH),
        fillcolor="rgba(0,0,0,0)",
        layer="above",
    )

    # Colour-bar legend for the median-based gradient
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale=plotly_colorscale,
            showscale=True,
            cmin=0,
            cmax=1,
            colorbar=dict(
                title="Median CN",
                thickness=15,
                len=0.5,
                yanchor="top",
                y=1,
                x=1.02,
            ),
        ),
        showlegend=False,
        hoverinfo="skip",
    ))

    layout = dict(
        width=figsize[0],
        height=figsize[1],
        violinmode="group",
        margin=dict(b=80, t=60, l=120),
    )
    fig.update_layout(**layout, **kwargs)
    return fig