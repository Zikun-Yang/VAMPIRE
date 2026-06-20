"""Centralized figure-size calculation utilities for vampire plotting.

All pixel constants are calibrated at ``font_size = 14`` and scale
proportionally with the active font size so that labels remain readable
regardless of styling choices.

This module is the foundation (Phase 1) for auto-sizing support.  Plot
functions still use their historical fixed defaults; individual functions
will be migrated to call these utilities in Phase 2.
"""

from __future__ import annotations
from typing import Callable

import logging

logger = logging.getLogger(__name__)


"""
#
# internal helpers
#
"""
# Absolute minimum figure width and height in pixels.
MIN_WIDTH: int = 300
MIN_HEIGHT: int = 200
def _scale(value: float, font_size: int) -> float:
    """
    Scale a baseline pixel value by ``font_size / BASE_FONT_SIZE``.

    Parameters
    ----------
    value
        Baseline pixel value (calibrated at :data:`BASE_FONT_SIZE`).
    font_size
        Target font size.

    Returns
    -------
    float
        Scaled pixel value.
    """
    return value * font_size / BASE_FONT_SIZE

BASE_FONT_SIZE: int = 14
def get_active_font_size() -> int:
    """
    Return the font size from the currently active Plotly template.

    Unlike a module-level constant, this reads the *current* default
    template every time it is called, so it stays in sync with
    ``set_default_plotstyle()`` calls made after import.

    Returns
    -------
    int
        Font size in points. Falls back to :data:`BASE_FONT_SIZE` if the
        active template does not declare one.
    """
    import plotly.io as pio

    template = pio.templates[pio.templates.default]
    if (
        hasattr(template, "layout")
        and template.layout.font
        and template.layout.font.size is not None
    ):
        return template.layout.font.size
    return BASE_FONT_SIZE


def get_active_line_width() -> float:
    """
    Return the axis line width from the currently active Plotly template.

    Reads the *current* default template dynamically, matching the behaviour
    of :func:`get_active_font_size`.

    Returns
    -------
    float
        Line width in pixels. Falls back to ``1.5`` if the active template
        does not declare one.
    """
    import plotly.io as pio

    template = pio.templates[pio.templates.default]
    if (
        hasattr(template, "layout")
        and template.layout.xaxis
        and template.layout.xaxis.linewidth is not None
    ):
        return template.layout.xaxis.linewidth
    return 1.5


"""
#
# core resolver
#
"""
def resolve_figsize(
    width: int | None,
    height: int | None,
    *,
    calc_width: Callable[[], int] | None = None,
    calc_height: Callable[[], int] | None = None,
    min_width: int = MIN_WIDTH,
    min_height: int = MIN_HEIGHT,
) -> tuple[int, int]:
    """
    Resolve a potentially partial figsize into absolute ``(width, height)``.

    If *both* dimensions are supplied (non-``None``) the function simply
    enforces the minimum-size floors and returns them — this is the
    zero-overhead path for fully user-specified sizes.

    Parameters
    ----------
    width
        Desired width in pixels, or ``None`` to trigger auto-computation.
    height
        Desired height in pixels, or ``None`` to trigger auto-computation.
    calc_width
        Callable returning the auto-computed width. **Required** when
        ``width is None``.
    calc_height
        Callable returning the auto-computed height. **Required** when
        ``height is None``.
    min_width
        Floor for width.
    min_height
        Floor for height.

    Returns
    -------
    tuple[int, int]
        Resolved ``(width, height)`` in pixels.

    Raises
    ------
    ValueError
        If a dimension is ``None`` but the corresponding ``calc_*`` callable
        is not provided.
    """
    if width is None:
        if calc_width is None:
            raise ValueError("width is None but no calc_width provided")
        width = calc_width()
    if height is None:
        if calc_height is None:
            raise ValueError("height is None but no calc_height provided")
        height = calc_height()

    resolved = (max(int(width), min_width), max(int(height), min_height))
    logger.debug("resolved figsize: %s", resolved)
    return resolved


"""
#
# waterfall function
#
"""
WATERFALL_MIN_WIDTH: int = 200
WATERFALL_MIN_HEIGHT: int = 200
WATERFALL_PX_PER_ITEM: float = 0.35
WATERFALL_TOP_MARGIN: int = 40
WATERFALL_BOTTOM_MARGIN: int = 55
WATERFALL_MIN_MARGIN: int = 120
WATERFALL_PX_PER_TRACK: float = 1.3

def waterfall_width(
    n_items: int,
    font_size: int = BASE_FONT_SIZE,
    max_name_length: int = 0,
    annotation_width_px: int = 0,
) -> tuple[int, int]:
    """
    Compute waterfall width from item count.

    The returned width accounts for the content area (``n_items`` at
    ``font_size * 0.35`` px each) plus the overhead that ``tracksplot``
    subtracts for track-name labels (``max_name_length * 8 + 40`` px).
    When row annotations are present, ``annotation_width_px`` is added to
    the total so the fixed-width annotation column does not squeeze the
    subplot area.

    Parameters
    ----------
    n_items
        Number of items (e.g. maximum sequence length in bases) along the
        x-axis.
    font_size
        Active font size.
    max_name_length
        Maximum length (in characters) of any track / sample name.
        Used to reserve space for the left-side labels in ``tracksplot``.
    annotation_width_px
        Extra horizontal space (in pixels) reserved for the row-annotation
        column.  Default is 0 (no annotation).

    Returns
    -------
    content_width, total_width
        Recommended width in pixels.
    """
    px_per_item = font_size * WATERFALL_PX_PER_TRACK #WATERFALL_PX_PER_ITEM
    content_width = n_items * px_per_item
    # tracksplot subtracts max_name_length * 8 + 40 from figsize[0] to obtain real_width.
    # We add that overhead back so the content area is preserved.
    tracksplot_overhead = max_name_length * 8 + 40
    padding = 0  # extra breathing room beyond tracksplot's own margin
    total_width = max(
        int(content_width + tracksplot_overhead + padding + annotation_width_px),
        WATERFALL_MIN_WIDTH,
    )
    return content_width, total_width


def waterfall_height(
    n_tracks: int, 
    font_size: int = BASE_FONT_SIZE
) -> tuple[int, int]:
    """
    Compute waterfall height from track count.

    Matches the existing ``waterfall`` logic:
    ``max(n_tracks * font_size * 1.6 + 95, 300)``.

    Parameters
    ----------
    n_tracks
        Number of tracks / samples along the y-axis.
    font_size
        Active font size.

    Returns
    -------
    content_height, total_height
        Recommended height in pixels.
    """
    px_per_track = font_size * WATERFALL_PX_PER_TRACK
    margins = WATERFALL_TOP_MARGIN + WATERFALL_BOTTOM_MARGIN
    content_height = n_tracks * px_per_track + margins
    total_height = max(int(content_height), WATERFALL_MIN_HEIGHT)
    return content_height, total_height

WATERFALL_LEGEND_MIN_WIDTH: int = 200
WATERFALL_LEGEND_MIN_HEIGHT: int = 10
WATERFALL_LEGEND_BASE_WIDTH: int = 80
WATERFALL_LEGEND_EXTRA_HEIGHT: int = 20
WATERFALL_LEGEND_PX_PER_ITEM_V: int = 30
WATERFALL_LEGEND_PX_PER_CHAR_H: float = 0.7

def waterfall_legend_width(
    max_label_len: int,
    font_size: int = BASE_FONT_SIZE,
) -> int:
    """
    Compute waterfall-legend width from longest label.

    Parameters
    ----------
    max_label_len
        Length (in characters) of the longest legend label.
    font_size
        Active font size.

    Returns
    -------
    int
        Recommended width in pixels.
    """
    label_width = max_label_len * font_size * WATERFALL_LEGEND_PX_PER_CHAR_H
    return max(int(WATERFALL_LEGEND_BASE_WIDTH + label_width + 20), WATERFALL_LEGEND_MIN_WIDTH)

def waterfall_legend_height(
    n_items: int,
    font_size: int = BASE_FONT_SIZE,
) -> int:
    """
    Compute waterfall-legend height from item count.

    Parameters
    ----------
    n_items
        Number of legend items.
    font_size
        Active font size.

    Returns
    -------
    int
        Recommended height in pixels.
    """
    px_per_item = _scale(WATERFALL_LEGEND_PX_PER_ITEM_V, font_size)
    return max(int(n_items * px_per_item + WATERFALL_LEGEND_EXTRA_HEIGHT), WATERFALL_LEGEND_MIN_HEIGHT)

def waterfall_legend_margin(
    max_label_len: int,
    font_size: int = BASE_FONT_SIZE,
) -> dict[str, int]:
    """
    Compute margins for waterfall-legend that keep labels inside the figure.

    The right margin is enlarged proportionally to the longest label so
    that long text does not get clipped by the figure boundary.

    Parameters
    ----------
    max_label_len
        Length (in characters) of the longest legend label.
    font_size
        Active font size.

    Returns
    -------
    dict[str, int]
        Plotly-compatible ``margin`` dict.
    """
    return dict(l=20, r=20, t=10, b=10)


"""
#
# tracksplot function
#
"""
TRACKSPLOT_WIDTH: int = 600
TRACKSPLOT_HEATMAP_HEIGHT: int = TRACKSPLOT_WIDTH / 2
TRACKSPLOT_NON_HEATMAP_HEIGHT: int = 25
TRACKSPLOT_NAME_PX_PER_CHAR: int = 10
TRACKSPLOT_MARGIN: int = 40

def tracksplot_width(
    tracks: list[dict],
    font_size: int = BASE_FONT_SIZE,
    max_name_length: int = 10,
    base_width: int = TRACKSPLOT_WIDTH,
) -> int:
    """Compute tracksplot width.

    The current tracksplot subtracts ``MAX_NAME_LENGTH * 8 + MIN_MARGIN``
    from the total width to obtain the *real* drawing area.  This function
    inverts that relationship.

    Parameters
    ----------
    tracks
        list of track configuration dictionaries.
    font_size
        Active font size.
    max_name_length
        Maximum length (in characters) of any track name.
    base_width
        The plotting width of the whole plot.

    Returns
    -------
    int
        Recommended width in pixels.
    """
    px_per_item = _scale(TRACKSPLOT_NAME_PX_PER_CHAR, font_size)
    name_offset = max_name_length * px_per_item
    content_width = base_width
    return int(content_width + name_offset + TRACKSPLOT_MARGIN)

def tracksplot_left_margin(
    tracks: list[dict],
    font_size: int = BASE_FONT_SIZE,
    max_name_length: int = 10,
) -> int:
    """Compute tracksplot width.

    The current tracksplot subtracts ``MAX_NAME_LENGTH * 8 + MIN_MARGIN``
    from the total width to obtain the *real* drawing area.  This function
    inverts that relationship.

    Parameters
    ----------
    tracks
        list of track configuration dictionaries.
    font_size
        Active font size.
    max_name_length
        Maximum length (in characters) of any track name.

    Returns
    -------
    int
        Recommended width in pixels.
    """
    px_per_item = _scale(TRACKSPLOT_NAME_PX_PER_CHAR, font_size)
    name_offset = max_name_length * px_per_item
    return int(name_offset) + int(TRACKSPLOT_MARGIN / 2)

def tracksplot_height(
    tracks: list[dict],
    font_size: int = BASE_FONT_SIZE,
    vertical_spacing: float = 0.02,
    max_name_length: int = 0,
    base_width: int = TRACKSPLOT_WIDTH,
) -> int:
    """Compute tracksplot height from track configuration.

    Heatmap tracks are sized at half the content width (square-ish cells).
    Non-heatmap tracks scale by their ``height`` multiplier.
    Vertical spacing between subplots is accounted for.

    Parameters
    ----------
    tracks
        list of track configuration dictionaries.
    font_size
        Active font size.
    vertical_spacing
        Fraction of total height used as spacing between subplots.
    max_name_length
        Maximum length (in characters) of any track name.
    base_width
        The plotting width of the whole plot.

    Returns
    -------
    int
        Recommended height in pixels.
    """
    n_tracks = len(tracks)
    if n_tracks == 0:
        return TRACKSPLOT_MIN_HEIGHT

    # Estimate content width so heatmap height = content_width / 2 matches
    # the value used inside tracksplot().
    estimated_width = tracksplot_width(tracks, font_size, max_name_length, base_width)
    content_width = estimated_width - max_name_length * _scale(TRACKSPLOT_NAME_PX_PER_CHAR, font_size) - TRACKSPLOT_MARGIN
    heatmap_px = content_width / 2
    base_px = TRACKSPLOT_NON_HEATMAP_HEIGHT

    # Total raw pixel height required by all tracks
    total_raw = 0.0
    for track in tracks:
        if track.get("type") == "heatmap":
            total_raw += heatmap_px
        else:
            total_raw += base_px * track.get("height", 1.0)

    # make_subplots vertical_spacing consumes a fraction of the plotting area.
    # plotting_area = total_raw + vertical_spacing * (n-1) * plotting_area
    # -> plotting_area = total_raw / (1 - vertical_spacing * (n-1))
    plotting_area = total_raw / (1 - vertical_spacing * (n_tracks - 1))

    return int(plotting_area + TRACKSPLOT_MARGIN * 2)


def tracksplot_subplot_heights(
    tracks: list[dict],
    real_width: float,
    real_height: float,
    vertical_spacing: float = 0.02,
    font_size: int = BASE_FONT_SIZE,
) -> tuple[list[float], float]:
    """Compute normalized subplot height fractions for tracksplot.

    Heatmap tracks are allocated a fixed pixel budget (``real_width / 2``).
    Non-heatmap tracks share the remaining space proportionally by their
    ``height`` multiplier.

    Parameters
    ----------
    tracks
        list of track configuration dictionaries.
    real_width
        Width of the plotting area in pixels (figsize minus margins and name
        labels).  Used to set heatmap track heights.
    real_height
        Height of the plotting area in pixels (figsize minus top/bottom
        margins).  Used to determine how much vertical room is available.
    vertical_spacing
        Vertical spacing between subplots (fraction of total height).
    font_size
        Active font size.

    Returns
    -------
    tuple[list[float], float]
        ``(heights, total_raw_px)`` where *heights* are normalized fractions
        suitable for ``make_subplots(row_heights=...)`` and *total_raw_px*
        is the un-normalized pixel sum.
    """
    import numpy as np

    n_tracks = len(tracks)
    if n_tracks == 0:
        return [], 0.0

    heatmap_px = real_width / 2

    heights = np.zeros(n_tracks, dtype=np.float32)
    have_heatmap = any(t.get("type") == "heatmap" for t in tracks)

    # Available height for all subplots (excluding vertical spacing).
    total_height = real_height * (1.0 - vertical_spacing * (n_tracks - 1))
    assignable_height = total_height
    total_ratio = sum(track.get("height", 1.0) for track in tracks)

    # First pass: reserve fixed pixel budget for heatmaps.
    if have_heatmap:
        for idx, track in enumerate(tracks):
            if track.get("type") == "heatmap":
                heights[idx] = heatmap_px
                assignable_height -= heatmap_px
                total_ratio -= track.get("height", 1.0)

    if assignable_height <= 0:
        raise ValueError("The figure height is too small to fit the heatmaps")

    # Second pass: distribute remaining space to non-heatmaps.
    for idx, track in enumerate(tracks):
        if track.get("type") != "heatmap":
            heights[idx] = (
                assignable_height / total_ratio * track.get("height", 1.0)
                if total_ratio > 0 else 0
            )

    # Normalise so make_subplots receives proper fractions.
    total_raw = float(heights.sum())
    if total_raw > 0:
        heights = heights / total_raw

    return heights.tolist(), total_raw


# ---- Heatmap (generic) ---------------------------------------------------- #

HEATMAP_PX_PER_ROW: int = 18
HEATMAP_PX_PER_COL: int = 18
HEATMAP_DENDROGRAM_PX: int = 120
HEATMAP_ANNOTATION_PX: int = 18
HEATMAP_MIN_WIDTH: int = 200
HEATMAP_MIN_HEIGHT: int = 100

def heatmap_width(
    n_cols: int,
    font_size: int = BASE_FONT_SIZE,
    cluster_rows: bool = True,
    has_row_annotation: bool = False,
    n_row_annotations: int = 0,
    l_margin: int = 80,
    r_margin: int = 120,
) -> int:
    """Compute generic heatmap width.

    Parameters
    ----------
    n_cols
        Number of columns in the matrix.
    font_size
        Active font size.
    cluster_rows
        Whether a row dendrogram will be rendered (occupies left side).
    has_row_annotation
        Whether a row annotation bar will be rendered (occupies left side).
        Kept for backward compatibility; prefer ``n_row_annotations``.
    n_row_annotations
        Number of row annotation dimensions. Each dimension adds
        :data:`HEATMAP_ANNOTATION_PX` to the total width.
    l_margin
        Left margin width.
    r_margin
        Right margin width.

    Returns
    -------
    int
        Recommended width in pixels.
    """
    px_per_col = _scale(HEATMAP_PX_PER_COL, font_size)
    extras = l_margin + r_margin
    if cluster_rows:
        extras += HEATMAP_DENDROGRAM_PX
    n_annot = n_row_annotations if n_row_annotations > 0 else (1 if has_row_annotation else 0)
    extras += HEATMAP_ANNOTATION_PX * n_annot
    return max(int(n_cols * px_per_col + extras), HEATMAP_MIN_WIDTH)


def heatmap_height(
    n_rows: int,
    font_size: int = BASE_FONT_SIZE,
    cluster_cols: bool = True,
    has_col_annotation: bool = False,
    n_col_annotations: int = 0,
    t_margin: int = 80,
    b_margin: int = 100,
) -> int:
    """
    Compute generic heatmap height.

    Parameters
    ----------
    n_rows
        Number of rows in the matrix.
    font_size
        Active font size.
    cluster_cols
        Whether a column dendrogram will be rendered (occupies top side).
    has_col_annotation
        Whether a column annotation bar will be rendered (occupies top side).
        Kept for backward compatibility; prefer ``n_col_annotations``.
    n_col_annotations
        Number of column annotation dimensions. Each dimension adds
        :data:`HEATMAP_ANNOTATION_PX` to the total height.
    t_margin
        Top margin width.
    b_margin
        Bottom margin width.

    Returns
    -------
    int
        Recommended height in pixels.
    """
    px_per_row = _scale(HEATMAP_PX_PER_ROW, font_size)
    extras = t_margin + b_margin
    if cluster_cols:
        extras += HEATMAP_DENDROGRAM_PX
    n_annot = n_col_annotations if n_col_annotations > 0 else (1 if has_col_annotation else 0)
    extras += HEATMAP_ANNOTATION_PX * n_annot
    return max(int(n_rows * px_per_row + extras), HEATMAP_MIN_HEIGHT)


"""
#
# single violin function
#
"""
VIOLIN_PX_PER_GROUP: int = 70
VIOLIN_BASE_WIDTH: int = 220
VIOLIN_MIN_WIDTH: int = 220
VIOLIN_HEIGHT: int = 530

def violin_width(
    n_groups: int,
    font_size: int = BASE_FONT_SIZE,
) -> int:
    """Compute single violin plot width.

    Width grows as ``base_width + n_groups * per_group``.  The fixed
    ``base_width`` covers y-axis labels, title and margins; the variable
    part guarantees each violin gets the same comfortable pixel budget
    regardless of how many groups are present.

    Parameters
    ----------
    n_groups
        Number of groups (x-axis categories).
    font_size
        Active font size.

    Returns
    -------
    int
        Recommended width in pixels.
    """
    base = _scale(VIOLIN_BASE_WIDTH, font_size)
    per_group = _scale(VIOLIN_PX_PER_GROUP, font_size)
    return max(int(base + n_groups * per_group), VIOLIN_MIN_WIDTH)

def violin_height(
    font_size: int = BASE_FONT_SIZE,
) -> int:
    """Compute single violin plot height.

    Parameters
    ----------
    font_size
        Active font size.

    Returns
    -------
    int
        Recommended height in pixels.
    """
    return VIOLIN_HEIGHT


"""
#
# stacked violin function
#
"""
STACKED_VIOLIN_PX_PER_GROUP: int = 70
STACKED_VIOLIN_BASE_WIDTH: int = 220
STACKED_VIOLIN_MIN_WIDTH: int = 220
STACKED_VIOLIN_ROW_HEIGHT: int = 50
STACKED_VIOLIN_EXTRA_HEIGHT: int = 120

def stacked_violin_width(
    n_groups: int = 1,
    font_size: int = BASE_FONT_SIZE,
) -> int:
    """Compute stacked-violin width.

    Parameters
    ----------
    n_groups
        Number of groups on the x-axis.
    font_size
        Active font size.

    Returns
    -------
    int
        Recommended width in pixels.
    """
    base = _scale(VIOLIN_BASE_WIDTH, font_size)
    per_group = _scale(VIOLIN_PX_PER_GROUP, font_size)
    return max(int(base + n_groups * per_group), VIOLIN_MIN_WIDTH)


def stacked_violin_height(
    n_motifs: int,
    row_height: int | None = None,
    font_size: int = BASE_FONT_SIZE,
) -> int:
    """Compute stacked-violin height from motif count.

    Matches the existing ``copy_number_stacked_violin`` logic:
    ``n_motifs * row_height + 120``.

    Parameters
    ----------
    n_motifs
        Number of motifs (rows).
    row_height
        Height per row in pixels.  If ``None``, the module default
        (:data:`STACKED_VIOLIN_ROW_HEIGHT`) is used.
    font_size
        Active font size.

    Returns
    -------
    int
        Recommended height in pixels.
    """
    rh = row_height if row_height is not None else STACKED_VIOLIN_ROW_HEIGHT
    return int(n_motifs * _scale(rh, font_size) + _scale(STACKED_VIOLIN_EXTRA_HEIGHT, font_size))


"""
#
# DNA logo function
#
"""
LOGO_PX_PER_BASE: int = 8
LOGO_HEIGHT: int = 160
LOGO_MARGIN_H: int = 160

def logo_width(
    seq_len: int,
    font_size: int = BASE_FONT_SIZE,
) -> int:
    """Compute sequence-logo width.

    Parameters
    ----------
    seq_len
        Sequence length (number of bases / positions).
    font_size
        Active font size.

    Returns
    -------
    int
        Recommended width in pixels.
    """
    px_per_base = LOGO_PX_PER_BASE
    return max(int(seq_len * px_per_base + LOGO_MARGIN_H), MIN_WIDTH)


def logo_height(
    font_size: int = BASE_FONT_SIZE,
) -> int:
    """Compute sequence-logo height.

    Parameters
    ----------
    font_size
        Active font size.

    Returns
    -------
    int
        Recommended height in pixels.
    """
    return LOGO_HEIGHT
