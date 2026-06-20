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


def logo(
    adata: ad.AnnData,
    feature: Literal["count", "probability", "information"] = "information",
    *,
    drop_gap: bool = False,
    colormap: dict | None = None,
    conserved_color: str | None = "#cccccc",
    title: str = "",
    figsize: tuple[int | None, int | None] | None = (None, None),
    save: str | bool | None = None,
    **kwargs
) -> go.Figure:
    """
    Plot the logo plot from anndata object. 

    Parameters
    ----------
    adata: ad.AnnData
        The AnnData object.
    feature: Literal["count", "probability", "information"]
        The feature to show. Default is "information".
    drop_gap: bool
        If True, exclude gap ("-") from counting and plotting. Only A/C/G/T
        are shown. Default is False (backward-compatible).
    colormap: dict[str, str] | None. optional
        The colors of the bases. Default is None, using the default colormap.
    conserved_color: str | None
        Override color for conserved sites (non-variant positions). Default is "#cccccc".
        If set to None, conserved sites will use the general base color instead.
    title: str
        The title of the plot. Default is empty.
    figsize : tuple[int | None, int | None], optional
        Figure size as (width, height) in pixels. Default is (None, None).
    save : str | bool | None, default=None
        If True or a str, save the figure. A string is appended to the default filename.
        Infer the filetype if ending on {'.pdf', '.png', '.svg'}.
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
    >>> vp.anno.pl.set_default_plotstyle()
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.tl.motif_msa(adata)
    >>> vp.anno.pl.logo(
    ...     adata,
    ...     feature = "information"
    ... )
    """
    import numpy as np
    import anndata as ad
    import plotly.graph_objects as go

    # config
    LETTERS: list[str] = ["A", "C", "G", "T"] if drop_gap else ["A", "C", "G", "T", "-"]
    LETTER_TO_IDX: dict[str, int] = {l: i for i, l in enumerate(LETTERS)}

    colormap = {**_COLORMAP_OPTIONS["dna"], **(colormap or {})}

    # Retrieve motif_msa alignment results (required)
    msa_result = adata.uns.get("motif_msa")
    if msa_result is None:
        raise KeyError(
            "Motif alignment not found at uns['motif_msa']. "
            "Run vp.anno.tl.motif_msa() first."
        )

    ref_seq = msa_result["reference"]
    alignment = msa_result["alignment"]

    # Build copy_number lookup by motif id
    id_to_cn = {
        str(idx): row["copy_number"]
        for idx, row in adata.var.iterrows()
    }

    # Accumulate count matrix
    count: np.ndarray = np.zeros((len(ref_seq), len(LETTERS)))

    if msa_result.get("mode") == "pairwise":
        # Pairwise mode: build count from variants (ins is ignored for logo)
        for motif_id in adata.var.index:
            cn = id_to_cn.get(str(motif_id), 1)
            for pos, base in enumerate(ref_seq):
                if base in LETTER_TO_IDX:
                    count[pos, LETTER_TO_IDX[base]] += cn

        variants = msa_result["variants"]
        for row in variants.iter_rows(named=True):
            motif_id = row["sample"]
            cn = id_to_cn.get(str(motif_id), 1)
            pos = row["pos"]
            vtype = row["type"]

            if vtype == "sub":
                ref_base = row["ref"]
                alt_base = row["alt"]
                if ref_base in LETTER_TO_IDX:
                    count[pos, LETTER_TO_IDX[ref_base]] -= cn
                if alt_base in LETTER_TO_IDX:
                    count[pos, LETTER_TO_IDX[alt_base]] += cn
            elif vtype == "del":
                del_seq = row["ref"]
                for i, base in enumerate(del_seq):
                    if pos + i < len(ref_seq) and base in LETTER_TO_IDX:
                        count[pos + i, LETTER_TO_IDX[base]] -= cn
                        if not drop_gap and "-" in LETTER_TO_IDX:
                            count[pos + i, LETTER_TO_IDX["-"]] += cn
            # ins is ignored
    else:
        # MSA mode: traverse the unified alignment
        ref_aln = alignment["reference"]
        for motif_id, seq_aln in alignment.items():
            if motif_id == "reference":
                continue
            cn = id_to_cn.get(motif_id, 1)
            ref_pos = 0
            for r, s in zip(ref_aln, seq_aln):
                if r == "-":
                    continue
                if s in LETTER_TO_IDX:
                    count[ref_pos, LETTER_TO_IDX[s]] += cn
                ref_pos += 1
            assert ref_pos == len(ref_seq)


    match feature:
        case "count":
            matrix = count
        case "probability":
            row_sums = count.sum(axis=1, keepdims=True)
            matrix = np.divide(count, row_sums, out=np.zeros_like(count, dtype=float), where=row_sums > 0)
        case "information":
            row_sums = count.sum(axis=1, keepdims=True)
            prob = np.divide(count, row_sums, out=np.zeros_like(count, dtype=float), where=row_sums > 0)
            matrix = _compute_information_content(prob, n_letter=4)
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

    if save:
        _save_figure(fig, save, "logo")

    return fig

def logo_from_matrix(
    matrix: np.ndarray,
    *,
    letters: list[str],
    feature: Literal["count", "probability", "information"] = "information",
    colormap: dict | None = None,
    conserved_color: str | None = "#cccccc",
    title: str = "",
    figsize: tuple[int | None, int | None] = (None, None),
    save: str | bool | None = None,
    **kwargs,
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

    letters : list[str]
        Symbols corresponding to matrix columns, e.g. ["A", "C", "G", "T", "-"].

    feature: Literal["count", "probability", "information"]
        The feature to use. Default is "information".

    colormap: dict | None
        The colors of the bases. Default is None, using default colormap.

    conserved_color: str | None
        Override color for conserved sites (non-variant positions). Default is "#cccccc".
        If set to None, conserved sites will use the general base color instead.

    title: str
        The title of the plot. Default is empty.

    figsize : tuple[int | None, int | None], optional
        Figure size as (width, height) in pixels. Default is (None, None).

    save : str | bool | None, default=None
        If ``True`` or a ``str``, save the figure.A string is appended to the default filename.
        Infer the filetype if ending on {``'.pdf'``, ``'.png'``, ``'.svg'``}.

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
    >>> vp.anno.pl.set_default_plotstyle()
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
    ...     colormap = {"V": "#f64021", "A": "#f98016", "M": "#ffff00", "P": "#00cc66", "I": "#496ddb", "R": "#7209b7", "E": "#a01a7d"}
    ... )
    """
    import re
    import numpy as np
    import plotly.graph_objects as go

    matrix = np.array(matrix, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0)

    LETTERS: list[str] = letters
    LETTER_PATHS: dict[str, str] = _get_letter_paths(letters = LETTERS)
    LETTER_WIDTH: float = 0.9 # width per position, 0-1

    # ensure baseline aligned
    all_verts = np.vstack([LETTER_PATHS[l]["vertices"] for l in LETTERS])
    global_min_x = all_verts[:, 0].min()
    global_max_x = all_verts[:, 0].max()
    global_min_y = all_verts[:, 1].min()
    global_max_y = all_verts[:, 1].max()

    global_sx = 1.0 / (global_max_x - global_min_x)
    global_sy = 1.0 / (global_max_y - global_min_y)

    colormap = {**_COLORMAP_OPTIONS["dna"], **(colormap or {})}

    # check colormap
    missing = set(LETTERS) - set(colormap.keys())
    if missing:
        raise ValueError(f"Letters {missing} are not covered in colormap!")

    fig: go.Figure = go.Figure()

    for pos, row in enumerate(matrix):
        order = np.argsort(row)
        y_offset = 0
        # get conservation
        is_conserved: bool = np.count_nonzero(row > 1e-6) == 1

        for idx in order:
            letter = LETTERS[idx]
            height = row[idx]

            if height <= 1e-6:
                continue

            glyph = LETTER_PATHS[letter]
            verts = glyph["vertices"].copy()
            codes = glyph["codes"]

            # normalize glyph using global bounds for consistent baseline alignment
            verts[:, 0] = (verts[:, 0] - global_min_x) * global_sx
            verts[:, 1] = (verts[:, 1] - global_min_y) * global_sy

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
        fig.update_yaxes(
            range=[0, 1],
            tickmode="array",
            tickvals=[0, 0.5, 1]
        )

    if feature == "information":
        fig.update_yaxes(
            range=[0, 2],
            tickmode="array",
            tickvals=[0, 1, 2]
        ) 

    # resolve figsize
    font_size = kwargs.get("font", {}).get("size")
    if font_size is None:
        font_size = _sizing.get_active_font_size()
    seq_len = len(matrix)

    width, height = _sizing.resolve_figsize(
        figsize[0],
        figsize[1],
        calc_width=lambda: _sizing.logo_width(seq_len, font_size),
        calc_height=lambda: _sizing.logo_height(font_size),
        min_width=10,
        min_height=100,
    )

    fig.update_layout(
        xaxis = dict(
            range=[0, seq_len],
            tickformat="d",
            title="Motif (bp)"
        ),
        yaxis = dict(
            title=feature
        ),
        title = title,
        width = width,
        height = height,
        margin = dict(l=80, r=40, t=30, b=80),
    )

    fig.update_xaxes(showline=True, linecolor="black", ticks="outside")
    fig.update_yaxes(showline=True, linecolor="black", ticks="outside")

    fig.update_layout(
        **kwargs
    )

    if save:
        _save_figure(fig, save, "logo_from_matrix")
    
    return fig

def _get_letter_paths(
    letters: list[str] = ["A", "C", "G", "T", "-"],
    fontsize: int = 1, 
    fontfamily: str = "DejaVu Sans Mono", 
    weight: str = "bold"
) -> dict[str, str]:
    """
    Get the letter paths

    Parameters
    ----------
    letters: list[str]
        list of letters. Default is ["A", "C", "G", "T"].
    fontsize: int
        Font size. Default is 1.
    fontfamily: str
        Font family. Default is "DejaVu Sans".
    weight: str
        Font weight. Default is "bold".

    Returns
    -------
    dict[str, str]
        dictionary of letter paths.
    """
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path

    fp = FontProperties(family=fontfamily, weight=weight)
    paths = {}

    # generate TextPath for letters except "-"
    normal_letters = [l for l in letters if l != "-"]
    for letter in normal_letters:
        tp = TextPath((0, 0), letter, size=fontsize, prop=fp)
        paths[letter] = {
            "vertices": tp.vertices.copy(),
            "codes": tp.codes.copy()
        }

    # generate "-" TextPath
    if "-" in letters:
        if normal_letters:
            all_verts = np.vstack([paths[l]["vertices"] for l in normal_letters])
            min_x, max_x = all_verts[:, 0].min(), all_verts[:, 0].max()
            min_y, max_y = all_verts[:, 1].min(), all_verts[:, 1].max()
        else:
            min_x, max_x = 0.0, 1.0
            min_y, max_y = 0.0, 1.0

        verts = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y],
        ])
        codes = np.array([
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ])
        paths["-"] = {
            "vertices": verts,
            "codes": codes
        }

    return paths

def _path_to_xy(
    path: str, 
    n_samples=30
) -> tuple[list[float], list[float]]:
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
    x : list[float]
        list of x-coordinates representing the polygon vertices.

    y : list[float]
        list of y-coordinates representing the polygon vertices.

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
    n_letter: int = 4,
) -> np.ndarray:
    """
    Compute the information content

    Parameters
    ----------
    mat: np.ndarray
        The 2D frequency/probability matrix.
    n_letter: int
        The number of possible letters. Default is 4 for DNA sequences.

    Returns
    -------
    pwm: np.ndarray
        The position weight matrix (PWM) matrix.
    """
    import numpy as np

    max_entropy = np.log2(n_letter)

    ic = []
    for row in mat:
        H = -np.sum(row * np.log2(row + 1e-12))
        R = max(0.0, max_entropy - H)
        ic.append(row * R)
    
    return np.array(ic)