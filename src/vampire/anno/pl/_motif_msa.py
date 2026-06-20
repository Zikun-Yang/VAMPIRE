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

def motif_msa(
    adata: "ad.AnnData",
    store_key: str = "motif_msa",
    sample_order: list[str] | None = None,
    base_colors: dict[str, str] | None = None,
    block_size: int = 15,
    stripe_width: int = 10,
    phase: int = 0,
    show_ins_bases: bool = False,
    figsize: tuple[int | None, int | None] | None = (None, None),
    save: str | bool | None = None,
    **kwargs,
) -> "go.Figure":
    """
    Plot motif alignment waterfall with variant pileup.

    Visualises sample sequences aligned against a reference motif.
    The reference row is fixed at the top; each sample occupies one row
    below it.  Only variant positions (substitution, insertion, deletion)
    are drawn -- matching bases are left blank.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with alignment results stored by
        ``tl.motif_msa()``.
    store_key : str, default="motif_msa"
        Key in ``adata.uns`` that holds the alignment result.
    sample_order : list[str] | None, optional
        Explicit order for samples on the y-axis.  If ``None``, samples
        are sorted alphabetically.
    base_colors : dict[str, str] | None, optional
        Mapping from nucleotide to hex colour. Defaults to a DNA palette.
    block_size : int, default=15
        Pixel size of each nucleotide block.
    stripe_width : int, default=10
        Width of alternating background stripes in bp.
    phase : int, default=0
        Circular phase shift.  When ``phase > 0`` the motif is rolled
        right by ``phase`` bases before plotting (e.g. ``phase=1``
        moves the first base to the end).
    show_ins_bases : bool, default=False
        When ``True``, display the inserted nucleotide sequence in
        purple text to the right of each insertion symbol.
    figsize : tuple[int | None, int | None] | None, default=(None, None)
        Figure size ``(width, height)`` in pixels.  ``(None, None)``
        triggers auto-computation from sequence length and sample count.
    save : str | bool | None, default=None
        If ``True`` or a ``str``, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ending on {``'.pdf'``, ``'.png'``, ``'.svg'``}.
    **kwargs
        Additional arguments passed to ``fig.update_layout()``.

    Returns
    -------
    go.Figure
        Plotly figure with the alignment waterfall.

    Examples
    --------
    >>> import vampire as vp
    >>> vp.anno.pl.set_default_plotstyle()
    >>> adata = vp.datasets.wdr7_hprc()
    >>> vp.anno.tl.motif_msa(adata)
    >>> vp.anno.pl.motif_msa(adata)
    """
    import plotly.graph_objects as go

    # Retrieve alignment results
    result = adata.uns.get(store_key)
    if result is None:
        raise KeyError(
            f"Alignment data not found at uns['{store_key}']. "
            f"Run tl.motif_msa() first."
        )

    mode: str = result.get("mode", "pairwise")
    ref_seq: str = result["reference"]
    variants_df: pl.DataFrame = result["variants"]
    logger.info(
        "You are using alignment data generated in %s mode.",
        mode
        )

    # Apply circular phase shift for display (roll left)
    seq_len = len(ref_seq)
    if phase:
        phase = phase % seq_len
        ref_seq = ref_seq[phase:] + ref_seq[:phase]

    colors = {**_COLORMAP_OPTIONS["dna"], **(base_colors or {})}

    # Resolve sample order – numeric sort so "0".."23" is ordered correctly
    def _sort_key(x):
        try:
            return (0, int(x))
        except (ValueError, TypeError):
            return (1, str(x))

    all_samples = sorted(
        [k for k in result["alignment"].keys() if k != "reference"],
        key=_sort_key,
    )

    if sample_order is not None:
        missing = set(sample_order) - set(all_samples)
        if missing:
            raise ValueError(f"sample_order contains unknown samples: {missing}")
        samples = [s for s in sample_order if s in all_samples]
    else:
        samples = all_samples

    n_samples = len(samples)
    sample_to_y = {s: n_samples - i for i, s in enumerate(samples)}

    # Auto-size figure
    font_size = kwargs.get("font", {}).get("size")
    if font_size is None:
        font_size = _sizing.get_active_font_size()

    # Prepare label strings so we can size the left margin dynamically
    if mode == "msa":
        ref_label = "consensus"
    else:
        ref_label = (
            f"reference ({result.get('reference_id')})"
            if result.get("reference_id") is not None
            else "reference"
        )
    _labels = [ref_label] + [str(s) for s in samples]
    _max_label_len = max(len(lbl) for lbl in _labels)
    _left_margin = int(_max_label_len * _sizing._scale(8, font_size) + _sizing._scale(40, font_size))
    _right_margin = 40
    _top_margin = 40
    _bottom_margin = 80

    # Natural size driven by block_size (scaleanchor="x" keeps blocks square)
    _nat_width = _left_margin + seq_len * block_size + _right_margin
    _nat_height = _top_margin + (n_samples + 1) * block_size + _bottom_margin

    # Apply user overrides from figsize
    _width = figsize[0] if figsize is not None and figsize[0] is not None else _nat_width
    _height = figsize[1] if figsize is not None and figsize[1] is not None else _nat_height

    fig = go.Figure()

    # ---- Layer 1: alternating background stripes (grey / transparent) ----
    for i in range(0, seq_len, stripe_width):
        if (i // stripe_width) % 2 == 0:
            continue  # leave white / transparent
        fig.add_vrect(
            x0=i,
            x1=min(i + stripe_width, seq_len),
            fillcolor="#e9e9e9",
            opacity=0.5,
            layer="below",
            line_width=0,
        )

    # Helper to add a nucleotide block
    def _add_block(
        x: int,
        y: int,
        base: str,
        hover_text: str = "",
        opacity: float = 1.0,
    ) -> None:
        color = colors.get(base.upper(), "#9E9E9E")
        fig.add_shape(
            type="rect",
            x0=x,
            x1=x + 1,
            y0=y - 0.5,
            y1=y + 0.5,
            fillcolor=color,
            line=dict(width=_sizing.get_active_line_width(), color="white"),
            opacity=opacity,
            layer="above",
        )
        _text_size = max(8, int(block_size * 0.55))
        fig.add_annotation(
            x=x + 0.5,
            y=y,
            text=f"<b>{base}</b>",
            showarrow=False,
            font=dict(size=_text_size, color="white"),
            xanchor="center",
            yanchor="middle",
        )
        if hover_text:
            fig.add_trace(go.Scatter(
                x=[x + 0.5],
                y=[y],
                mode="markers",
                marker=dict(size=block_size, color="rgba(0,0,0,0)"),
                hovertext=hover_text,
                hoverinfo="text",
                showlegend=False,
            ))

    # ---- Layer 2: reference row ----
    ref_y = n_samples + 1
    for pos, base in enumerate(ref_seq):
        _add_block(
            x=pos,
            y=ref_y,
            base=base,
            hover_text=f"Ref: {base}<br>Pos: {pos}",
        )

    # ---- Layer 3: sample rows ----
    if len(variants_df) > 0:
        variant_groups: dict[tuple[str, int], list[dict]] = {}
        for row in variants_df.iter_rows(named=True):
            key = (row["sample"], row["pos"])
            rec = {k: v for k, v in row.items() if k not in ("sample", "pos")}
            variant_groups.setdefault(key, []).append(rec)

        for (sample, pos), records in variant_groups.items():
            if sample not in sample_to_y:
                continue
            base_y = sample_to_y[sample]
            for stack_idx, rec in enumerate(records):
                y = base_y + stack_idx * 0.9
                vtype = rec["type"]
                # Map original position to display position after phase shift
                display_pos = (pos - phase) % seq_len
                # Insertion at the very end (pos == seq_len) should stay at the
                # right edge, not wrap to 0.
                if vtype == "ins" and pos >= seq_len:
                    display_pos = seq_len
                if vtype == "sub":
                    alt = rec.get("alt", "N")
                    _add_block(
                        x=display_pos,
                        y=y,
                        base=alt,
                        hover_text=(
                            f"Sample: {sample}<br>"
                            f"Pos: {display_pos}<br>"
                            f"Type: substitution<br>"
                            f"{rec.get('ref', '')} → {alt}"
                        ),
                    )
                elif vtype == "ins":
                    ins_seq = rec.get("seq", "")
                    # Place the tick on the boundary after display_pos
                    ins_x = display_pos
                    _INS_COLOR = "#8945dc"
                    # Draw entirely inside the current row [y-0.5, y+0.5]
                    tri_top = y + 0.5          # row top edge = triangle base
                    stick_top = y + 0.1        # triangle tip / stick top
                    stick_bottom = y - 0.5     # stick bottom = row bottom edge

                    # Vertical stick
                    fig.add_shape(
                        type="line",
                        x0=ins_x, x1=ins_x,
                        y0=stick_bottom, y1=stick_top,
                        line=dict(color=_INS_COLOR, width=2),
                        layer="above",
                    )
                    # Downward-pointing triangle (wide, right-angle tip)
                    fig.add_shape(
                        type="path",
                        path=f"M {ins_x - 0.4} {tri_top} "
                             f"L {ins_x + 0.4} {tri_top} "
                             f"L {ins_x} {stick_top} Z",
                        fillcolor=_INS_COLOR,
                        line=dict(width=0),
                        layer="above",
                    )
                    # Invisible hover target
                    fig.add_trace(go.Scatter(
                        x=[ins_x],
                        y=[y],
                        mode="markers",
                        marker=dict(size=22, color="rgba(0,0,0,0)"),
                        hovertext=(
                            f"Sample: {sample}<br>"
                            f"Pos: {display_pos}<br>"
                            f"Type: insertion<br>"
                            f"Seq: {ins_seq}"
                        ),
                        hoverinfo="text",
                        showlegend=False,
                    ))
                    # Optionally show inserted bases to the right of the symbol
                    if show_ins_bases:
                        for offset, ins_base in enumerate(ins_seq):
                            fig.add_annotation(
                                x=ins_x + 0.4 + offset * 0.5,
                                y=y,
                                text=f"<b>{ins_base}</b>",
                                showarrow=False,
                                font=dict(size=max(8, int(block_size * 0.55)), color=_INS_COLOR),
                                xanchor="center",
                                yanchor="middle",
                            )
                elif vtype == "del":
                    length = rec.get("length", 1)
                    for offset in range(length):
                        _add_block(
                            x=(display_pos + offset) % seq_len,
                            y=y,
                            base="-",
                            hover_text=(
                                f"Sample: {sample}<br>"
                                f"Pos: {(display_pos + offset) % seq_len}<br>"
                                f"Type: deletion<br>"
                                f"Length: {length}"
                            ),
                        )

    # ---- Axes and layout ----
    # Choose tick step so that ~6–15 ticks are shown regardless of motif length
    def _tick_step(n: int) -> int:
        for step in (1, 5, 10):
            if n // step + 1 <= 15:
                return step
        return 10

    _step = _tick_step(seq_len)
    tick_vals = list(range(0, seq_len + 1, _step))
    tick_text = [str(i) for i in tick_vals]

    fig.update_layout(
        xaxis=dict(
            range=[0, seq_len],
            tickvals=tick_vals,
            ticktext=tick_text,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=_sizing.get_active_line_width(),
            ticks="outside",
            tickfont=dict(size=font_size),
            title="Position (bp)",
        ),
        yaxis=dict(
            tickvals=[ref_y] + [sample_to_y[s] for s in samples],
            ticktext=[ref_label] + [str(s) for s in samples],
            range=[0.5, n_samples + 1.5],
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=_sizing.get_active_line_width(),
            ticks="outside",
            tickfont=dict(size=font_size),
        ),
        width=_width,
        height=_height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=_left_margin, r=_right_margin, t=_top_margin, b=_bottom_margin),
        **kwargs,
    )

    if save is not None:
        _save_figure(fig, save, "motif_msa")

    return fig