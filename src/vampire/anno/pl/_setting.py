import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

import logging

logger = logging.getLogger(__name__)

# Module-level colormap constants
_RAINBOW_COLORMAP: list[str] = [
    "#f94144", "#f8961e", "#f9c74f", "#90be6d", "#43aa8b", "#4d908e", "#577590", "#277da1", "#5983f2", "#898bf1",
    "#8945dc"
]
_GLASBEY_COLORMAP: list[str] = [
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
_DNA_BASE_COLORS: dict[str, str] = {
    "A": "#2ca02c",
    "C": "#1f77b4",
    "G": "#ff7f0e",
    "T": "#d62728",
    "-": "#403d39",
    "N": "#403d39",
}

_COLORMAP_OPTIONS: dict[str, list[str] | dict[str, str]] = {
    "rainbow": _RAINBOW_COLORMAP,
    "glasbey": _GLASBEY_COLORMAP,
    "dna": _DNA_BASE_COLORS,
}

def set_default_plotstyle(
    font_size: int = 14,
    font_family: str = "Arial",
    line_width: float = 1.5,
    width: int = 900,
    height: int = 400,
    showgrid: bool = False
):
    """
    Set the plotly template for the vampire package.

    Parameters
    ----------
        font_size: int, the font size
        font_family: str, the font family
        width: int, the width of the plot
        height: int, the height of the plot
        showgrid: bool, whether to show the grid
    
    Returns
    -------
        None

    Examples
    --------
    >>> import vampire as vp
    >>> vp.anno.pl.set_default_plotstyle(font_size=8, line_width=1)
    """
    
    pio.templates["vampire"] = go.layout.Template(
        layout=dict(
            font=dict(
                size=font_size,
                family=font_family,
            ),
            width=width,
            height=height,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(
                showgrid=showgrid,
                zeroline=False,
                linewidth=line_width,
                tickwidth=line_width,
                linecolor="black",
                ticks="outside",
            ),
            yaxis=dict(
                showgrid=showgrid,
                zeroline=False,
                linewidth=line_width,
                tickwidth=line_width,
                linecolor="black",
                ticks="outside",
            ),
            legend=dict(
                borderwidth=0,
            ),
            margin=dict(l=80, r=80, t=80, b=80),
        )
    )

    pio.templates.default = "vampire"


def _get_categorical_colormap(
    labels: list[str],
    colormap: dict[str, str] | list[str] | str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """
    Generate a color mapping for categorical labels.

    This function assigns colors to each unique label and returns:
    1. A list of colors corresponding to the input labels (in original order)
    2. A dictionary mapping each unique label to its assigned color

    Parameters
    ----------
    labels : array-like
        Sequence of categorical labels.
    colormap : dict[str, str] | list[str] | str | None, optional
        Color specification:
        - None: use default Glasbey-like color palette
        - dict[str, str]: explicit mapping from label to color
        - list[str]: list of colors cycled over unique labels
        - str: Plotly palette name (qualitative, sequential, or diverging)

    Returns
    -------
    colors : list[str]
        Color assigned to each label in input order.
    mapping : dict[str, str]
        Mapping from unique label -> color.

    Notes
    -----
    - Unrecognized labels in dict mode fall back to '#0c0c0c'
    - In list/str modes, colors are assigned in sorted label order
    - Sorting of labels is deterministic to ensure reproducibility
    """

    # check label uniqueness and order
    if len(set(labels)) != len(labels):
        raise ValueError("Labels must be unique")

    # None -> Glasbey / fallback colormap
    if colormap is None:
        colors = _GLASBEY_COLORMAP

    # dict[str, str] -> direct mapping
    elif isinstance(colormap, dict):
        colors: list[str] = []
        # check that all labels have a mapping, if not raise a error
        for l in labels:
            if l in colormap.keys():
                colors.append(colormap[l])
            else:
                logger.warning(
                    "Label '%s' not found in colormap dict, it will be colored in #0c0c0c",
                    l
                )
                colors.append("#0c0c0c")

    # list[str] -> cycle colors
    elif isinstance(colormap, list):
        if len(colormap) == 0:
            raise ValueError("colormap list cannot be empty")
        colors = colormap

    # str -> plotly built-in palettes
    elif isinstance(colormap, str):
        if _COLORMAP_OPTIONS.get(colormap) is not None:
            colors = _COLORMAP_OPTIONS[colormap]
        elif hasattr(px.colors.qualitative, colormap):
            colors = getattr(px.colors.qualitative, colormap)
        elif hasattr(px.colors.sequential, colormap):
            colors = getattr(px.colors.sequential, colormap)
        elif hasattr(px.colors.diverging, colormap):
            colors = getattr(px.colors.diverging, colormap)
        else:
            raise ValueError(
                f"Unknown plotly colormap: {colormap}"
            )

    else:
        raise TypeError(
            "colormap must be one of: dict[str, str], list[str], str, None"
        )

    if len(labels) > len(colors):
        logger.warning(
            "Number of unique labels (%d) exceeds colormap size (%d), the overflow labels will be colored in #0c0c0c",
            len(labels),
            len(colors)
        )

    mapping: dict[str, str] = {}
    for i, cat in enumerate(labels):
        if i < len(colors):
            mapping[cat] = colors[i]
        else:
            mapping[cat] = "#0c0c0c"

    return (
        [mapping.get(l, "#0c0c0c") for l in labels],
        mapping,
    )


def _save_figure(
    fig: go.Figure, 
    save: str | bool | None, 
    default_name: str
) -> None:
    """
    Save a Plotly figure following scanpy-style ``save`` semantics.

    Parameters
    ----------
    fig: go.Figure
        Plotly figure to save.
    save: str | bool | None
        ``None`` or ``False`` — do nothing.
        ``True`` — save to ``<default_name>.pdf``.
        ``str`` — if it ends with ``.pdf``, ``.png``, or ``.svg``, use it as the
        full file path; otherwise prepend it to ``default_name`` (e.g.
        ``save="prefix_"`` → ``prefix_<default_name>.pdf``).
    default_name: str
        Base filename used when ``save`` is ``True`` or a prefix string.
    """
    if save is None or save is False:
        return

    import pathlib

    if save is True:
        path = f"{default_name}.pdf"
    elif isinstance(save, str):
        save_lower = save.lower()
        if save_lower.endswith((".pdf", ".png", ".svg")):
            path = save
        else:
            path = f"{save}{default_name}.pdf"
    else:
        return

    ext = pathlib.Path(path).suffix.lstrip(".").lower()
    fmt = ext if ext in ("pdf", "png", "svg", "jpeg", "jpg", "webp") else "pdf"
    if fmt == "jpg":
        fmt = "jpeg"
    write_kwargs = {}
    if fmt == "png":
        write_kwargs["scale"] = 4
    try:
        fig.write_image(path, format=fmt, **write_kwargs)
    except Exception as e:
        logger.warning("Failed to save figure to %s: %s", path, e)