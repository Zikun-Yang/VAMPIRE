import plotly.io as pio
import plotly.graph_objects as go

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