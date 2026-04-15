import plotly.io as pio
import plotly.graph_objects as go

def set_default_plotstyle(
    font_size=14,
    font_family="Arial",
    width=900,
    height=400,
    showgrid=False
):
    """
    Set the plotly template for the vampire package.
    Input:
        font_size: int, the font size
        font_family: str, the font family
        width: int, the width of the plot
        height: int, the height of the plot
        showgrid: bool, whether to show the grid
    Output:
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
                linecolor="black",
                ticks="outside",
            ),
            yaxis=dict(
                showgrid=showgrid,
                zeroline=False,
                linecolor="black",
                ticks="outside",
            ),
            legend=dict(
                borderwidth=0,
            ),
            margin=dict(l=40, r=40, t=40, b=40),
        )
    )

    pio.templates.default = "vampire"