import polars as pl
import plotly.graph_objects as go
import plotly.subplots as sp

def plot_entropy_distribution(df: pl.DataFrame) -> go.Figure:
    """
    Plot the entropy distribution
    Input:
        df: pl.DataFrame
    Output:
        fig: go.Figure (histogram plot)
    """
    fig = sp.make_subplots(rows=1, cols=1)
    fig.add_trace(go.Histogram(x=df["entropy"], nbinsx=200, name="Entropy", marker=dict(color="#C1121F"))) # red
    fig.update_layout(
        height=400,
        width=500,
        # transparent background
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        # grid color
        xaxis=dict(
            title="Entropy",
            range=[0, 2],
            showline=True,       # show axis line
            linecolor='black',   # axis line color
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=0.7
        ),
        yaxis=dict(
            title="Count",
            showline=True,
            linecolor='black',
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=0.7
        ),
        margin=dict(
            l=20,   # left margin
            r=20,   # right margin
            t=20,   # top margin
            b=20    # bottom margin
        )
    )

    return fig