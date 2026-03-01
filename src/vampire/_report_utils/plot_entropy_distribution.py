import polars as pl
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np

def plot_entropy_distribution(df: pl.DataFrame, bin_width: float = 0.02) -> go.Figure:
    """
    Plot the entropy distribution
    Input:
        df: pl.DataFrame
        bin_width: float, the width of the bin
    Output:
        fig: go.Figure (histogram plot)
    """
    entropy = df["entropy"].to_numpy()
    entropy_idx = np.floor(entropy / bin_width).astype(int)
    counts = np.bincount(entropy_idx)

    entropy_counts = pl.DataFrame({
        "entropy": np.arange(len(counts)) * bin_width,
        "count": counts
    })

    fig = sp.make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(x=entropy_counts["entropy"], 
                        y=entropy_counts["count"], 
                        name="Entropy", 
                        marker=dict(color="#C1121F"), # red
                        hoverlabel=dict(
                            bgcolor="lightgrey" # background color of the hover box
                        )))
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