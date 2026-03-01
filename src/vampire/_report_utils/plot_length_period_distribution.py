import polars as pl
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np

def plot_length_period_distribution(df: pl.DataFrame, nbin: int = 100) -> go.Figure:
    """
    Plot the length/period distribution
    Input:
        df: pl.DataFrame
    Output:
        fig: go.Figure (histogram plot)
    """
    # prepare data
    df = df.with_columns([
        (pl.col("end") - pl.col("start") + 1).alias("length")])
    df = df.with_columns([
        (pl.col("length").log10()).alias("log10_length"),
        (pl.col("period").log10()).alias("log10_period")
    ])
    # plot
    fig = sp.make_subplots(rows=1, cols=1)
    metrics = ["log10_length", "log10_period", "length", "period"]
    metric_names = ["log10 Length", "log10 Period", "Length", "Period"]
    colors = ["rgba(193,18,31,1)", "rgba(88,129,87,1)", "rgba(193,18,31,1)", "rgba(88,129,87,1)"]  # red, green, red, green
    for metric, name, color in zip(metrics, metric_names, colors):
        data = df[metric].to_numpy()
        min_val, max_val = np.min(data), np.max(data)
        if min_val == max_val:
            data_counts = pl.DataFrame({
                "data": [min_val],
                "count": [len(data)]
            })
        else:
            bin_width = float(max_val - min_val) / nbin
            data_idx = np.floor((data - min_val) / bin_width).astype(int)
            counts = np.bincount(data_idx)
            data_counts = pl.DataFrame({
                "data": min_val + np.arange(len(counts)) * bin_width,
                "count": counts
            })
        # set visibility
        visible = True if metric == metrics[0] else False
        fig.add_trace(go.Bar(x=data_counts["data"],
                            y=data_counts["count"],
                            name=name,
                            visible=visible,
                            marker=dict(color=color),
                            hoverlabel=dict(
                            bgcolor="lightgrey" # background color of the hover box
                        )))
    buttons = [dict(label=name,
                method="update",
                args=[{"visible": [metric == m for m in metrics]}
                ])
            for metric, name in zip(metrics, metric_names)]
    fig.update_layout(
        height=400,
        width=500,
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            bgcolor="white",
            bordercolor="black",
            x=0.15,
            y=1.2,
            showactive=True,
            buttons=buttons
        )],
        # transparent background
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        # grid color
        xaxis=dict(
            title=metric_names[0],
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