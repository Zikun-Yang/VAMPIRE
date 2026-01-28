import polars as pl
import plotly.graph_objects as go
import plotly.subplots as sp

def plot_length_period_distribution(df: pl.DataFrame) -> go.Figure:
    """
    Plot the length/period distribution
    Input:
        df: pl.DataFrame
    Output:
        fig: go.Figure (histogram plot)
    """
    df = df.with_columns([
        (pl.col("end") - pl.col("start") + 1).alias("length")
    ])
    fig = sp.make_subplots(rows=1, cols=1)
    fig.add_trace(go.Histogram(x=df["length"], nbinsx=100, name="Length"))
    fig.add_trace(go.Histogram(x=df["period"], nbinsx=100, name="Period"))
    metrics = ["length", "period"]
    metric_names = ["Length", "Period"]
    colors = ["rgba(193,18,31)", "rgba(88,129,87)"]  # red, green
    for metric, name, color in zip(metrics, metric_names, colors):
        fig.add_trace(go.Histogram(x=df[metric], nbinsx=100, name=name, marker=dict(color=color)))
    buttons = [dict(label=name,
                method="update",
                args=[{"visible": [metric == m for m in metrics]}])
            for metric, name in zip(metrics, metric_names)]
    fig.update_layout(
        height=400,
        width=500,
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            bgcolor="white",
            bordercolor="black",
            x=0.0,
            y=1.2,
            showactive=True,
            buttons=buttons
        )],
        # transparent background
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        # grid color
        xaxis=dict(
            title="Length/Period",
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