import polars as pl
import plotly.graph_objects as go
import plotly.subplots as sp

def plot_enriched_motifs_by_copy_number(df: pl.DataFrame, topn: int = 10, max_bubble_size: int = 50, min_bubble_size: int = 10) -> go.Figure:
    """
    Plot the enriched motifs by copy number
    Input:
        df: pl.DataFrame
    Output:
        fig: go.Figure (bubble plot)
    """
    enriched_df = (
        df.group_by("canonical_motif")
            .agg(pl.len().alias("tr_number"),
                 pl.col("copyNumber").sum().alias("total_copy_number"))
            .sort("total_copy_number", descending=True)
            .select(["canonical_motif", "tr_number", "total_copy_number"])
            .head(topn)
        )
    max_tr_number = enriched_df["tr_number"].max()
    sizes = (enriched_df["tr_number"] / max_tr_number * max_bubble_size).to_list()
    sizes = [max(min_bubble_size, s) for s in sizes]

    fig = sp.make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=enriched_df["total_copy_number"],
                            y=enriched_df["canonical_motif"],
                            mode="markers",
                            marker_size=sizes,
                            customdata=enriched_df,
                            hovertemplate=(
                                "Canonical Motif: %{customdata[0]}<br>"
                                "TR Number: %{customdata[1]}<br>"
                                "Total Copy Number: %{customdata[2]}"
                            ),
                            marker=dict(color="rgba(193,18,31,0.4)"), # red with 0.4 opacity
                            ))
    fig.update_layout(
        height=50 * topn,
        width=500,
        # transparent background
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        # grid color
        xaxis=dict(
            title="Total Copy Number",
            showline=True,       # show axis line
            linecolor='black',   # axis line color
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=0.7
        ),
        yaxis=dict(
            title="Canonical Motif",
            showline=True,
            linecolor='black',
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=0.7,
            autorange="reversed"
        ),
        margin=dict(
            l=20,   # left margin
            r=20,   # right margin
            t=20,   # top margin
            b=20    # bottom margin
        )
    )
    return fig