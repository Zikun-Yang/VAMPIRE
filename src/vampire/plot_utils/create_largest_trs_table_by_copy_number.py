import polars as pl
import plotly.graph_objects as go

def create_largest_trs_table_by_copy_number(df: pl.DataFrame, topn: int = 10) -> go.Figure:
    """
    Plot the largest TRs by copy number
    Input:
        df: pl.DataFrame
    Output:
        fig: go.Figure
    """
    df = (
        df.sort("copyNumber", descending=True)
        .head(topn)
        .select(["chrom", "start", "end", "period", "copyNumber", "percentMatches", "percentIndels", "score", "entropy", "motif"])
    )
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df.columns),
                    fill_color="lightgrey",
                    align="center"
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    align="center"
                )
            )
        ]
    )

    fig.update_layout(
        height=100 * topn,
        width=1000,
        # transparent background
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(
            l=20,   # left margin
            r=20,   # right margin
            t=20,   # top margin
            b=20    # bottom margin
        )
    )
    return fig
