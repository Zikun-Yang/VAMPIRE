import polars as pl
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
from typing import List, Dict
from ._motif_processing import collapse_long_motif
from ._stats_basic import calculate_window_stats

def plot_enriched_motifs_by_copy_number(df: pl.DataFrame,
                                        max_motif_length_to_collapse: int = 10,
                                        topn: int = 10,
                                        max_bubble_size: int = 50,
                                        min_bubble_size: int = 10) -> go.Figure:
    """
    Plot the enriched motifs by copy number
    Input:
        df: pl.DataFrame
    Output:
        fig: go.Figure (bubble plot)
    """
    # prepare data
    enriched_df = (
        df.group_by("canonical_motif")
            .agg(pl.len().alias("tr_number"),
                 pl.col("copyNumber").sum().alias("total_copy_number"))
            .sort("total_copy_number", descending=True)
            .select(["canonical_motif", "tr_number", "total_copy_number"])
            .head(topn)
        )
    enriched_df = enriched_df.with_columns([
        pl.col("canonical_motif").map_elements(lambda x: collapse_long_motif(x, max_length=max_motif_length_to_collapse), return_dtype=pl.Utf8).alias("shortened_motif")
    ])
    max_tr_number = enriched_df["tr_number"].max()
    sizes = (enriched_df["tr_number"] / max_tr_number * max_bubble_size).to_list()
    sizes = [max(min_bubble_size, s) for s in sizes]

    fig = sp.make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=enriched_df["total_copy_number"],
                            y=enriched_df["shortened_motif"],
                            mode="markers",
                            name="Enriched",
                            marker_size=sizes,
                            customdata=enriched_df,
                            hovertemplate=(
                                "Canonical Motif: %{customdata[0]}<br>"
                                "TR Number: %{customdata[1]}<br>"
                                "Total Copy Number: %{customdata[2]}"
                            ),
                            marker=dict(color="rgba(193,18,31,0.4)"), # red with 0.4 opacity
                            hoverlabel=dict(
                                bgcolor="lightgrey" # background color of the hover box
                            )))
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

def plot_enriched_motifs_by_tr_number(df: pl.DataFrame,
                                      max_motif_length_to_collapse: int = 10,
                                      topn: int = 10,
                                      max_bubble_size: int = 50,
                                      min_bubble_size: int = 10) -> go.Figure:
    """
    Plot the enriched motifs by TR number
    Input:
        df: pl.DataFrame
    Output:
        fig: go.Figure (bubble plot)
    """
    # prepare data
    enriched_df = (
        df.group_by("canonical_motif")
            .agg(pl.len().alias("tr_number"),
                 pl.col("copyNumber").sum().alias("total_copy_number"))
            .sort("tr_number", descending=True)
            .select(["canonical_motif", "tr_number", "total_copy_number"])
            .head(topn)
        )
    enriched_df = enriched_df.with_columns([
        pl.col("canonical_motif").map_elements(lambda x: collapse_long_motif(x, max_length=max_motif_length_to_collapse), return_dtype=pl.Utf8).alias("shortened_motif")
    ])
    max_copy_number = enriched_df["total_copy_number"].max()
    sizes = (enriched_df["total_copy_number"] / max_copy_number * max_bubble_size).to_list()
    sizes = [max(min_bubble_size, s) for s in sizes]
    
    # plot
    fig = sp.make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=enriched_df["tr_number"],
                            y=enriched_df["shortened_motif"],
                            mode="markers",
                            name="Enriched",
                            marker_size=sizes,
                            customdata=enriched_df,
                            hovertemplate=(
                                "Canonical Motif: %{customdata[0]}<br>"
                                "TR Number: %{customdata[1]}<br>"
                                "Total Copy Number: %{customdata[2]}"
                            ),
                            marker=dict(color="rgba(193,18,31,0.4)"), # red with 0.4 opacity
                            hoverlabel=dict(
                                bgcolor="lightgrey" # background color of the hover box
                            )))

    fig.update_layout(
        height=50 * topn,
        width=500,
        # transparent background
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        # grid color
        xaxis=dict(
            title="TR Number",
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




def plot_smoothness_score_distribution(smoothness: Dict[int, pl.DataFrame]) -> go.Figure:
    """
    Plot the smoothness score distribution
    Input:
        smoothness: Dict[int, pl.DataFrame]
    Output:
        fig: go.Figure
    """

    ksize_list: List[int] = sorted(list(smoothness.keys()))
    max_ksize: int = max(ksize_list)

    fig = sp.make_subplots(rows=1, cols=1)
    for ksize in ksize_list:
        data = smoothness[ksize]
        # set visibility
        visible = True if ksize == max_ksize else False
        subfig = go.Bar(x=data["score"][1:100],
                        y=data["count"][1:100],
                        name=f"k={ksize}",
                        visible=visible,
                        marker=dict(color="#C1121F"), # red
                        hoverlabel=dict(
                            bgcolor="lightgrey" # background color of the hover box
                        ))
        fig.add_trace(subfig)
    
    # dropdown buttons
    buttons = []
    for ksize in ksize_list:
        btn = dict(
            label=f"k={ksize}",
            method="update",
            args=[{"visible": [t == ksize for t in ksize_list]}]
        )
        buttons.append(btn)

    fig.update_layout(
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
            title="Smoothness",
            range=[0, 100],
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

def plot_tr_distribution(df: pl.DataFrame, fasta_metainfo: dict, resolution: int = 100, chrom_width: float = 0.2) -> go.Figure:
    """
    Plot TR distribution with pulldown to select metric
    Input:
        df: pl.DataFrame
        fasta_metainfo: dict
        resolution: int, the window numder of the longest chromosome
        chrom_width: float
    Output:
        fig: go.Figure (bar plot)
    """

    df = df.with_columns([
        (pl.col("end") - pl.col("start") + 1).alias("length")
    ])

    # windowing
    win_len = max(fasta_metainfo.values()) // resolution
    records = []

    for chrom, chr_len in fasta_metainfo.items():
        chrom_df = df.filter(pl.col("chrom") == chrom)
        n_win = int(chr_len / win_len) + 1
        for i in range(n_win):
            w_start = int(i * win_len)
            w_end = int(min((i + 1) * win_len - 1, chr_len - 1))
            if w_end - w_start + 1 <= 100:
                continue

            tr_fraction, med_len, med_motif_len = calculate_window_stats(chrom_df, w_start, w_end)

            records.append({
                "chrom": chrom,
                "win_start": w_start,
                "win_end": w_end,
                "tr_fraction": tr_fraction,
                "tr_len_median": med_len,
                "motif_len_median": med_motif_len,
            })

    if len(records) == 0:
        return go.Figure()

    win_df = pl.DataFrame(records)
    win_df = win_df.with_columns([
        (pl.col("tr_fraction") * 100).alias("tr_fraction")
    ])

    chroms = list(fasta_metainfo.keys())
    chrom_idx = {c: i for i, c in enumerate(chroms)}

    # create subplots (1 row, 1 col)
    fig = sp.make_subplots(rows=1, cols=1)

    # create 3 traces: tr_fraction, tr_length, motif_length
    metrics = ["tr_fraction", "tr_len_median", "motif_len_median"]
    metric_names = ["TR Fraction", "Median TR Length", "Median Motif Length"]
    colors = ["rgba(193,18,31,{})", "rgba(88,129,87,{})", "rgba(0,119,182,{})"]  # red, green, blue

    for metric, name, color_template in zip(metrics, metric_names, colors):
        x_vals = []
        y_vals = []
        widths = []
        bases = []
        customdata = []

        for row in win_df.iter_rows(named=True):
            val = row[metric]
            if val is None or val == 0:
                continue
            x_vals.append(row["win_end"] - row["win_start"])
            widths.append(chrom_width)
            bases.append(row["win_start"])
            y_vals.append(chrom_idx[row["chrom"]])
            # for hover
            customdata.append([row["win_start"], row["win_end"], row["tr_fraction"], row["tr_len_median"], row["motif_len_median"]])

        fig.add_trace(go.Bar(
            x=x_vals,
            y=y_vals,
            base=bases,
            width=widths,
            marker=dict(color=[color_template.format(min(1, row["tr_fraction"]/100.0)) for row in win_df.iter_rows(named=True) if row[metric] is not None and row[metric] > 0]),
            customdata=customdata,
            hovertemplate=(
                "Window: %{customdata[0]} - %{customdata[1]} bp<br>"
                "TR fraction: %{customdata[2]:.2f}%<br>"
                "Median TR length: %{customdata[3]} bp<br>"
                "Median motif length: %{customdata[4]} bp"
            ),
            orientation="h",
            name=name,
            visible=True if metric == "tr_fraction" else False
        ))

    # add background chromosome rectangles
    for chrom, chr_len in fasta_metainfo.items():
        y = chrom_idx[chrom]
        fig.add_shape(
            type="rect",
            x0=0,
            x1=chr_len,
            y0=y - chrom_width/2,
            y1=y + chrom_width/2,
            fillcolor="lightgrey",
            opacity=0.2,
            line=dict(width=0),
            layer="below"
        )

    fig.update_yaxes(
        tickvals=list(chrom_idx.values()),
        ticktext=list(chrom_idx.keys()),
        autorange="reversed"
    )

    buttons = [dict(label=name,
                method="update",
                args=[{"visible": [metric == m for m in metrics]},
                    {"title": f"Tandem Repeat Distribution: {name}"}])
            for metric, name in zip(metrics, metric_names)]

    fig.update_layout(
        height=20 + 200 + 50 * (len(chroms) - 1),
        width=1000,
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                bgcolor="white",
                bordercolor="black",
                x=0.0,
                y=1.4,
                showactive=True,
                buttons=buttons
            )
        ],
        xaxis_title="Genomic Coordinate (bp)",
        yaxis_title="Chromosome",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20)
    )

    return fig