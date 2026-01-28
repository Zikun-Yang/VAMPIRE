import polars as pl
import plotly.graph_objects as go
import plotly.subplots as sp

from vampire.stats_utils import calculate_window_stats

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
        height=200 + 50 * (len(chroms) - 1),
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