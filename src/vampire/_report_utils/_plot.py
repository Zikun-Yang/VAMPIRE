import polars as pl
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
from typing import List, Dict
from ._motif_processing import collapse_long_motif
from ._stats_basic import calculate_window_stats

DROPMENU_HEIGHT: int = 60

def plot_enriched_motifs_by_copy_number(df: pl.DataFrame,
                                        max_motif_length_to_collapse: int = 10,
                                        topn: int = 10,
                                        max_bubble_size: int = 50,
                                        min_bubble_size: int = 10,
                                        legend_bubble_num: int = 5,
) -> go.Figure:
    """
    Plot the enriched motifs by copy number
    Input:
        df: pl.DataFrame
        max_motif_length_to_collapse: int
        topn: int
        max_bubble_size: int
        min_bubble_size: int
        legend_bubble_num: int
    Output:
        fig: go.Figure (bubble plot)
    """
    # prepare data
    enriched_df = (
        df.group_by("canonical_motif")
            .agg(pl.len().alias("tr_count"),
                 pl.col("copyNumber").sum().alias("total_copy_number"))
            .sort("total_copy_number", descending=True)
            .select(["canonical_motif", "tr_count", "total_copy_number"])
            .head(topn)
        )
    enriched_df = enriched_df.with_columns([
        pl.col("canonical_motif").map_elements(lambda x: collapse_long_motif(x, max_length=max_motif_length_to_collapse), return_dtype=pl.Utf8).alias("shortened_motif")
    ])

    max_tr_count = enriched_df["tr_count"].max()
    min_tr_count = enriched_df["tr_count"].min()

    def _scale_size(v):
        if max_tr_count == min_tr_count:
            return (min_bubble_size + max_bubble_size) / 2
        norm = max(v - min_tr_count, 0) / (max_tr_count - min_tr_count)
        return min_bubble_size + np.sqrt(norm) * (max_bubble_size - min_bubble_size)

    sizes = [_scale_size(v) for v in enriched_df["tr_count"]]

    # subplot: main + legend panel
    fig = sp.make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.85, 0.15],
        horizontal_spacing=0.05,
    )

    # main bubble plot
    fig.add_trace(
        go.Scatter(
            x=enriched_df["total_copy_number"],
            y=enriched_df["shortened_motif"],
            mode="markers",
            customdata=enriched_df,
            hovertemplate=(
                "Canonical Motif: %{customdata[0]}<br>"
                "TR Count: %{customdata[1]}<br>"
                "Total Copy Number: %{customdata[2]}"
            ),
            marker=dict(
                size=sizes,
                sizemode="diameter",
                sizeref=1,
                color="rgba(193,18,31,0.4)",
            ),
            hoverlabel=dict(bgcolor="lightgrey"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # legend values
    min_tr = enriched_df["tr_count"].min()
    max_tr = enriched_df["tr_count"].max()

    if legend_bubble_num >= len(enriched_df):
        ref_vals = enriched_df["tr_count"].to_list()
    else:
        t_vals = np.linspace(0, 1, legend_bubble_num)
        ref_vals = min_tr + (t_vals ** 2) * (max_tr - min_tr)
        ref_vals = [int(round(v)) for v in ref_vals]

    ref_vals = sorted(set(ref_vals)) # from small numbers to large numbers
    diameter_list = [_scale_size(v) for v in ref_vals]
    radii_list: List[float] = [s / 2 for s in diameter_list]

    # --- legend layout calculation ---
    FIG_WIDTH: int = 500
    margin_l, margin_r = 20, 20
    h_spacing = 0.02
    col_widths = [0.6, 0.4]

    # legend panel pixel size
    usable_width = FIG_WIDTH - margin_l - margin_r
    gap_px = h_spacing * usable_width
    panel_width_px = (usable_width - gap_px) * col_widths[1] / sum(col_widths)
    panel_height_px = 50 * topn

    fixed_gap_px = 14
    title_height_px = 50
    x_offset_data = 0.05  # small left padding to avoid clipping

    # x: all circles center-aligned; largest circle's left edge at x_offset_data
    # xaxis2 range [0, 1], 1 data unit = panel_width_px pixels
    max_r = max(radii_list)
    x_center = x_offset_data + max_r / panel_width_px
    x_array = [x_center] * len(ref_vals)

    # fixed x for text labels: aligned to right edge of largest circle + small gap
    text_gap_px = 14
    text_x = x_center + max_r / panel_width_px + text_gap_px / panel_width_px

    # y: compact, top-down
    fixed_gap_data = fixed_gap_px / panel_height_px
    title_gap_data = title_height_px / panel_height_px

    y_array = []
    y_cur = title_gap_data
    for i, r in enumerate(radii_list):
        r_data = r / panel_height_px
        y_center = y_cur + r_data
        y_array.append(y_center)
        y_cur = y_center + r_data + fixed_gap_data

    # y-axis range: keep legend in upper portion only
    total_height = y_array[-1] + radii_list[-1] / panel_height_px
    y_range_bottom = max(1.0, total_height * 1.5)

    # subplot: main + legend panel
    fig = sp.make_subplots(
        rows=1,
        cols=2,
        column_widths=col_widths,
        horizontal_spacing=h_spacing
    )

    # main bubble plot
    fig.add_trace(
        go.Scatter(
            x=enriched_df["total_copy_number"],
            y=enriched_df["shortened_motif"],
            mode="markers",
            customdata=enriched_df,
            hovertemplate=(
                "Canonical Motif: %{customdata[0]}<br>"
                "TR Count: %{customdata[1]}<br>"
                "Total Copy Number: %{customdata[2]}"
            ),
            marker=dict(
                size=sizes,
                sizemode="diameter",
                sizeref=1,
                color="rgba(193,18,31,0.4)",
            ),
            hoverlabel=dict(bgcolor="lightgrey"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # legend panel
    fig.add_trace(
        go.Scatter(
            x=x_array,
            y=y_array,
            mode="markers",
            marker=dict(
                size=diameter_list,
                sizemode="diameter",
                sizeref=1,
                color="rgba(193,18,31,0.4)",
            ),
            hoverinfo="skip",
            hovertemplate=None,
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # text labels: left-aligned to the right edge of the largest circle
    for y, val in zip(y_array, ref_vals):
        fig.add_annotation(
            x=text_x,
            y=y,
            xref="x2",
            yref="y2",
            text=str(val),
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=14),
        )

    # legend title: left-aligned with largest circle's left edge
    fig.add_annotation(
        text="TR count",
        x=x_offset_data,
        y=title_gap_data / 2,
        xref="x2",
        yref="y2",
        showarrow=False,
        font=dict(size=16),
        xanchor="left",
        yanchor="middle",
    )

    # layout
    fig.update_layout(
        height=50 * topn,
        width=FIG_WIDTH,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=margin_l, r=margin_r, t=20, b=20),
    )

    # main figure axis
    fig.update_xaxes(
        title="Total Copy Number",
        showline=True,
        linecolor="black",
        showgrid=True,
        gridcolor="lightgrey",
        gridwidth=0.7,
        ticks="outside",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title="Canonical Motif",
        showline=True,
        linecolor="black",
        showgrid=True,
        gridcolor="lightgrey",
        gridwidth=0.7,
        autorange="reversed",
        row=1,
        col=1,
    )

    # legend panel axis
    fig.update_xaxes(
        range=[0, 1],
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        fixedrange=True,
        row=1,
        col=2,
    )

    fig.update_yaxes(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[y_range_bottom, 0],
        fixedrange=True,
        row=1,
        col=2,
    )

    return fig

def plot_enriched_motifs_by_tr_count(df: pl.DataFrame,
                                      max_motif_length_to_collapse: int = 10,
                                      topn: int = 10,
                                      max_bubble_size: int = 50,
                                      min_bubble_size: int = 10,
                                      legend_bubble_num: int = 5,
) -> go.Figure:
    """
    Plot the enriched motifs by TR count
    Input:
        df: pl.DataFrame
        max_motif_length_to_collapse: int
        topn: int
        max_bubble_size: int
        min_bubble_size: int
        legend_bubble_num: int
    Output:
        fig: go.Figure (bubble plot)
    """
    # prepare data
    enriched_df = (
        df.group_by("canonical_motif")
            .agg(pl.len().alias("tr_count"),
                 pl.col("copyNumber").sum().alias("total_copy_number"))
            .sort("tr_count", descending=True)
            .select(["canonical_motif", "tr_count", "total_copy_number"])
            .head(topn)
        )
    enriched_df = enriched_df.with_columns([
        pl.col("canonical_motif").map_elements(lambda x: collapse_long_motif(x, max_length=max_motif_length_to_collapse), return_dtype=pl.Utf8).alias("shortened_motif")
    ])

    max_tr_count = enriched_df["total_copy_number"].max()
    min_tr_count = enriched_df["total_copy_number"].min()

    def _scale_size(v):
        if max_tr_count == min_tr_count:
            return (min_bubble_size + max_bubble_size) / 2
        norm = max(v - min_tr_count, 0.0) / (max_tr_count - min_tr_count)
        return min_bubble_size + np.sqrt(norm) * (max_bubble_size - min_bubble_size)

    # main figure values
    sizes = [_scale_size(v) for v in enriched_df["total_copy_number"]]

    # legend values
    min_tr = enriched_df["total_copy_number"].min()
    max_tr = enriched_df["total_copy_number"].max()

    if legend_bubble_num >= len(enriched_df):
        ref_vals = enriched_df["total_copy_number"].to_list()
    else:
        t_vals = np.linspace(0, 1, legend_bubble_num)
        ref_vals = min_tr + (t_vals ** 2) * (max_tr - min_tr)
        ref_vals = [int(round(v)) for v in ref_vals]

    ref_vals = sorted(set(ref_vals))
    diameter_list = [_scale_size(v) for v in ref_vals]
    radii_list: List[float] = [s / 2 for s in diameter_list]

    # --- legend layout calculation ---
    FIG_WIDTH: int = 500
    margin_l, margin_r = 20, 20
    h_spacing = 0.02
    col_widths = [0.6, 0.4]

    # legend panel pixel size
    usable_width = FIG_WIDTH - margin_l - margin_r
    gap_px = h_spacing * usable_width
    panel_width_px = (usable_width - gap_px) * col_widths[1] / sum(col_widths)
    panel_height_px = 50 * topn

    fixed_gap_px = 14
    title_height_px = 50
    x_offset_data = 0.05  # small left padding to avoid clipping

    # x: all circles center-aligned; largest circle's left edge at x_offset_data
    # xaxis2 range [0, 1], 1 data unit = panel_width_px pixels
    max_r = max(radii_list)
    x_center = x_offset_data + max_r / panel_width_px
    x_array = [x_center] * len(ref_vals)

    # fixed x for text labels: aligned to right edge of largest circle + small gap
    text_gap_px = 14
    text_x = x_center + max_r / panel_width_px + text_gap_px / panel_width_px

    # y: compact, top-down
    fixed_gap_data = fixed_gap_px / panel_height_px
    title_gap_data = title_height_px / panel_height_px

    y_array = []
    y_cur = title_gap_data
    for i, r in enumerate(radii_list):
        r_data = r / panel_height_px
        y_center = y_cur + r_data
        y_array.append(y_center)
        y_cur = y_center + r_data + fixed_gap_data

    # y-axis range: keep legend in upper portion only
    total_height = y_array[-1] + radii_list[-1] / panel_height_px
    y_range_bottom = max(1.0, total_height * 1.5)

    # subplot: main + legend panel
    fig = sp.make_subplots(
        rows=1,
        cols=2,
        column_widths=col_widths,
        horizontal_spacing=h_spacing
    )

    # main bubble plot
    fig.add_trace(
        go.Scatter(
            x=enriched_df["tr_count"],
            y=enriched_df["shortened_motif"],
            mode="markers",
            customdata=enriched_df,
            hovertemplate=(
                "Canonical Motif: %{customdata[0]}<br>"
                "TR Count: %{customdata[1]}<br>"
                "Total Copy Number: %{customdata[2]}"
            ),
            marker=dict(
                size=sizes,
                sizemode="diameter",
                sizeref=1,
                color="rgba(193,18,31,0.4)",
            ),
            hoverlabel=dict(bgcolor="lightgrey"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # legend panel
    fig.add_trace(
        go.Scatter(
            x=x_array,
            y=y_array,
            mode="markers",
            marker=dict(
                size=diameter_list,
                sizemode="diameter",
                sizeref=1,
                color="rgba(193,18,31,0.4)",
            ),
            hoverinfo="skip",
            hovertemplate=None,
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # text labels: left-aligned to the right edge of the largest circle
    for y, val in zip(y_array, ref_vals):
        fig.add_annotation(
            x=text_x,
            y=y,
            xref="x2",
            yref="y2",
            text=str(val),
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=14),
        )

    # legend title: left-aligned with largest circle's left edge
    fig.add_annotation(
        text="Total copy number",
        x=x_offset_data,
        y=title_gap_data / 2,
        xref="x2",
        yref="y2",
        showarrow=False,
        font=dict(size=16),
        xanchor="left",
        yanchor="middle",
    )

    # layout
    fig.update_layout(
        height=50 * topn,
        width=FIG_WIDTH,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=margin_l, r=margin_r, t=20, b=20),
    )

    # main figure axis
    fig.update_xaxes(
        title="TR Count",
        showline=True,
        linecolor="black",
        showgrid=True,
        gridcolor="lightgrey",
        gridwidth=0.7,
        ticks="outside",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title="Canonical Motif",
        showline=True,
        linecolor="black",
        showgrid=True,
        gridcolor="lightgrey",
        gridwidth=0.7,
        autorange="reversed",
        row=1,
        col=1,
    )

    # legend panel axis
    fig.update_xaxes(
        range=[0, 1],
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        fixedrange=True,
        row=1,
        col=2,
    )

    fig.update_yaxes(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[y_range_bottom, 0],
        fixedrange=True,
        row=1,
        col=2,
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
            gridwidth=0.7,
            ticks="outside",
            showticklabels=True
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
                args=[{"visible": [metric == m for m in metrics]},
                      {"xaxis": {"title": {"text": name},
                                 "showline": True,
                                 "linecolor": "black",
                                 "showgrid": True,
                                 "gridcolor": "lightgrey",
                                 "gridwidth": 0.7,
                                 "ticks": "outside",
                                 "showticklabels": True}}
                ])
            for metric, name in zip(metrics, metric_names)]
    fig_height = 400
    fig.update_layout(
        height=fig_height,
        width=500,
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            bgcolor="white",
            bordercolor="black",
            x=0.0,
            xanchor="left",
            y=1 + DROPMENU_HEIGHT / fig_height,
            yanchor="top",
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
            gridwidth=0.7,
            ticks="outside",
            showticklabels=True
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
            l=20,
            r=20,
            t=55,
            b=20
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

    max_ksize_idx = ksize_list.index(max_ksize)
    fig_height = 400
    fig.update_layout(
        height=fig_height,
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            bgcolor="white",
            bordercolor="black",
            x=0.0,
            xanchor="left",
            y=1 + DROPMENU_HEIGHT / fig_height,
            yanchor="top",
            active=max_ksize_idx,
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
            gridwidth=0.7,
            ticks="outside",
            showticklabels=True
        ),
        yaxis=dict(
            title="Count",
            showline=True,
            linecolor='black',
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=0.7,
            ticks="outside",
            showticklabels=True
        ),
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20
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
    import numpy as np

    df = df.with_columns([
        (pl.col("end") - pl.col("start") + 1).alias("length")
    ])

    # windowing
    win_len: int = max(fasta_metainfo.values()) // resolution
    win_len = max(10, win_len) # minimum length is 10 bp
    records = []

    for chrom, chr_len in fasta_metainfo.items():
        chrom_df = df.filter(pl.col("chrom") == chrom)
        n_win = int(chr_len / win_len) + 1
        for i in range(n_win):
            w_start = int(i * win_len)
            w_end = int(min((i + 1) * win_len - 1, chr_len - 1))

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

    for metric, name, color in zip(metrics, metric_names, colors):
        x_vals = []
        y_vals = []
        widths = []
        bases = []
        customdata = []
        metric_values = []

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
            metric_values.append(val)

        max_val = max(metric_values) if metric_values else 1

        fig.add_trace(go.Bar(
            x=x_vals,
            y=y_vals,
            base=bases,
            width=widths,
            marker=dict(
                color=metric_values,
                colorscale=[
                    [0.0, color.format(0.0)],
                    [1.0, color.format(1.0)]
                ],
                cmin=0,
                cmax=max_val,
                showscale=True,
                colorbar=dict(
                    title=name,
                    bgcolor="rgba(0,0,0,0)",
                    outlinewidth=0
                )
            ),
            customdata=customdata,
            hovertemplate=(
                "Window: %{customdata[0]} - %{customdata[1]} bp<br>"
                "TR fraction: %{customdata[2]:.2f}%<br>"
                "Median TR length: %{customdata[3]} bp<br>"
                "Median motif length: %{customdata[4]} bp"
            ),
            orientation="h",
            name=name,
            visible=True if metric == "tr_fraction" else False # show tr_fraction by default
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
        range=[-0.5, len(chroms) - 0.5],
        autorange="reversed"
    )

    buttons = [dict(label=name,
                method="update",
                args=[{"visible": [metric == m for m in metrics]},
                    {"title": f"Tandem Repeat Distribution: {name}"}])
            for metric, name in zip(metrics, metric_names)]

    fig_height = 20 + 200 + 50 * (len(chroms) - 1)
    fig.update_layout(
        height=fig_height,
        width=1000,
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                bgcolor="white",
                bordercolor="black",
                x=0.0,
                xanchor="left",
                y=1 + DROPMENU_HEIGHT / fig_height,
                yanchor="top",
                showactive=True,
                buttons=buttons
            )
        ],
        xaxis=dict(
            title="Genomic coordinate (bp)",
            showline=True,
            linewidth=1,
            linecolor="black",
            ticks="outside",
            showticklabels=True
        ),
        yaxis_title="Chromosome",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=55, b=20)
    )

    return fig