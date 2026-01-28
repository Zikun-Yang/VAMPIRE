import plotly.graph_objects as go
import plotly.subplots as sp
from pathlib import Path
import polars as pl
from typing import List

def read_smoothness_file(file_path: str) -> pl.DataFrame:
    """
    Read a smoothness_distribution_*.txt file and convert to a Polars DataFrame
    with columns: score, count
    """
    # multiple lines
    counts: List[int] = []
    with open(file_path, "r") as f:
        lines: List[str] = f.readlines()
        for line in lines:
            if not counts:
                counts = [int(x) for x in line.split("\t")]  # split by tabs and convert to int
            else:
                ll: List[str] = line.split("\t")
                for i in range(len(ll)):
                    counts[i] += int(ll[i])

    scores = list(range(len(counts)))

    df = pl.DataFrame({
        "score": scores,
        "count": counts
    })

    return df

def plot_smoothness_score_distribution(job_dir: str) -> go.Figure:
    """
    Plot the smoothness score distribution
    Input:
        job_dir: str
    Output:
        fig: go.Figure
    """

    # get all smoothness_distribution_*.txt files
    stats_path = Path(job_dir) / "stats"
    file_list: list[Path] = list(stats_path.glob("smoothness_distribution_*.txt"))
    ksize_list: list[int] = [int(file.stem.split("_")[-1]) for file in file_list]
    max_ksize: int = max(ksize_list)

    # sort by ksize
    file_ksize_tuple = sorted(zip(file_list, ksize_list), key=lambda x: x[1])
    
    # raise error if no files found
    if not file_ksize_tuple:
        raise FileNotFoundError(f"No smoothness files found in {stats_path}")

    fig = sp.make_subplots(rows=1, cols=1)
    # iterate over files, each k size has a trace
    visibility = []
    for file, ksize in file_ksize_tuple:
        data = read_smoothness_file(file)
        # set visibility
        visible = True if ksize == max_ksize else False
        visibility.append(visible)
        subfig = go.Bar(x=data["score"][1:100],
                        y=data["count"][1:100],
                        name=f"k={ksize}",
                        visible=visible,
                        marker=dict(color="#C1121F"), # red
                        hoverlabel=dict(
                            ###font=dict(color="#6B2626"),  # color of the hover text
                            bgcolor="lightgrey"              # optional: background color of the hover box
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