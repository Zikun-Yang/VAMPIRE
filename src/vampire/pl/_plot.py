from typing import List, Tuple
import polars as pl
import plotly.graph_objects as go
import plotly.subplots as sp

def trackplot(tracks: List, region: str, figsize: Tuple[int, int] = (800, 200)) -> go.Figure:
    """
    Annotate the tracks with the given tracks.
    Each track gets its own subplot with independent y-axis, but shared x-axis.
    """
    # get coordinates
    region = region.split(":")
    chrom = region[0]
    start = int(region[1].split("-")[0])
    end = int(region[1].split("-")[1])

    # create subplots: one row per track, shared x-axis
    n_tracks = len(tracks)
    if n_tracks == 0:
        return go.Figure()
    # Set subplot titles: empty for all tracks (names will be shown as annotations)
    fig = sp.make_subplots(
        rows=n_tracks,
        cols=1,
        shared_xaxes=True,  # share x-axis across all subplots
        vertical_spacing=0.05,  # spacing between subplots
        subplot_titles=[""] * n_tracks  # No titles, use annotations instead
    )

    # transform lazy frame to dataframes
    track_idx = 0
    for track in tracks:
        name = track["name"]
        type = track["type"]
        data = track["data"]
        data = data.filter((pl.col("chrom") == chrom) & 
                           (pl.col("end") >= start) & 
                           (pl.col("start") <= end)).collect()
        
        match type:
            case "bedgraph":
                # Build line plot from bedgraph data
                x_coords = []
                y_coords = []
                
                # Sort by start position
                data_sorted = data.sort("start")
                
                for row_data in data_sorted.iter_rows(named=True):
                    x_coords.extend([row_data["start"], row_data["end"]])
                    y_coords.extend([row_data["value"], row_data["value"]])
                
                # Add trace with lines (step plot) to the specific subplot
                fig.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='lines',
                        name=name,
                        line=dict(shape='hv'),  # Step plot (horizontal-vertical)
                        showlegend=False
                    ),
                    row=track_idx + 1,
                    col=1
                )
                
            case "bed":
                # Build rectangular blocks for bed data using efficient Bar plot
                data_sorted = data.sort("start")
                
                # Prepare data for batch plotting
                bases = []  # x starting positions
                widths = []  # widths (end - start)
                colors = []  # colors for each rectangle
                y_positions = []  # y positions (all same for one track)
                custom_data_list = []  # custom data for each rectangle
                
                # Rectangle height
                rect_height = 0.8
                y_center = 0.5  # Center at 0.5 for each subplot
                
                # Check if color column exists
                has_itemRgb = "itemRgb" in data.columns
                
                # Get all column names for custom_data
                all_columns = data.columns
                
                for row_data in data_sorted.iter_rows(named=True):
                    bases.append(row_data["start"])
                    widths.append(row_data["end"] - row_data["start"])
                    y_positions.append(y_center)
                    
                    # Determine color
                    color = "navy"
                    if has_itemRgb:
                        item_rgb = row_data.get("itemRgb")
                        if item_rgb and item_rgb != ".":
                            # format: r,g,b
                            if ',' in item_rgb:
                                rgb = item_rgb.split(",")
                                if len(rgb) == 3:
                                    color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
                            elif item_rgb.startswith("#"):
                                color = item_rgb
                    colors.append(color)
                    
                    # Prepare custom data: store all fields as a list (Plotly expects list of lists)
                    custom_data_list.append([row_data.get(col) for col in all_columns])
                
                # Add single trace with all rectangles (batch plotting for efficiency)
                if bases:  # Only add if there are rectangles
                    fig.add_trace(
                        go.Bar(
                            x=widths,  # Bar lengths (widths)
                            y=y_positions,  # y positions
                            base=bases,  # x starting positions
                            orientation='h',  # Horizontal bars
                            marker=dict(
                                color=colors,
                                line=dict(color=colors, width=1)
                            ),
                            name=name,
                            opacity=0.7,
                            width=rect_height,  # Height of bars in y direction
                            showlegend=False,
                            customdata=custom_data_list,  # Add custom data for hover and interaction
                            meta=all_columns,  # Store column names for reference
                            hovertemplate = (
                                "<br>".join(
                                    f"{col}: %{{customdata[{i}]}}"
                                    for i, col in enumerate(all_columns)
                                )
                                + "<extra></extra>"
                            )
                        ),
                        row=track_idx + 1,
                        col=1
                    )
                    # Set y-axis for bed tracks: range and hide ticks/labels
                    fig.update_yaxes(
                        range=[0, 1],
                        showticklabels=False,  # Hide tick labels (numbers)
                        ticks="",  # Hide tick marks (short lines) - empty string means no ticks
                        row=track_idx + 1,
                        col=1
                    )
                
            case _:
                raise ValueError(f"Cannot identify the track type from the columns: {track.columns}")
        track_idx += 1
    
    # Update layout: only show x-axis label on bottom subplot
    fig.update_xaxes(title_text="Genomic Position", row=n_tracks, col=1)
    
    # Add annotations for all tracks on the left side
    # Calculate y positions for each subplot in paper coordinates
    vertical_spacing = 0.05
    subplot_height = (1.0 - vertical_spacing * (n_tracks - 1)) / n_tracks
    
    for idx, track in enumerate(tracks):
        # Calculate y position in paper coordinates (0 to 1)
        # Subplots are ordered from top to bottom
        y_top = 1.0 - idx * (subplot_height + vertical_spacing)
        y_center = y_top - subplot_height / 2
        
        # Add annotation on the left side for all tracks
        fig.add_annotation(
            text=track["name"],
            xref="paper",
            yref="paper",
            x=-0.05,  # Position on the left (negative value)
            y=y_center,
            xanchor="right",  # Right-align text
            yanchor="middle",  # Center vertically
            showarrow=False,
            xshift=0  # Additional shift if needed
        )

        # Set x-axis range to be consistent across all subplots
        fig.update_xaxes(range=[start, end])
        
        # Set figure size
        fig.update_layout(
            height=figsize[1],
            width=figsize[0],
            margin=dict(l=150, r=40, t=40, b=40),
            autosize=False
        )
    
    return fig

def _get_track_type(track: pl.DataFrame) -> str:
    """
    Get the type of the track.
    """
    BEDGRAPH_COLS = [
        "chrom",
        "start",
        "end",
        "value",
    ]
    BED_COLS = [
        "chrom",
        "start",
        "end",
        "name"
    ]
    columns = track.columns
    if set(columns) >= set(BEDGRAPH_COLS):
        return "bedgraph"
    elif len(columns) >= 3 and all(col in columns for col in BED_COLS[:3]):
        return "bed"
    else:
        raise ValueError(f"unknown track type with columns: {columns}")