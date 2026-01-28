import plotly.graph_objects as go

def fig_to_html(fig: go.Figure) -> str:
    """
    Convert a figure to an HTML string
    Input:
        fig: go.Figure
    Output:
        html_div: str
    """
    html_div = fig.to_html(full_html=False,
                       include_plotlyjs='cdn',
                       config={"displayModeBar": True,
                               "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d"],
                               "displaylogo": False,
                               "responsive": True})
    return html_div
