"""
Helper functions for generating HTML dashboards with Plotly charts.

This module provides utilities to easily add Plotly charts to the
scan_web_summary_web.html template using the addChart JavaScript function.
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Union
import plotly.graph_objects as go


def plotly_fig_to_dict(fig: go.Figure) -> dict:
    """
    Convert Plotly figure to dictionary format compatible with addChart function.
    
    Args:
        fig: Plotly figure object
        
    Returns:
        dict: Chart data in format {data: [...], layout: {...}, config: {...}}
    """
    return {
        'data': list(fig.data),
        'layout': dict(fig.layout) if fig.layout else {},
        'config': dict(fig.config) if hasattr(fig, 'config') and fig.config else {}
    }


def generate_charts_script(charts: List[Tuple[str, go.Figure, Optional[str]]]) -> str:
    """
    Generate JavaScript code to add multiple charts using addChart function.
    
    Args:
        charts: List of tuples (title, plotly_figure, chart_id)
                chart_id is optional - will be auto-generated if not provided
        
    Returns:
        str: JavaScript code as string
    """
    script_lines = []
    
    for i, chart_info in enumerate(charts):
        if len(chart_info) == 2:
            title, fig = chart_info
            chart_id = f'chart-{i+1}'
        elif len(chart_info) == 3:
            title, fig, chart_id = chart_info
            if chart_id is None:
                chart_id = f'chart-{i+1}'
        else:
            raise ValueError(
                "Each chart should be (title, fig) or (title, fig, chart_id). "
                f"Got: {chart_info}"
            )
        
        # Convert Plotly figure to JSON
        chart_data = plotly_fig_to_dict(fig)
        chart_json = json.dumps(chart_data, cls=go.utils.PlotlyJSONEncoder)
        
        # Create JavaScript call
        script_line = f"addChart({json.dumps(title)}, {chart_json}, {json.dumps(chart_id)});"
        script_lines.append(script_line)
    
    return '\n            '.join(script_lines)


def create_dashboard_html(
    charts: List[Tuple[str, go.Figure, Optional[str]]],
    template_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Create HTML dashboard with charts embedded using addChart function.
    
    Args:
        charts: List of tuples (title, plotly_figure, chart_id)
                chart_id is optional
        template_path: Path to HTML template. If None, uses default template.
        output_path: Path to save HTML file. If None, returns HTML as string.
        
    Returns:
        str: Generated HTML content
    """
    # Get template path
    if template_path is None:
        template_path = Path(__file__).parent / 'scan_web_summary_web.html'
    else:
        template_path = Path(template_path)
    
    # Load template
    with open(template_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Generate charts script
    charts_script = generate_charts_script(charts)
    
    # Insert script before initialization
    insertion_script = f"""
        // Add charts on page load
        window.addEventListener('DOMContentLoaded', function() {{
            {charts_script}
        }});
        
        // Initialize on page load"""
    
    html_content = html_content.replace(
        '        // Initialize on page load',
        insertion_script
    )
    
    # Save or return
    if output_path:
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Dashboard HTML saved to: {output_path}")
    
    return html_content


# Convenience function for quick usage
def add_charts_to_dashboard(
    charts: List[Tuple[str, go.Figure]],
    output_path: Union[str, Path]
):
    """
    Quick function to create dashboard with charts.
    
    Args:
        charts: List of (title, figure) tuples
        output_path: Path to save HTML file
    """
    create_dashboard_html(charts, output_path=output_path)


# Example usage
if __name__ == '__main__':
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    
    # Create sample charts
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], name='Data'))
    fig1.update_layout(title='Example Chart 1')
    
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    fig2 = px.bar(df, x='x', y='y', title='Example Chart 2')
    
    # Create dashboard
    charts = [
        ('Chart 1', fig1),
        ('Chart 2', fig2, 'custom-id'),  # With custom ID
    ]
    
    output = Path(__file__).parent / 'test_dashboard.html'
    create_dashboard_html(charts, output_path=output)

