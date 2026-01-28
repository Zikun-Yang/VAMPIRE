"""
Example: How to use addChart function in Python with Plotly

This script demonstrates how to generate Plotly charts in Python
and embed them into the HTML template using the addChart JavaScript function.
"""

import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


def create_chart_data(fig):
    """
    Convert Plotly figure to JSON format that can be used with addChart function.
    
    Args:
        fig: Plotly figure object
        
    Returns:
        dict: Chart data in format {data: [...], layout: {...}, config: {...}}
    """
    return {
        'data': fig.data,
        'layout': fig.layout,
        'config': fig.config if hasattr(fig, 'config') else {}
    }


def generate_html_with_charts(charts_list, output_path):
    """
    Generate HTML file with charts using addChart function.
    
    Args:
        charts_list: List of tuples (title, plotly_figure, chart_id)
                    chart_id is optional
        output_path: Path to save the HTML file
    """
    # Load HTML template
    template_path = Path(__file__).parent / 'scan_web_summary_web.html'
    with open(template_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Generate JavaScript code to add charts
    chart_scripts = []
    for i, chart_info in enumerate(charts_list):
        if len(chart_info) == 2:
            title, fig = chart_info
            chart_id = f'chart-{i+1}'
        elif len(chart_info) == 3:
            title, fig, chart_id = chart_info
        else:
            raise ValueError("Each chart should be (title, fig) or (title, fig, chart_id)")
        
        # Convert Plotly figure to JSON
        chart_data = create_chart_data(fig)
        chart_json = json.dumps(chart_data, cls=go.utils.PlotlyJSONEncoder)
        
        # Create JavaScript call to addChart
        script = f"addChart({json.dumps(title)}, {chart_json}, {json.dumps(chart_id)});"
        chart_scripts.append(script)
    
    # Combine all chart scripts
    all_charts_script = '\n            '.join(chart_scripts)
    
    # Insert script before closing body tag
    insertion_point = '        // Initialize on page load'
    insertion_script = f"""
        // Add charts on page load
        window.addEventListener('DOMContentLoaded', function() {{
            {all_charts_script}
        }});
        
        {insertion_point}"""
    
    html_content = html_content.replace(
        f'        {insertion_point}',
        insertion_script
    )
    
    # Save HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML file generated: {output_path}")
    print(f"Added {len(charts_list)} charts")


# Example usage
if __name__ == '__main__':
    # Example 1: Create a simple line chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=[1, 2, 3, 4, 5],
        y=[10, 11, 12, 13, 14],
        mode='lines+markers',
        name='Series 1'
    ))
    fig1.update_layout(
        title='Simple Line Chart',
        xaxis_title='X Axis',
        yaxis_title='Y Axis'
    )
    
    # Example 2: Create a bar chart using plotly express
    import pandas as pd
    df = pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D'],
        'Value': [20, 35, 30, 25]
    })
    fig2 = px.bar(df, x='Category', y='Value', title='Bar Chart Example')
    
    # Example 3: Create a scatter plot
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=[1, 2, 3, 4, 5, 6, 7, 8],
        y=[2, 4, 6, 8, 10, 12, 14, 16],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Scatter Data'
    ))
    fig3.update_layout(title='Scatter Plot Example')
    
    # Create list of charts
    charts = [
        ('Line Chart Example', fig1),
        ('Bar Chart Example', fig2),
        ('Scatter Plot Example', fig3, 'custom-scatter-id'),  # With custom ID
    ]
    
    # Generate HTML
    output_file = Path(__file__).parent / 'example_output.html'
    generate_html_with_charts(charts, output_file)


# Alternative method: Direct HTML embedding (simpler but less flexible)
def generate_html_simple_method(charts_list, output_path):
    """
    Simpler method: Embed Plotly HTML directly into the template.
    This method doesn't use addChart function but embeds charts directly.
    
    Args:
        charts_list: List of tuples (title, plotly_figure)
        output_path: Path to save the HTML file
    """
    template_path = Path(__file__).parent / 'scan_web_summary_web.html'
    with open(template_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Generate chart HTML
    charts_html = []
    for title, fig in charts_list:
        chart_html = fig.to_html(include_plotlyjs=False, div_id=f"chart-{len(charts_html)+1}")
        # Extract just the div and script
        import re
        div_match = re.search(r'<div[^>]*id="[^"]*"[^>]*>.*?</div>', chart_html, re.DOTALL)
        script_match = re.search(r'<script[^>]*>.*?</script>', chart_html, re.DOTALL)
        
        card_html = f"""
            <div class="card">
                <div class="card-title">{title}</div>
                <div class="chart-container">
                    {div_match.group(0) if div_match else '<div class="chart-placeholder">Chart placeholder</div>'}
                </div>
            </div>
        """
        charts_html.append(card_html)
        
        # Add script if exists
        if script_match:
            charts_html.append(f"        {script_match.group(0)}")
    
    # Insert charts into container
    charts_html_str = '\n'.join(charts_html)
    html_content = html_content.replace(
        '<div class="dashboard-grid" id="charts-container">',
        f'<div class="dashboard-grid" id="charts-container">\n{charts_html_str}'
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML file generated (simple method): {output_path}")

