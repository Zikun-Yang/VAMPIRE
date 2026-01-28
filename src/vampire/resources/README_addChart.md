# How to Use addChart Function in Python

This guide explains how to use the `addChart` JavaScript function from Python to dynamically add Plotly charts to the HTML dashboard.

## Quick Start

### Method 1: Using the Helper Function (Recommended)

```python
from pathlib import Path
from vampire.resources.html_chart_helper import create_dashboard_html
import plotly.graph_objects as go
import plotly.express as px

# Create your Plotly charts
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13]))
fig1.update_layout(title='My Chart')

fig2 = px.bar(x=['A', 'B', 'C'], y=[1, 3, 2], title='Bar Chart')

# Create list of charts: (title, figure, chart_id)
# chart_id is optional
charts = [
    ('Scatter Plot', fig1),
    ('Bar Chart', fig2, 'my-custom-id'),  # Optional custom ID
]

# Generate HTML
output_path = Path('dashboard.html')
create_dashboard_html(charts, output_path=output_path)
```

### Method 2: Manual JavaScript Injection

```python
import json
from pathlib import Path
import plotly.graph_objects as go

# Load HTML template
template_path = Path('scan_web_summary_web.html')
with open(template_path, 'r') as f:
    html = f.read()

# Create your chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
fig.update_layout(title='My Chart')

# Convert to JSON
chart_data = {
    'data': list(fig.data),
    'layout': dict(fig.layout),
    'config': {}
}
chart_json = json.dumps(chart_data, cls=go.utils.PlotlyJSONEncoder)

# Generate JavaScript call
script = f"""
window.addEventListener('DOMContentLoaded', function() {{
    addChart('My Chart', {chart_json}, 'chart-1');
}});
"""

# Insert before initialization
html = html.replace(
    '        // Initialize on page load',
    script + '\n        // Initialize on page load'
)

# Save
with open('output.html', 'w') as f:
    f.write(html)
```

## Function Reference

### `create_dashboard_html()`

Main function to create dashboard with charts.

**Parameters:**
- `charts`: List of tuples `(title, figure, chart_id)` where:
  - `title` (str): Chart title
  - `figure` (go.Figure): Plotly figure object
  - `chart_id` (str, optional): Custom chart ID (auto-generated if not provided)
- `template_path` (Path, optional): Path to HTML template (defaults to scan_web_summary_web.html)
- `output_path` (Path, optional): Path to save HTML file (if None, returns HTML string)

**Returns:**
- `str`: Generated HTML content

**Example:**
```python
charts = [
    ('Chart 1', fig1),
    ('Chart 2', fig2, 'custom-id'),
]
html = create_dashboard_html(charts, output_path='dashboard.html')
```

### `plotly_fig_to_dict()`

Convert Plotly figure to dictionary format.

**Parameters:**
- `fig` (go.Figure): Plotly figure

**Returns:**
- `dict`: Chart data in format `{data: [...], layout: {...}, config: {...}}`

### `generate_charts_script()`

Generate JavaScript code for multiple charts.

**Parameters:**
- `charts`: List of chart tuples

**Returns:**
- `str`: JavaScript code as string

## Complete Example

```python
from pathlib import Path
from vampire.resources.html_chart_helper import create_dashboard_html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Example 1: Line chart
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 11, 12, 13, 14],
    mode='lines+markers',
    name='Series 1'
))
fig1.update_layout(
    title='Time Series Data',
    xaxis_title='Time',
    yaxis_title='Value'
)

# Example 2: Bar chart using plotly express
df = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Value': [20, 35, 30, 25]
})
fig2 = px.bar(df, x='Category', y='Value', title='Category Comparison')

# Example 3: Scatter plot
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[2, 4, 6, 8, 10],
    mode='markers',
    marker=dict(size=10, color='red')
))
fig3.update_layout(title='Scatter Plot')

# Create dashboard
charts = [
    ('Time Series', fig1),
    ('Bar Chart', fig2),
    ('Scatter Plot', fig3, 'scatter-plot'),  # With custom ID
]

output_path = Path('my_dashboard.html')
create_dashboard_html(charts, output_path=output_path)
print(f"Dashboard created: {output_path}")
```

## Notes

1. **Chart IDs**: If you don't provide a chart_id, it will be auto-generated as `chart-1`, `chart-2`, etc.

2. **Plotly Versions**: Works with both `plotly.graph_objects` and `plotly.express`.

3. **Chart Updates**: Charts are added when the page loads. You can also call `addChart()` manually from browser console.

4. **Performance**: For many charts (>10), consider lazy loading or pagination.

5. **Customization**: You can modify the HTML template to change styling, add more controls, etc.

## Troubleshooting

**Charts not showing?**
- Check browser console for JavaScript errors
- Ensure Plotly library is loaded (check network tab)
- Verify chart data is valid JSON

**Charts overlapping?**
- Check CSS grid layout in template
- Adjust `minmax()` values in `.dashboard-grid`

**Export not working?**
- Ensure Plotly.js is loaded from CDN
- Check browser permissions for downloads

