"""
AdamOps Dashboard Module

Simple monitoring dashboard.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from adamops.utils.logging import get_logger
from adamops.monitoring.performance import PerformanceMonitor

logger = get_logger(__name__)


def generate_dashboard_html(
    monitors: Dict[str, PerformanceMonitor],
    title: str = "AdamOps Monitoring Dashboard"
) -> str:
    """
    Generate HTML dashboard for monitoring.
    
    Args:
        monitors: Dict of model names to monitors.
        title: Dashboard title.
    
    Returns:
        HTML string.
    """
    
    model_cards = ""
    for name, monitor in monitors.items():
        summary = monitor.summary()
        metrics = summary.get("latest_metrics", {})
        
        metrics_html = ""
        for metric, value in metrics.items():
            if isinstance(value, float):
                metrics_html += f'<div class="metric"><span class="label">{metric}</span><span class="value">{value:.4f}</span></div>'
        
        model_cards += f'''
        <div class="card">
            <h3>{name}</h3>
            <p>Entries: {summary.get("entries", 0)}</p>
            <p>Last update: {summary.get("latest_timestamp", "N/A")}</p>
            <div class="metrics">{metrics_html}</div>
        </div>
        '''
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #4a90d9; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background: #16213e; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .card h3 {{ color: #4a90d9; margin-top: 0; }}
        .metrics {{ margin-top: 15px; }}
        .metric {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #333; }}
        .label {{ color: #888; }}
        .value {{ font-weight: bold; color: #4ecdc4; }}
        .timestamp {{ color: #666; font-size: 12px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="grid">{model_cards}</div>
        <p class="timestamp">Generated: {datetime.now().isoformat()}</p>
    </div>
</body>
</html>'''
    
    return html


def save_dashboard(
    monitors: Dict[str, PerformanceMonitor],
    output_path: str,
    title: str = "AdamOps Monitoring Dashboard"
):
    """Save dashboard to HTML file."""
    html = generate_dashboard_html(monitors, title)
    with open(output_path, 'w') as f:
        f.write(html)
    logger.info(f"Dashboard saved to {output_path}")


def create_streamlit_dashboard(monitors: Dict[str, PerformanceMonitor]) -> str:
    """Generate Streamlit dashboard code."""
    return '''"""AdamOps Monitoring Dashboard
Run with: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from adamops.monitoring.performance import PerformanceMonitor

st.set_page_config(page_title="AdamOps Monitor", layout="wide")
st.title("AdamOps Monitoring Dashboard")

# Load monitors (customize paths)
# monitor = PerformanceMonitor("model_name")

# Placeholder content
st.subheader("Model Performance")
st.info("Configure monitors to see metrics")

# Example metrics display
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", "0.95", "+0.02")
col2.metric("Latency", "45ms", "-5ms")
col3.metric("Predictions", "1,234", "+100")
'''
