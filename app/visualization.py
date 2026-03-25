import pandas as pd
import plotly.graph_objects as go
# ==============================
# FORECAST TIMELINE VISUALIZATION
# ==============================
def plot_forecast_timeline(
    preds: list,
    start_index: int,
    mode: str,
    window: int = 8,
):
    """
    Interactive forecast timeline with sliding focus window

    Parameters
    ----------
    preds : list
        Model predictions
    start_index : int
        Starting hour (for Hour mode) or weekday index (for Day mode)
    mode : str
        "Hour" or "Day"
    window : int
        Auto-focus window size
    """

    steps = len(preds)

    # -----------------------------
    # Build labels
    # -----------------------------
    if mode == "Hour":
        labels = [(start_index + i ) % 24 for i in range(steps)]
        labels = [f"{h}:00" for h in labels]
        x_title = "Time (Hour)"
        title = "Hourly Bike Demand Forecast"
    else:
        DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        labels = [DAY_NAMES[(start_index + i) % 7] for i in range(steps)]
        x_title = "Day"
        title = "Daily Bike Demand Forecast"

    df = pd.DataFrame({
        "Time": labels,
        "Demand": preds
    })

    # -----------------------------
    # Sliding window
    # -----------------------------
    max_start = max(0, steps - window)
    view_df = df.iloc[:window]

    # -----------------------------
    # Plot (SINGLE LINE ONLY)
    # -----------------------------
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=view_df["Time"],
            y=view_df["Demand"],
            mode="lines+markers",
            name="Predicted Demand",
            line=dict(width=3),
            marker=dict(size=8)
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Bike Demand",
        height=420,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig, df
