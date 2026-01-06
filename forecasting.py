import numpy as np
import pandas as pd
# ==============================
# DAY RECURSIVE FORECAST
# ==============================
def recursive_forecast_day(model, last_row, feature_cols, steps=6):
    """
    Recursive multi-day forecast
    """
    history = last_row.copy()
    preds = []

    for _ in range(steps):
        X = history[feature_cols].values.reshape(1, -1)
        pred = model.predict(X)[0]
        preds.append(pred)

        # --- Update lags ---
        history["cnt_lag_7"] = history["cnt_lag_1"]
        history["cnt_lag_1"] = pred

        # --- Update rolling stats ---
        history["cnt_roll_mean_3"] = (history["cnt_roll_mean_3"] * 2 + pred) / 3
        history["cnt_roll_mean_7"] = (history["cnt_roll_mean_7"] * 6 + pred) / 7

        # --- Advance calendar ---
        history["weekday"] = (history["weekday"] + 1) % 7
        history["mnth"] = (history["mnth"] % 12) + 1

        # --- Recompute cyclic features ---
        history["mnth_sin"] = np.sin(2 * np.pi * history["mnth"] / 12)
        history["mnth_cos"] = np.cos(2 * np.pi * history["mnth"] / 12)
        history["weekday_sin"] = np.sin(2 * np.pi * history["weekday"] / 7)
        history["weekday_cos"] = np.cos(2 * np.pi * history["weekday"] / 7)

    return preds

# ==============================
# HOUR RECURSIVE FORECAST
# ==============================
def recursive_forecast_hour(model, last_row, feature_cols, steps=24):
    """
    Recursive multi-hour forecast
    """
    history = last_row.copy()
    preds = []

    for _ in range(steps):
        X = history[feature_cols].values.reshape(1, -1)
        pred = model.predict(X)[0]
        preds.append(pred)

        # --- Update lags ---
        history["cnt_lag_24"] = history["cnt_lag_1"]
        history["cnt_lag_1"] = pred

        # --- Advance hour ---
        history["hr"] = (history["hr"] + 1) % 24

        # --- Day rollover ---
        if history["hr"] == 0:
            history["weekday"] = (history["weekday"] + 1) % 7

    return preds
