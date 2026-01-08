import numpy as np
import pandas as pd

# ==============================
# SHARED: LIGHTGBM SAFE INPUT
# ==============================
def prepare_lgbm_input(history, feature_cols):
    """
    Ensures LightGBM feature consistency:
    - same columns
    - same order
    - no missing features
    """
    X = pd.DataFrame([history])

    # Add missing features
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    # Keep only training features & correct order
    X = X[feature_cols]

    return X


# ==============================
# DAY RECURSIVE FORECAST
# ==============================
def recursive_forecast_day(model, last_known_row, feature_cols, steps=6):
    """
    Recursive multi-day forecast (Day-level)
    Fully LightGBM-safe
    """
    history = last_known_row.copy()
    preds = []

    for _ in range(steps):
        X = prepare_lgbm_input(history, feature_cols)
        pred = model.predict(X)[0]
        preds.append(pred)

        # --- Update lags ---
        history["cnt_lag_7"] = history.get("cnt_lag_1", pred)
        history["cnt_lag_1"] = pred

        # --- Rolling stats ---
        history["cnt_roll_mean_3"] = (
            history.get("cnt_roll_mean_3", pred) * 2 + pred
        ) / 3

        history["cnt_roll_mean_7"] = (
            history.get("cnt_roll_mean_7", pred) * 6 + pred
        ) / 7

        # --- Calendar ---
        history["weekday"] = (history["weekday"] + 1) % 7
        history["mnth"] = (history["mnth"] % 12) + 1

        # --- Cyclic ---
        history["mnth_sin"] = np.sin(2 * np.pi * history["mnth"] / 12)
        history["mnth_cos"] = np.cos(2 * np.pi * history["mnth"] / 12)
        history["weekday_sin"] = np.sin(2 * np.pi * history["weekday"] / 7)
        history["weekday_cos"] = np.cos(2 * np.pi * history["weekday"] / 7)

    return preds


# ==============================
# HOUR FEATURE ENGINEERING
# ==============================
def update_engineered_features_hour(row):
    row["mnth_sin"] = np.sin(2 * np.pi * row["mnth"] / 12)
    row["mnth_cos"] = np.cos(2 * np.pi * row["mnth"] / 12)

    row["weekday_sin"] = np.sin(2 * np.pi * row["weekday"] / 7)
    row["weekday_cos"] = np.cos(2 * np.pi * row["weekday"] / 7)

    row["hr_sin"] = np.sin(2 * np.pi * row["hr"] / 24)
    row["hr_cos"] = np.cos(2 * np.pi * row["hr"] / 24)

    return row


# ==============================
# HOUR RECURSIVE FORECAST
# ==============================
def recursive_forecast_hour(model, last_known_row, feature_cols, steps=24):
    """
    Recursive multi-hour forecasting
    Fully LightGBM-safe (22 features)
    """

    history = last_known_row.copy()
    preds = []

    # ---- Ensure required lag & rolling features exist ----
    for col in [
        "cnt_lag_1", "cnt_lag_24", "cnt_lag_168",
        "cnt_roll_mean_3", "cnt_roll_mean_24"
    ]:
        if col not in history:
            history[col] = 0

    for _ in range(steps):

        # --- Predict safely ---
        X = prepare_lgbm_input(history, feature_cols)
        pred = model.predict(X)[0]
        preds.append(pred)

        # --- Update lag features ---
        history["cnt_lag_168"] = history["cnt_lag_24"]
        history["cnt_lag_24"] = history["cnt_lag_1"]
        history["cnt_lag_1"] = pred

        # --- Update rolling means ---
        history["cnt_roll_mean_3"] = (
            history["cnt_roll_mean_3"] * 2 + pred
        ) / 3

        history["cnt_roll_mean_24"] = (
            history["cnt_roll_mean_24"] * 23 + pred
        ) / 24

        # --- Advance time ---
        history["hr"] += 1

        if history["hr"] == 24:
            history["hr"] = 0
            history["weekday"] = (history["weekday"] + 1) % 7

            # Dataset-style month rollover
            if history["weekday"] == 0:
                history["mnth"] = (history["mnth"] % 12) + 1

        # --- Recompute cyclic features ---
        history = update_engineered_features_hour(history)

    return preds
