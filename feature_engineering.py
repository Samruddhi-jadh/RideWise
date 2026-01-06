import numpy as np

def add_engineered_features(df):
    df = df.copy()

    # Cyclic encoding
    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    # Interactions
    df["temp_hum"] = df["temp"] * df["hum"]
    df["temp_wind"] = df["temp"] * df["windspeed"]
    df["hum_wind"] = df["hum"] * df["windspeed"]
    df["working_weather"] = df["workingday"] * df["weathersit"]

    return df
