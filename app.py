import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from visualization import plot_forecast_timeline
from feature_engineering import add_engineered_features
from forecasting import (
    recursive_forecast_day,
    recursive_forecast_hour
)
from datetime import datetime
import os
from gtts import gTTS
import tempfile
import base64
from pathlib import Path
import streamlit as st

def speak_streamlit(text):
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")

# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(
    page_title="RideWise — AI Bike Demand Forecast",
    page_icon="🚲",
    layout="wide"
)
st.markdown(
    """
    <h1 style="text-align:center; font-weight:700;">
        <span style="color:#58c472;">🚲 RideWise</span>
        <span style="color:#ffffff;">— AI Bike Demand Forecast</span>
    </h1>
    <p style='text-align:center; color:#6c757d; font-size:16px;'>
        Smart, data-driven bike rental demand forecasting using production-grade machine learning
    </p>
    """,
    unsafe_allow_html=True
)




DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
BASE_DIR = Path(__file__).parent

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model(mode: str):
    model_path = (
        BASE_DIR / "Models" / "bike_day_demand_model.pkl"
        if mode == "Day"
        else BASE_DIR / "Models" / "bike_hour_demand_model.pkl"
    )
    if not model_path.exists():
        st.error(f"❌ Model not found: {model_path}")
        st.stop()

    return joblib.load(model_path)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(180deg, #000000, #0b1220);
}
.glass {
    background: rgba(30,55,95,.75);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 40px;
}
</style>
""", unsafe_allow_html=True)


# Initialize session state FIRST (before any access)
if "forecast_history" not in st.session_state:
    st.session_state.forecast_history = []

if "last_voice_text" not in st.session_state:
    st.session_state.last_voice_text = None

# Logo
if (BASE_DIR / "logo.png").exists():
    st.sidebar.image(str(BASE_DIR / "logo.png"), width=180)

st.sidebar.title("⚙️ Forecast Settings")

mode = st.sidebar.radio(
    "Forecast Granularity",
    ["Day", "Hour"]
)


# -------------------------------
# Recent Forecasts Section
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.title("🕘 Recent Forecasts")

if st.session_state.forecast_history:
    st.sidebar.dataframe(
        pd.DataFrame(st.session_state.forecast_history),
        width="stretch"
    )
else:
    st.sidebar.info("No forecasts yet.")


# Load model AFTER mode selection
model = load_model(mode)

st.sidebar.markdown("---")
st.sidebar.caption(
    "📌 Production ML System\n"
    "- LightGBM / XGBoost\n"
    "- Feature-aligned inference\n"
    "- Recursive forecasting"
)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.divider()

# --------------------------------------------------
# Inputs (dataset-aligned)
# --------------------------------------------------

SEASON_MAP = {
    "Spring": 1,
    "Summer": 2,
    "Fall": 3,
    "Winter": 4
}

WEEKDAY_MAP = {
    "Sunday": 0,
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6
}

HOLIDAY_MAP = {
    "No": 0,
    "Yes": 1
}

WORKINGDAY_MAP = {
    "No (Weekend / Holiday)": 0,
    "Yes (Working Day)": 1
}

WEATHER_MAP = {
    "Clear / Few Clouds": 1,
    "Mist / Cloudy": 2,
    "Light Rain / Snow": 3,
    "Heavy Rain / Snow / Fog": 4
}

YEAR_MAP = {
    "2011": 0,
    "2012": 1
}

# =========================
# Streamlit UI
# =========================

st.subheader("📥 Environmental & Calendar Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    season_label = st.selectbox("Season", list(SEASON_MAP.keys()))
    season = SEASON_MAP[season_label]

    st.subheader("📅 Month Inputs")

    MONTH_NAMES = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }

    mnth_label = st.radio(
        "Select Month",
        list(MONTH_NAMES.values()),
        horizontal=True
    )

    mnth = [k for k, v in MONTH_NAMES.items() if v == mnth_label][0]


    weekday_label = st.selectbox("Day of Week", list(WEEKDAY_MAP.keys()))
    weekday = WEEKDAY_MAP[weekday_label]

with col2:
    holiday_label = st.selectbox("Is it a Holiday?", list(HOLIDAY_MAP.keys()))
    holiday = HOLIDAY_MAP[holiday_label]

    workingday_label = st.selectbox(
        "Is it a Working Day?",
        list(WORKINGDAY_MAP.keys())
    )
    workingday = WORKINGDAY_MAP[workingday_label]

    weathersit_label = st.selectbox(
        "Weather Condition",
        list(WEATHER_MAP.keys())
    )
    weathersit = WEATHER_MAP[weathersit_label]

with col3:
    st.subheader("🌦️ Weather Conditions")

    # --- REAL-WORLD INPUTS ---
    temp_c = st.slider(
        "🌡️ Temperature (°C)",
        min_value=0,
        max_value=41,
        value=20,
        help="Actual temperature in Celsius"
    )

    feels_like_c = st.slider(
        "🤒 Feels Like Temperature (°C)",
        min_value=0,
        max_value=50,
        value=25,
        help="Perceived temperature in Celsius"
    )

    humidity_pct = st.slider(
        "💧 Humidity (%)",
        min_value=0,
        max_value=100,
        value=60,
        help="Relative humidity percentage"
    )

    wind_kmh = st.slider(
        "💨 Wind Speed",
        min_value=0,
        max_value=67,
        value=15,
        help="Wind speed (dataset max = 67)"
    )

    # --- NORMALIZATION (MATCHES DATASET EXACTLY) ---
    temp = temp_c / 41
    atemp = feels_like_c / 50
    hum = humidity_pct / 100
    windspeed = wind_kmh / 67

    st.caption(
    f"🔎 Normalized values → "
    f"temp={temp:.2f}, atemp={atemp:.2f}, hum={hum:.2f}, wind={windspeed:.2f}"
)
# =========================
# Hourly Control (Only if Hour Mode)
# =========================

if mode == "Hour":
    st.subheader("🕒 Hour of Day")

    col1, col2 = st.columns(2)

    with col1:
        hour_12 = st.selectbox(
            "Hour",
            list(range(1, 13)),
            index=11  # default = 12
        )

    with col2:
        am_pm = st.selectbox(
            "AM / PM",
            ["AM", "PM"]
        )

    # Convert to 24-hour format (model-ready)
    if am_pm == "AM":
        hr = 0 if hour_12 == 12 else hour_12
    else:
        hr = 12 if hour_12 == 12 else hour_12 + 12
        
with st.container():
    if holiday == 1 and workingday == 1:
        st.warning("Holiday marked as working day. Please verify input.")

    if weekday in [0, 6] and workingday == 1:
        st.info("⚠️ Weekend selected as Working Day. Assuming special working conditions.")

    if weathersit >= 3 and temp > 0.8:
        st.info("🌧️ Severe weather with high temperature is uncommon. Forecast uncertainty may increase.")

# =========================
# Year Selector
# =========================
st.subheader("🗓️ Year Selection")

input_year = st.number_input(
    "Year",
    min_value=2011,
    max_value=2100,
    value=2025,
    step=1
)
# Encode year relative to base year
BASE_YEAR = 2011
yr = input_year - BASE_YEAR
# --------------------------------------------------
# Feature Columns (MUST match training)
# --------------------------------------------------
DAY_FEATURE_COLS = [
    "season", "mnth", "holiday", "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed",
    "mnth_sin", "mnth_cos", "weekday_sin", "weekday_cos",
    "temp_hum", "temp_wind", "hum_wind", "working_weather",
    "cnt_lag_1", "cnt_lag_7",
    "cnt_roll_mean_3", "cnt_roll_mean_7", "cnt_roll_std_7"
]

HOUR_FEATURE_COLS = [
    "season", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp",
    "atemp", "hum", "windspeed"
]
def hour_to_12h_label(hour_24: int) -> str:
    hour = hour_24 % 12
    hour = 12 if hour == 0 else hour
    am_pm = "AM" if hour_24 < 12 else "PM"
    return f"{hour}:00 {am_pm}"


if st.button("🔁 Repeat Last Forecast"):
    if st.session_state.forecast_history:
        last = st.session_state.forecast_history[-1]

        st.subheader("🔄 Replaying Last Forecast")

        replay_df = pd.DataFrame({
            "Time": last["labels"],
            "Predicted Demand": last["preds"]
        })

        st.dataframe(replay_df, use_container_width=True, hide_index=True)

        # 🔊 Speak again
        speak_streamlit(last["voice_text"])
    else:
        st.info("No previous forecast available.")

        st.info("No previous forecast available.")



# Prediction
# --------------------------------------------------
if st.button("🔮 Predict Demand", use_container_width=True):

    # Base row (aligned with dataset)
    base_row = pd.Series({
        "season": season,
        "mnth": mnth,
        "hr": hr if mode == "Hour" else 0,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weathersit,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed,

        # Safe lag defaults
        "cnt_lag_1": 300,
        "cnt_lag_7": 320,
        "cnt_lag_24": 280,
        "cnt_roll_mean_3": 310,
        "cnt_roll_mean_7": 315,
        "cnt_roll_std_7": 45
    })

    # ---------------- Day ----------------
    if mode == "Day":
        base_row = add_engineered_features(base_row)

        preds = recursive_forecast_day(
            model=model,
            last_row=base_row,
            feature_cols=DAY_FEATURE_COLS,
            steps=6
        )

        labels = [DAY_NAMES[(weekday + i) % 7] for i in range(6)]
        x_label = "Day"

    # ---------------- Hour ----------------
    else:
        preds = recursive_forecast_hour(
            model=model,
            last_row=base_row,
            feature_cols=HOUR_FEATURE_COLS,
            steps=7
        )

        labels = [
            hour_to_12h_label((hr + i ) % 24)
            for i in range(len(preds))
        ]

        x_label = "Hour"
    preds = np.maximum(0, np.round(preds).astype(int))
    def demand_insight_text(demand, mode, weathersit_label, temp_c):
        if mode == "Day":
            if demand > 6000:
                level = "very high"
                suggestion = (
                    "Consider deploying additional bikes and staff. "
                    "Peak usage is expected across major stations."
                )
            elif demand > 4000:
                level = "moderate"
                suggestion = (
                    "Bike availability should be sufficient, but monitor demand in busy areas."
                )
            else:
                level = "low"
                suggestion = (
                    "This is a good day for maintenance or redistribution of bikes."
                )
        else:  # Hour mode
            if demand > 300:
                level = "very high"
                suggestion = (
                    "High short-term demand expected. Ensure bikes are available near transit hubs."
                )
            elif demand > 150:
                level = "moderate"
                suggestion = (
                    "Steady demand expected. Normal operations should be sufficient."
                )
            else:
                level = "low"
                suggestion = (
                    "Low demand predicted. This is a good time for rebalancing bikes."
                )

        weather_note = ""
        if weathersit_label in ["Light Rain / Snow", "Heavy Rain / Snow / Fog"]:
            weather_note = " Weather conditions may slightly reduce rider comfort."
        elif temp_c > 35:
            weather_note = " High temperature may affect rider activity."

        return level, suggestion + weather_note
    


    demand = int(preds[0])

    level, suggestion = demand_insight_text(
        demand=demand,
        mode=mode,
        weathersit_label=weathersit_label,
        temp_c=temp_c
    )

    voice_text = (
        f"Expected bike demand {'today' if mode == 'Day' else 'this hour'} "
        f"is {demand} bikes, which indicates {level} demand. "
        f"{suggestion}"
    )


    # --------------------------------------------------
    # Results
    # --------------------------------------------------
    result_df = pd.DataFrame({
        x_label: labels,
        "Predicted Demand": np.round(preds).astype(int)
    })
    
    
    st.session_state.forecast_history.append({
    "Mode": mode,
    "Time": datetime.now().strftime("%A %I:%M %p"),
    "Demand": demand,
    "Weather": weathersit_label,
    "Temp (°C)": temp_c,
    "voice_text": voice_text,
    "labels": labels,
    "preds": preds.tolist()
})


        # Keep only last 10
    st.session_state.forecast_history = st.session_state.forecast_history[-10:]
    st.success(voice_text)
    speak_streamlit(voice_text)
    

    df = pd.DataFrame({
        "Time": labels,
        "Predicted Demand": preds
    })
    if mode == "Day":
        day_name = DAY_NAMES[weekday]
        day_demand = int(np.round(preds[0]))  # First day prediction

        # Demand level (daily scale)
        if day_demand > 6000:
            level = "🔥 Very High Demand"
        elif day_demand > 4000:
            level = "⚡ Moderate Demand"
        else:
            level = "🌿 Low Demand"

        st.markdown(
            f"""
            📝 **Forecast Summary**

            Based on the selected conditions, the **expected rental bike demand on
            _{day_name}_** is approximately **{day_demand} bikes** for the day.

            """
        )

    else:
        # Convert hour to 12-hour format
        display_hr = hr % 12
        display_hr = 12 if display_hr == 0 else display_hr
        ampm = "AM" if hr < 12 else "PM"

        weekday_name = DAY_NAMES[weekday]
        hour_demand = int(np.round(preds[0]))  # First hour prediction

        # Demand level (hourly scale)
        if hour_demand > 300:
            level = "🔥 Very High Demand"
        elif hour_demand > 150:
            level = "⚡ Moderate Demand"
        else:
            level = "🌿 Low Demand"

        st.markdown(
            f"""
            📝 **Forecast Summary**

            Based on the selected inputs, the **expected rental bike demand on
            _{weekday_name} at {display_hr}:00 {ampm}_**
            is approximately **{hour_demand} bikes**.

            """
        )
    

    st.success(f"Overall Demand Level: **{level}**")

    st.subheader("📌 Key Insights")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "📈 Peak Demand",
        f"{int(np.max(preds))}",
    )

    c2.metric(
        "📉 Lowest Demand",
        f"{int(np.min(preds))}",
    )

    c3.metric(
        "📊 Average Demand",
        f"{int(np.mean(preds))}",
    )

    trend = "⬆ Increasing" if preds[-1] > preds[0] else "⬇ Decreasing"
    c4.metric(
        "🔄 Trend",
        trend
    )
    # --------------------------------------------------
    # Forecast Results Table
    # --------------------------------------------------
    st.subheader("📊 Forecast Results")

    styled_df = (
    result_df
    .style
    .set_properties(**{
        "text-align": "center",
        "font-size": "14px",
        "font-weight": "500",
        "padding": "4px 6px"   # 👈 reduces cell spacing
    })
    .set_table_styles([
        {
            "selector": "th",
            "props": [
                ("text-align", "center"),
                ("font-size", "14px"),
                ("padding", "4px 6px")
            ]
        },
        {
            "selector": "td",
            "props": [
                ("padding", "4px 6px")
            ]
        }
    ])
)


    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # --------------------------------------------------
    # Interactive Forecast Timeline (BEST VERSION)
    # --------------------------------------------------
    window = 8 if mode == "Hour" else 6

    # Starting index for focus
    start_idx = hr if mode == "Hour" else weekday

    fig, full_df = plot_forecast_timeline(
        preds=np.round(preds).astype(int),   # ✅ force integer counts
        start_index=start_idx,
        mode=mode,
        window=window
    )

    st.plotly_chart(fig, use_container_width=True)


    st.success("Forecast generated successfully ✔️")
    

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption(
    "Models: LightGBM / XGBoost | "
    "Day R² ≈ 0.75 | Hour R² ≈ 0.88 | "
    "Feature-aligned Production ML System"
)






