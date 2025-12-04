
# RideWise: Predicting Bike-Sharing Demand Based on Weather and Urban Events

##  Project Overview

This project explores how weather, season, and temporal patterns influence bike-rental demand in Washington D.C. The dataset includes both hourly and daily rental logs, enabling the analysis of short-term fluctuations (such as peak commuting hours) and long-term seasonal trends.

The primary objective of this stage was to prepare a clean, analysis-ready dataset suitable for downstream forecasting tasks.

---

##  Dataset Description

The dataset originates from the Capital Bikeshare program and includes usage data for two years (2011–2012). It combines bike usage with weather conditions, making it possible to study relationships between temperature, humidity, rainfall, windspeed, and rental behavior.

Two datasets were used:

* **day.csv** — aggregated daily counts
* **hour.csv** — detailed hourly counts

Both include features such as season, holiday flag, working-day status, temperature, humidity, windspeed, and rental count. The hourly dataset additionally provides the hour of the day.

To strengthen the analysis, both files were used, allowing comparison between hourly and daily patterns.

---

##  Dataset Source & Citation

Dataset by **Hadi Fanaee-T, University of Porto**.

> Fanaee-T, Hadi, and Gama, Joao, *Event labeling combining ensemble detectors and background knowledge*, Progress in Artificial Intelligence (2013): 1–15, Springer, doi:10.1007/s13748-013-0040-3.

Source: Kaggle (Bike Sharing Dataset) https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset?select=hour.csv

---

##  Preprocessing Steps

* **Analyzed dataset structure** — Explored info, shape, null values, column categories, and duplicates.
* **Standardized date format** — Converted `dteday` into consistent `datetime` format.
* **Validated logical consistency** — Checked contradictory combinations (e.g., `holiday=1` & `workingday=1`).
* **Merged datasets** — Concatenated aligned datasets using `pd.concat()` to build a unified analytical dataset.
* **Performed post-merge cleanup** — Removed duplicates, corrected dtypes, sorted by datetime, and exported cleaned data.

---

##  **Exploratory Data Analysis (EDA)**

* **Explored dataset structure** — Verified datatypes, index settings, missing values, and duplicates for both datasets.
* **Removed irrelevant columns** — Dropped fields like `instant`, `dteday`, `yr`, `casual`, and `registered` to retain modelling-focused features.
* **Cleaned the data** — Removed duplicate records and confirmed the absence of null values.
* **Handled outliers** — Applied the IQR method across numerical variables to remove extreme abnormal points.
* **Validated cleaning results** — Plotted boxplots, histograms, and pairplots to ensure proper distribution and pattern consistency after cleaning.

---

## **Model Training & Evaluation**

* **Prepared modeling datasets** — Loaded the feature-engineered daily and hourly files and separated predictors (`X`) from the target variable (`cnt`).

* **Split data into train/test sets** — Allocated 80% for training and 20% for testing using a fixed random state for reproducibility.

* **Trained baseline regression models** — Implemented Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor to establish initial performance benchmarks on both datasets.

* **Evaluated model performance** — Computed RMSE, MAE, and R² scores to quantify accuracy and identify the most effective model for each dataset.

* **Compared actual vs predicted values** — Visualized real and predicted rental counts using line plots to assess how closely the models follow real trends.

* **Analyzed feature importance** — Used Random Forest feature importance scores to understand which variables (e.g., temperature, hour, season) had the strongest influence on bike-rental demand.

* **Compared daily vs hourly models** — Summarized performance differences between datasets, noting that hourly data achieved higher accuracy due to richer temporal granularity.

---

