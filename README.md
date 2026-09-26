# Toronto Island Ferry Demand Forecasting & Predictive Decision Support System

## Overview

This project develops a short-term ferry ticket demand forecasting and predictive decision support system for Toronto Island Park ferry operations.

The system uses historical ferry ticket sales data and machine learning techniques to forecast short-term passenger demand at multiple forecasting horizons. The objective is to support operational planning, capacity management, and peak-demand preparation.

## Objectives

- Forecast ferry ticket demand for short-term horizons:
  - 15 minutes
  - 30 minutes
  - 1 hour
  - 2 hours
- Identify periods of increasing or peak demand.
- Compare forecasting performance across different horizons.
- Provide operational insights through an interactive Streamlit dashboard.
- Support data-driven ferry capacity and operational decision-making.

## Dataset

The project uses Toronto Island ferry ticket count data containing time-based sales information.

The dataset is processed as a time series and used to create forecasting features such as:

- Hour
- Day of week
- Month
- Weekend indicator
- Lagged demand
- Rolling mean
- Rolling standard deviation
- Rolling maximum

## Machine Learning

The project includes Random Forest forecasting models for multiple forecasting horizons.

### Forecast Horizons

| Horizon | Model |
|---|---|
| 15 minutes | Random Forest |
| 30 minutes | Random Forest |
| 1 hour | Random Forest |
| 2 hours | Random Forest |

The models use historical demand patterns and engineered time-series features to generate short-term forecasts.

## Evaluation

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Horizon-wise forecasting error
- Peak Miss Rate

The project also considers operational indicators such as forecast accuracy, error drift, confidence intervals, and peak-demand detection.

## Streamlit Dashboard

An interactive Streamlit dashboard is included to provide a user-friendly forecasting interface.

The dashboard allows users to:

- Select a forecasting horizon
- Select a forecast date and time
- Generate demand forecasts
- View historical and predicted demand
- Compare predicted and actual demand when actual data is available
- View peak-demand thresholds
- Review model information
- View forecasting performance results
- Receive operational recommendations

## Project Structure

```text
Toronto-Island-Ferry-Demand-Forcasting/
│
├── app.py
│
├── data/
│   └── ferry_forecasting_data.csv
│
├── results/
│   ├── Forecast_Accuracy_Results.csv
│   ├── Forecast_Horizon_Results.csv
│   └── Peak_Miss_Rate_Results.csv
│
├── requirements.txt
│
└── README.md
