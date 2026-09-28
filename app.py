# ============================================================
# TORONTO ISLAND FERRY DEMAND FORECASTING
# PHASE 7I - FORECAST VISUALIZATION
# ============================================================

import os
import warnings
from datetime import datetime, timedelta

import joblib
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Toronto Island Ferry Demand Forecasting",
    page_icon="⛴️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ferry_forecasting_data.csv"
)

MODULES_DIR = os.path.join(
    BASE_DIR,
    "modules"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_FILES = {
    "15 Minutes": "random_forest_15m.pkl",
    "30 Minutes": "random_forest_30m.pkl",
    "1 Hour": "random_forest_1h.pkl",
    "2 Hours": "random_forest_2h.pkl"
}

HORIZON_MINUTES = {
    "15 Minutes": 15,
    "30 Minutes": 30,
    "1 Hour": 60,
    "2 Hours": 120
}

# Hugging Face model repository
MODEL_REPO_ID = "someswar05/toronto-ferry-demand-models"

# ============================================================
# FEATURE CONFIGURATION
# ============================================================

EXPECTED_FEATURES = [
    "Hour",
    "DayOfWeek",
    "Month",
    "IsWeekend",
    "Lag_1",
    "Lag_2",
    "Lag_4",
    "Lag_8",
    "Rolling_Mean_4",
    "Rolling_Std_4",
    "Rolling_Max_4"
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .description {
        font-size: 16px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .small-note {
        font-size: 13px;
        opacity: 0.8;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⛴️ Toronto Island Ferry Demand Forecasting</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Predictive Decision Support System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="description">'
    'Short-term ferry ticket demand forecasting for operational planning, '
    'staffing and crowd management.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    # --------------------------------------------------------
    # Find timestamp column
    # --------------------------------------------------------

    timestamp_column = None

    possible_timestamp_columns = [
        "Timestamp",
        "timestamp",
        "DateTime",
        "Datetime",
        "datetime",
        "Date",
        "date"
    ]

    for column in possible_timestamp_columns:
        if column in data.columns:
            timestamp_column = column
            break

    if timestamp_column is None:
        raise ValueError(
            "Timestamp column was not found in the dataset."
        )

    # --------------------------------------------------------
    # Convert timestamp
    # --------------------------------------------------------

    data[timestamp_column] = pd.to_datetime(
        data[timestamp_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[timestamp_column]
    )

    # Rename timestamp internally
    if timestamp_column != "Timestamp":
        data = data.rename(
            columns={timestamp_column: "Timestamp"}
        )

    # --------------------------------------------------------
    # Find sales column
    # --------------------------------------------------------

    sales_column = None

    possible_sales_columns = [
        "Sales Count",
        "Sales_Count",
        "SalesCount",
        "sales_count",
        "Sales"
    ]

    for column in possible_sales_columns:
        if column in data.columns:
            sales_column = column
            break

    if sales_column is None:
        raise ValueError(
            "Sales Count column was not found in the dataset."
        )

    if sales_column != "Sales Count":
        data = data.rename(
            columns={sales_column: "Sales Count"}
        )

    # --------------------------------------------------------
    # Sort data
    # --------------------------------------------------------

    data = data.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Convert sales to numeric
    # --------------------------------------------------------

    data["Sales Count"] = pd.to_numeric(
        data["Sales Count"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid sales
    # --------------------------------------------------------

    data = data.dropna(
        subset=["Sales Count"]
    )

    return data


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_filename):

    # Download the selected model from Hugging Face
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=model_filename
    )

    # Load the downloaded Random Forest model
    model = joblib.load(model_path)

    return model


# ============================================================
# LOAD RESULTS FILES
# ============================================================

@st.cache_data
def load_result_file(filename):

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    if os.path.exists(path):

        try:
            return pd.read_csv(path)

        except Exception:
            return None

    return None


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features_from_timestamp(
    data,
    timestamp
):

    timestamp = pd.Timestamp(timestamp)

    # --------------------------------------------------------
    # Find historical rows up to selected timestamp
    # --------------------------------------------------------

    history = data[
        data["Timestamp"] <= timestamp
    ].copy()

    if len(history) < 8:

        raise ValueError(
            "Not enough historical records before the selected "
            "timestamp to create lag features."
        )

    history = history.sort_values(
        "Timestamp"
    )

    sales_values = history["Sales Count"].values

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    lag_1 = sales_values[-1]
    lag_2 = sales_values[-2]
    lag_4 = sales_values[-4]
    lag_8 = sales_values[-8]

    # --------------------------------------------------------
    # Rolling statistics
    # --------------------------------------------------------

    recent_4 = sales_values[-4:]

    rolling_mean_4 = np.mean(
        recent_4
    )

    rolling_std_4 = np.std(
        recent_4,
        ddof=1
    ) if len(recent_4) > 1 else 0

    rolling_max_4 = np.max(
        recent_4
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    features = {
        "Hour": timestamp.hour,
        "DayOfWeek": timestamp.dayofweek,
        "Month": timestamp.month,
        "IsWeekend": int(timestamp.dayofweek >= 5),

        "Lag_1": lag_1,
        "Lag_2": lag_2,
        "Lag_4": lag_4,
        "Lag_8": lag_8,

        "Rolling_Mean_4": rolling_mean_4,
        "Rolling_Std_4": rolling_std_4,
        "Rolling_Max_4": rolling_max_4
    }

    feature_df = pd.DataFrame(
        [features]
    )

    return feature_df


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(
    model,
    feature_df
):

    # --------------------------------------------------------
    # If model remembers feature names
    # --------------------------------------------------------

    if hasattr(model, "feature_names_in_"):

        required_features = list(
            model.feature_names_in_
        )

        missing_features = [
            feature
            for feature in required_features
            if feature not in feature_df.columns
        ]

        if missing_features:

            raise ValueError(
                "Model requires features that are not available: "
                + ", ".join(missing_features)
            )

        feature_df = feature_df[
            required_features
        ]

    else:

        feature_df = feature_df[
            EXPECTED_FEATURES
        ]

    return feature_df


# ============================================================
# GENERATE FORECAST
# ============================================================

def generate_forecast(
    data,
    model,
    selected_timestamp
):

    feature_df = create_features_from_timestamp(
        data,
        selected_timestamp
    )

    model_input = prepare_model_input(
        model,
        feature_df
    )

    prediction = model.predict(
        model_input
    )

    prediction_value = float(
        prediction[0]
    )

    # Demand cannot be negative
    prediction_value = max(
        0,
        prediction_value
    )

    return prediction_value, feature_df


# ============================================================
# FIND ACTUAL VALUE
# ============================================================

def find_actual_value(
    data,
    timestamp,
    tolerance_minutes=5
):

    timestamp = pd.Timestamp(timestamp)

    differences = (
        data["Timestamp"] - timestamp
    ).abs()

    if differences.empty:
        return None

    closest_index = differences.idxmin()

    closest_difference = differences.loc[
        closest_index
    ]

    if closest_difference <= pd.Timedelta(
        minutes=tolerance_minutes
    ):

        return float(
            data.loc[
                closest_index,
                "Sales Count"
            ]
        )

    return None


# ============================================================
# HISTORICAL DATA FOR CHART
# ============================================================

def get_historical_data(
    data,
    timestamp,
    periods=24
):

    timestamp = pd.Timestamp(timestamp)

    historical = data[
        data["Timestamp"] <= timestamp
    ].copy()

    historical = historical.tail(
        periods
    )

    return historical


# ============================================================
# PEAK THRESHOLD
# ============================================================

def calculate_peak_threshold(data):

    # --------------------------------------------------------
    # Use 95th percentile as peak demand threshold
    # --------------------------------------------------------

    threshold = data[
        "Sales Count"
    ].quantile(0.95)

    return float(
        threshold
    )


# ============================================================
# OPERATIONAL STATUS
# ============================================================

def get_operational_status(
    prediction,
    peak_threshold,
    average_demand
):

    # --------------------------------------------------------
    # High demand
    # --------------------------------------------------------

    if prediction >= peak_threshold:

        return (
            "🔴 HIGH DEMAND EXPECTED",
            "High demand is expected. "
            "Consider increased staffing, crowd-management "
            "readiness and proactive operational planning.",
            "high"
        )

    # --------------------------------------------------------
    # Moderate demand
    # --------------------------------------------------------

    elif prediction >= average_demand:

        return (
            "🟡 MODERATE DEMAND EXPECTED",
            "Forecasted demand is around or above the "
            "historical average. Maintain normal operational "
            "readiness and monitor demand.",
            "moderate"
        )

    # --------------------------------------------------------
    # Low demand
    # --------------------------------------------------------

    else:

        return (
            "🟢 LOW DEMAND EXPECTED",
            "Normal staffing is likely sufficient. "
            "Continue routine monitoring.",
            "low"
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Forecast Controls"
)

st.sidebar.markdown(
    "### Forecast Date & Time"
)


# ============================================================
# DATE RANGE
# ============================================================

minimum_date = data_min_date = None

try:

    data = load_data()

    minimum_date = data[
        "Timestamp"
    ].min().date()

    maximum_date = data[
        "Timestamp"
    ].max().date()

except Exception as e:

    st.error(
        str(e)
    )

    st.stop()


# ============================================================
# DATE SELECTOR
# ============================================================

selected_date = st.sidebar.date_input(
    "Select Date",
    value=maximum_date,
    min_value=minimum_date,
    max_value=maximum_date
)


# ============================================================
# TIME SELECTOR
# ============================================================

selected_time = st.sidebar.time_input(
    "Select Time",
    value=datetime.strptime(
        "12:00",
        "%H:%M"
    ).time()
)


# ============================================================
# HORIZON SELECTOR
# ============================================================

selected_horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    options=[
        "15 Minutes",
        "30 Minutes",
        "1 Hour",
        "2 Hours"
    ]
)


# ============================================================
# BUILD FORECAST ORIGIN
# ============================================================

forecast_origin = pd.Timestamp(
    datetime.combine(
        selected_date,
        selected_time
    )
)

horizon_minutes = HORIZON_MINUTES[
    selected_horizon
]

forecast_timestamp = (
    forecast_origin
    + pd.Timedelta(
        minutes=horizon_minutes
    )
)


# ============================================================
# VALIDATE SELECTED TIME
# ============================================================

if forecast_origin < data["Timestamp"].min():

    st.error(
        "Selected date/time is earlier than the available "
        "historical data."
    )

    st.stop()


if forecast_origin > data["Timestamp"].max():

    st.warning(
        "The selected forecast origin is after the last "
        "available dataset timestamp. The model can still "
        "attempt a forecast using the latest historical "
        "observations."
    )


# ============================================================
# LOAD SELECTED MODEL
# ============================================================

model_filename = MODEL_FILES[
    selected_horizon
]


try:

    model = load_model(
        model_filename
    )

except Exception as e:

    st.error(
        f"Unable to load forecasting module:\n\n{e}"
    )

    st.stop()


# ============================================================
# GENERATE FORECAST
# ============================================================

try:

    prediction, feature_df = generate_forecast(
        data,
        model,
        forecast_origin
    )

except Exception as e:

    st.error(
        f"Prediction error: {e}"
    )

    st.info(
        "This usually means the features used by the saved "
        "model do not match the dashboard feature engineering."
    )

    st.stop()


# ============================================================
# ACTUAL VALUE
# ============================================================

actual_value = find_actual_value(
    data,
    forecast_timestamp
)


# ============================================================
# HISTORICAL AVERAGE
# ============================================================

historical_average = float(
    data["Sales Count"].mean()
)


# ============================================================
# PEAK THRESHOLD
# ============================================================

peak_threshold = calculate_peak_threshold(
    data
)


# ============================================================
# OPERATIONAL STATUS
# ============================================================

status_title, status_description, status_type = (
    get_operational_status(
        prediction,
        peak_threshold,
        historical_average
    )
)


# ============================================================
# FORECAST OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📊 Forecast Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Predicted Demand",
        f"{prediction:.0f} tickets"
    )


with col2:

    st.metric(
        "Forecast Target",
        forecast_timestamp.strftime(
            "%Y-%m-%d %H:%M"
        )
    )


with col3:

    st.metric(
        "Peak Threshold",
        f"{peak_threshold:.0f} tickets"
    )


with col4:

    st.metric(
        "Forecast Horizon",
        selected_horizon
    )


# ============================================================
# OPERATIONAL DEMAND STATUS
# ============================================================

st.markdown(
    '<div class="section-title">🚨 Operational Demand Status</div>',
    unsafe_allow_html=True
)


if status_type == "high":

    st.error(
        f"**{status_title}**"
    )

elif status_type == "moderate":

    st.warning(
        f"**{status_title}**"
    )

else:

    st.success(
        f"**{status_title}**"
    )


st.write(
    status_description
)


# ============================================================
# FORECAST DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">🔎 Forecast Details</div>',
    unsafe_allow_html=True
)

detail_col1, detail_col2 = st.columns(2)


with detail_col1:

    st.write(
        f"**Selected Date:** "
        f"{selected_date}"
    )

    st.write(
        f"**Selected Time:** "
        f"{selected_time.strftime('%H:%M')}"
    )

    st.write(
        f"**Forecast Origin:** "
        f"{forecast_origin}"
    )


with detail_col2:

    st.write(
        f"**Forecast Timestamp:** "
        f"{forecast_timestamp}"
    )

    st.write(
        "**Model:** Random Forest"
    )

    st.write(
        f"**Model Module:** "
        f"{model_filename}"
    )


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

st.markdown(
    '<div class="section-title">📈 Actual vs Predicted Demand</div>',
    unsafe_allow_html=True
)


historical_chart_data = get_historical_data(
    data,
    forecast_origin,
    periods=24
)


fig_history = go.Figure()


# ------------------------------------------------------------
# Historical demand line
# ------------------------------------------------------------

fig_history.add_trace(
    go.Scatter(
        x=historical_chart_data["Timestamp"],
        y=historical_chart_data["Sales Count"],
        mode="lines+markers",
        name="Historical Demand",
        hovertemplate=(
            "%{x}<br>"
            "Demand: %{y:.0f} tickets"
            "<extra></extra>"
        )
    )
)


# ------------------------------------------------------------
# Forecast point
# ------------------------------------------------------------

fig_history.add_trace(
    go.Scatter(
        x=[forecast_timestamp],
        y=[prediction],
        mode="markers",
        name="Forecast",
        marker=dict(
            size=14,
            symbol="diamond"
        ),
        hovertemplate=(
            "%{x}<br>"
            "Forecast: %{y:.0f} tickets"
            "<extra></extra>"
        )
    )
)


# ------------------------------------------------------------
# Actual future value
# ------------------------------------------------------------

if actual_value is not None:

    fig_history.add_trace(
        go.Scatter(
            x=[forecast_timestamp],
            y=[actual_value],
            mode="markers",
            name="Actual",
            marker=dict(
                size=12,
                symbol="circle"
            ),
            hovertemplate=(
                "%{x}<br>"
                "Actual: %{y:.0f} tickets"
                "<extra></extra>"
            )
        )
    )


# ------------------------------------------------------------
# Peak threshold
# ------------------------------------------------------------

fig_history.add_hline(
    y=peak_threshold,
    line_dash="dash",
    annotation_text=(
        f"Peak Threshold: {peak_threshold:.0f}"
    ),
    annotation_position="top left"
)


# ------------------------------------------------------------
# Average demand
# ------------------------------------------------------------

fig_history.add_hline(
    y=historical_average,
    line_dash="dot",
    annotation_text=(
        f"Historical Average: "
        f"{historical_average:.0f}"
    ),
    annotation_position="bottom left"
)


fig_history.update_layout(
    title=(
        f"Historical Demand and "
        f"{selected_horizon} Forecast"
    ),
    xaxis_title="Timestamp",
    yaxis_title="Ticket Demand",
    hovermode="x unified",
    height=500,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)


st.plotly_chart(
    fig_history,
    use_container_width=True
)


# ============================================================
# FORECAST VS ACTUAL SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Forecast Verification</div>',
    unsafe_allow_html=True
)


verify_col1, verify_col2, verify_col3 = st.columns(3)


with verify_col1:

    st.metric(
        "Predicted Demand",
        f"{prediction:.0f}"
    )


with verify_col2:

    if actual_value is not None:

        st.metric(
            "Actual Demand",
            f"{actual_value:.0f}"
        )

    else:

        st.metric(
            "Actual Demand",
            "Not Available"
        )


with verify_col3:

    if actual_value is not None:

        error = abs(
            prediction - actual_value
        )

        st.metric(
            "Absolute Error",
            f"{error:.0f} tickets"
        )

    else:

        st.metric(
            "Absolute Error",
            "N/A"
        )


# ============================================================
# FORECAST VISUALIZATION
# ============================================================

st.markdown(
    '<div class="section-title">🔮 Forecast Visualization</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Create forecast visualization with historical point,
# forecast point and target timestamp
# ------------------------------------------------------------

recent_data = data[
    data["Timestamp"] <= forecast_origin
].tail(12).copy()


forecast_plot = go.Figure()


forecast_plot.add_trace(
    go.Scatter(
        x=recent_data["Timestamp"],
        y=recent_data["Sales Count"],
        mode="lines+markers",
        name="Historical Demand",
        hovertemplate=(
            "%{x}<br>"
            "%{y:.0f} tickets"
            "<extra></extra>"
        )
    )
)


forecast_plot.add_trace(
    go.Scatter(
        x=[
            forecast_origin,
            forecast_timestamp
        ],
        y=[
            float(
                recent_data["Sales Count"].iloc[-1]
            ),
            prediction
        ],
        mode="lines+markers",
        name="Forecast Path",
        line=dict(
            dash="dash"
        ),
        hovertemplate=(
            "%{x}<br>"
            "%{y:.0f} tickets"
            "<extra></extra>"
        )
    )
)


forecast_plot.add_hline(
    y=peak_threshold,
    line_dash="dash",
    annotation_text="Peak Threshold"
)


forecast_plot.update_layout(
    title=(
        f"{selected_horizon} Short-Term Demand Forecast"
    ),
    xaxis_title="Time",
    yaxis_title="Tickets",
    height=450,
    hovermode="x unified"
)


st.plotly_chart(
    forecast_plot,
    use_container_width=True
)


# ============================================================
# FEATURE INFORMATION
# ============================================================

with st.expander(
    "🔧 View Model Input Features"
):

    st.write(
        "The Random Forest model receives the following "
        "time-series and calendar features:"
    )

    feature_display = feature_df.T.reset_index()

    feature_display.columns = [
        "Feature",
        "Value"
    ]

    st.dataframe(
        feature_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "🤖 View Model Information"
):

    st.write(
        "**Model Type:** Random Forest Regressor"
    )

    if hasattr(
        model,
        "n_estimators"
    ):

        st.write(
            f"**Number of Trees:** "
            f"{model.n_estimators}"
        )

    if hasattr(
        model,
        "feature_names_in_"
    ):

        st.write(
            "**Features used by saved model:**"
        )

        st.write(
            list(
                model.feature_names_in_
            )
        )


# ============================================================
# KPI RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">📋 Forecast Performance Results</div>',
    unsafe_allow_html=True
)


accuracy_results = load_result_file(
    "Forecast_Accuracy_Results.csv"
)

horizon_results = load_result_file(
    "Forecast_Horizon_Results.csv"
)

peak_results = load_result_file(
    "Peak_Miss_Rate_Results.csv"
)


result_tabs = st.tabs(
    [
        "Forecast Accuracy",
        "Forecast Horizon",
        "Peak Miss Rate"
    ]
)


# ------------------------------------------------------------
# Accuracy results
# ------------------------------------------------------------

with result_tabs[0]:

    if accuracy_results is not None:

        st.dataframe(
            accuracy_results,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Forecast_Accuracy_Results.csv was not found."
        )


# ------------------------------------------------------------
# Horizon results
# ------------------------------------------------------------

with result_tabs[1]:

    if horizon_results is not None:

        st.dataframe(
            horizon_results,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Forecast_Horizon_Results.csv was not found."
        )


# ------------------------------------------------------------
# Peak miss rate
# ------------------------------------------------------------

with result_tabs[2]:

    if peak_results is not None:

        st.dataframe(
            peak_results,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Peak_Miss_Rate_Results.csv was not found."
        )


# ============================================================
# OPERATIONAL RECOMMENDATION
# ============================================================

st.markdown(
    '<div class="section-title">💡 Operational Recommendation</div>',
    unsafe_allow_html=True
)


if status_type == "high":

    st.error(
        """
        **Recommended Action**

        • Prepare additional operational staff.

        • Monitor passenger queues closely.

        • Prepare crowd-management resources.

        • Review ferry service readiness.

        • Closely monitor subsequent forecast updates.
        """
    )

elif status_type == "moderate":

    st.warning(
        """
        **Recommended Action**

        • Maintain normal staffing levels.

        • Monitor demand and passenger queues.

        • Keep operational staff ready for a potential increase.

        • Review the next forecast update.
        """
    )

else:

    st.success(
        """
        **Recommended Action**

        • Normal staffing is likely sufficient.

        • Continue routine monitoring.

        • No immediate additional crowd-management
          preparation is indicated.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="small-note">

    Toronto Island Ferry Demand Forecasting | 
    Predictive Decision Support System

    <br><br>

    Forecasts are generated using historical ferry ticket
    demand and Random Forest machine-learning models.

    </div>
    """,
    unsafe_allow_html=True
)