import os
import glob
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BTCUSDT AI Prediction Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.metric-card {
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid #374151;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 10px;
}

.dashboard-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.dashboard-subtitle {
    color: #9ca3af;
    font-size: 17px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 20px;
}

.prediction-up {
    background: linear-gradient(135deg, #064e3b, #065f46);
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #10b981;
    text-align: center;
}

.prediction-down {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #ef4444;
    text-align: center;
}

.info-box {
    background-color: #1f2937;
    border-left: 4px solid #f59e0b;
    padding: 15px;
    border-radius: 8px;
}

.small-text {
    color: #9ca3af;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARTIFACT_DIR = os.path.join(BASE_DIR, "artifact")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output csv")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def find_files(directory, extensions):
    """Find files recursively."""
    files = []

    if not os.path.exists(directory):
        return files

    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(tuple(extensions)):
                files.append(os.path.join(root, filename))

    return files


def find_file_by_keywords(files, keywords):
    """Find first file containing one of the keywords."""
    for file in files:
        name = os.path.basename(file).lower()

        for keyword in keywords:
            if keyword.lower() in name:
                return file

    return None


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


# ============================================================
# FIND DATASET
# ============================================================

csv_files = find_files(DATASET_DIR, [".csv"])

dataset_path = find_file_by_keywords(
    csv_files,
    ["btc", "btcusdt", "ohlcv", "daily"]
)

if dataset_path is None and csv_files:
    dataset_path = csv_files[0]


# ============================================================
# LOAD DATASET
# ============================================================

df = None

if dataset_path:

    try:

        df = load_csv(dataset_path)

        # Normalize column names
        df.columns = [
            str(col).strip().lower().replace(" ", "_")
            for col in df.columns
        ]

        # Detect timestamp column
        timestamp_candidates = [
            "timestamp",
            "time",
            "date",
            "datetime",
            "open_time",
            "id"
        ]

        timestamp_col = None

        for col in timestamp_candidates:

            if col in df.columns:
                timestamp_col = col
                break

        if timestamp_col:

            # Try numeric timestamp first
            if pd.api.types.is_numeric_dtype(df[timestamp_col]):

                # Detect milliseconds vs seconds
                sample = df[timestamp_col].dropna()

                if len(sample) > 0:

                    median_value = sample.median()

                    if median_value > 1e11:
                        df["datetime"] = pd.to_datetime(
                            df[timestamp_col],
                            unit="ms",
                            errors="coerce"
                        )
                    else:
                        df["datetime"] = pd.to_datetime(
                            df[timestamp_col],
                            unit="s",
                            errors="coerce"
                        )

            else:

                df["datetime"] = pd.to_datetime(
                    df[timestamp_col],
                    errors="coerce"
                )

        # Sort chronologically
        if "datetime" in df.columns:
            df = df.sort_values("datetime").reset_index(drop=True)

    except Exception as e:

        st.error(f"Dataset loading error: {e}")

else:

    st.warning(
        "No CSV dataset was found inside the dataset folder."
    )


# ============================================================
# DETECT OHLCV COLUMNS
# ============================================================

if df is not None:

    def detect_column(possible_names):

        for name in possible_names:

            if name in df.columns:
                return name

        return None


    open_col = detect_column(["open"])
    high_col = detect_column(["high"])
    low_col = detect_column(["low"])
    close_col = detect_column(["close"])
    volume_col = detect_column([
        "volume",
        "volume_usdt",
        "quote_volume"
    ])


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    data = data.copy()

    if "close" not in data.columns:
        return data

    close = data["close"]

    data["return_1d"] = close.pct_change(1)

    data["return_3d"] = close.pct_change(3)

    data["return_7d"] = close.pct_change(7)

    data["return_14d"] = close.pct_change(14)

    data["return_30d"] = close.pct_change(30)

    # Lag prices
    for lag in [1, 2, 3, 5, 7, 14, 30]:

        data[f"close_lag_{lag}"] = close.shift(lag)

    # Moving averages
    for window in [5, 7, 10, 14, 20, 30, 50]:

        data[f"sma_{window}"] = close.rolling(window).mean()

    # EMA
    for window in [7, 14, 20, 30]:

        data[f"ema_{window}"] = close.ewm(
            span=window,
            adjust=False
        ).mean()

    # Volatility
    data["volatility_7"] = (
        data["return_1d"]
        .rolling(7)
        .std()
    )

    data["volatility_14"] = (
        data["return_1d"]
        .rolling(14)
        .std()
    )

    data["volatility_30"] = (
        data["return_1d"]
        .rolling(30)
        .std()
    )

    # Price range
    if "high" in data.columns and "low" in data.columns:

        data["price_range"] = (
            data["high"] - data["low"]
        )

        data["range_pct"] = (
            data["high"] - data["low"]
        ) / data["close"]

    # OHLC relationships
    if "open" in data.columns:

        data["open_close_change"] = (
            data["close"] - data["open"]
        ) / data["open"]

    if "high" in data.columns:

        data["close_high_ratio"] = (
            data["close"] / data["high"]
        )

    if "low" in data.columns:

        data["close_low_ratio"] = (
            data["close"] / data["low"]
        )

    # Volume features
    if "volume" in data.columns:

        data["volume_change"] = (
            data["volume"].pct_change()
        )

        data["volume_sma_7"] = (
            data["volume"]
            .rolling(7)
            .mean()
        )

        data["volume_sma_30"] = (
            data["volume"]
            .rolling(30)
            .mean()
        )

    # Date features
    if "datetime" in data.columns:

        data["day_of_week"] = (
            data["datetime"].dt.dayofweek
        )

        data["day_of_month"] = (
            data["datetime"].dt.day
        )

        data["month"] = (
            data["datetime"].dt.month
        )

    return data


# ============================================================
# FIND MODELS
# ============================================================

model_files = find_files(
    ARTIFACT_DIR,
    [".pkl", ".joblib"]
)


regression_model_path = find_file_by_keywords(
    model_files,
    [
        "regression",
        "reg_model",
        "regressor"
    ]
)


classification_model_path = find_file_by_keywords(
    model_files,
    [
        "classification",
        "cls_model",
        "classifier",
        "direction"
    ]
)


scaler_path = find_file_by_keywords(
    model_files,
    [
        "scaler",
        "standard"
    ]
)


feature_path = find_file_by_keywords(
    model_files,
    [
        "feature",
        "features"
    ]
)


# ============================================================
# LOAD MODELS
# ============================================================

regression_model = None
classification_model = None
scaler = None
feature_names = None

if regression_model_path:
    try:
        regression_model = load_model(regression_model_path)
        st.sidebar.success("✅ Regression Model Loaded")
    except Exception as e:
        st.sidebar.error("❌ Regression Model Loading Failed")
        st.sidebar.code(str(e))

if classification_model_path:
    classification_model = load_model(
        classification_model_path
    )


if scaler_path:
    scaler = load_model(scaler_path)


if feature_path:

    try:

        feature_names = joblib.load(feature_path)

        if isinstance(feature_names, dict):

            feature_names = (
                feature_names.get("features")
                or feature_names.get("feature_names")
            )

    except Exception:
        feature_names = None


# Try extracting feature names directly from model

if feature_names is None:

    if regression_model is not None:

        if hasattr(regression_model, "feature_names_in_"):

            feature_names = list(
                regression_model.feature_names_in_
            )


if feature_names is None:

    if classification_model is not None:

        if hasattr(classification_model, "feature_names_in_"):

            feature_names = list(
                classification_model.feature_names_in_
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ₿ BTCUSDT AI"
    )

    st.caption(
        "Machine Learning Prediction Dashboard"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "📈 Market Analysis",
            "🤖 AI Prediction",
            "🏆 Model Performance",
            "🔍 Feature Importance",
            "📁 Dataset Explorer"
        ]
    )

    st.divider()

    st.markdown("### System Status")

    if df is not None:

        st.success("Dataset Loaded")

    else:

        st.error("Dataset Missing")

    if regression_model is not None:

        st.success("Regression Model Loaded")

    else:

        st.warning("Regression Model Not Found")

    if classification_model is not None:

        st.success("Classification Model Loaded")

    else:

        st.warning("Classification Model Not Found")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">₿ BTCUSDT AI Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Machine Learning powered analysis and next-day Bitcoin market prediction'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# HANDLE MISSING DATASET
# ============================================================

if df is None:

    st.error(
        "Dataset could not be loaded. "
        "Please place your BTCUSDT CSV file inside the dataset folder."
    )

    st.stop()


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "📊 Overview":

    latest = df.iloc[-1]

    current_price = (
        latest[close_col]
        if close_col
        else 0
    )

    previous_price = (
        df.iloc[-2][close_col]
        if close_col and len(df) > 1
        else current_price
    )

    price_change = (
        (current_price - previous_price)
        / previous_price
        * 100
        if previous_price != 0
        else 0
    )

    # Metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Current Price",
            f"${current_price:,.2f}",
            f"{price_change:+.2f}%"
        )

    with col2:

        st.metric(
            "Dataset Records",
            f"{len(df):,}"
        )

    with col3:

        if volume_col:

            volume = latest[volume_col]

            st.metric(
                "Latest Volume",
                f"{volume:,.2f}"
            )

        else:

            st.metric(
                "Latest Volume",
                "N/A"
            )

    with col4:

        if "datetime" in df.columns:

            st.metric(
                "Latest Date",
                df["datetime"].iloc[-1].strftime(
                    "%Y-%m-%d"
                )
            )

        else:

            st.metric(
                "Latest Date",
                "N/A"
            )

    st.divider()

    # Price chart

    st.subheader("BTCUSDT Price History")

    chart_df = df.tail(300)

    if all(
        col is not None
        for col in [
            open_col,
            high_col,
            low_col,
            close_col
        ]
    ):

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=chart_df["datetime"],
                    open=chart_df[open_col],
                    high=chart_df[high_col],
                    low=chart_df[low_col],
                    close=chart_df[close_col],
                    name="BTCUSDT"
                )
            ]
        )

        fig.update_layout(
            height=550,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Quick analysis

    st.subheader("Market Snapshot")

    feature_df = create_features(df)

    c1, c2, c3 = st.columns(3)

    with c1:

        if "return_1d" in feature_df:

            ret = (
                feature_df["return_1d"].iloc[-1]
                * 100
            )

            st.metric(
                "Daily Return",
                f"{ret:+.2f}%"
            )

    with c2:

        if "volatility_7" in feature_df:

            vol = (
                feature_df["volatility_7"].iloc[-1]
                * 100
            )

            st.metric(
                "7D Volatility",
                f"{vol:.2f}%"
            )

    with c3:

        if "sma_30" in feature_df:

            sma = feature_df["sma_30"].iloc[-1]

            st.metric(
                "30D Moving Average",
                f"${sma:,.2f}"
            )


# ============================================================
# MARKET ANALYSIS
# ============================================================

elif page == "📈 Market Analysis":

    st.header("📈 Market Analysis")

    feature_df = create_features(df)

    tab1, tab2, tab3 = st.tabs(
        [
            "Returns",
            "Volume",
            "Volatility"
        ]
    )

    with tab1:

        fig = px.line(
            feature_df.tail(500),
            x="datetime",
            y="return_1d",
            title="Daily Returns"
        )

        fig.update_layout(
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with tab2:

        if volume_col:

            fig = px.line(
                df.tail(500),
                x="datetime",
                y=volume_col,
                title="Trading Volume"
            )

            fig.update_layout(
                template="plotly_dark",
                height=450
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with tab3:

        vol_df = feature_df.tail(500)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=vol_df["datetime"],
                y=vol_df["volatility_7"],
                name="7D Volatility"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=vol_df["datetime"],
                y=vol_df["volatility_30"],
                name="30D Volatility"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=450,
            title="Rolling Volatility"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# AI PREDICTION
# ============================================================

elif page == "🤖 AI Prediction":

    st.header("🤖 AI Market Prediction")

    if regression_model is None and classification_model is None:

        st.error(
            "No trained models were found inside the artifact folder."
        )

        st.info(
            "Place your .pkl/.joblib model files inside artifact."
        )

        st.stop()

    feature_df = create_features(df)

    latest_row = feature_df.iloc[-1:]

    st.subheader("Latest Market Data")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Current BTCUSDT",
            f"${df[close_col].iloc[-1]:,.2f}"
        )

    with c2:

        if "return_1d" in feature_df:

            st.metric(
                "Latest Return",
                f"{feature_df['return_1d'].iloc[-1] * 100:+.2f}%"
            )

    with c3:

        if "volatility_7" in feature_df:

            st.metric(
                "7D Volatility",
                f"{feature_df['volatility_7'].iloc[-1] * 100:.2f}%"
            )

    st.divider()

    # ========================================================
    # PREPARE FEATURES
    # ========================================================

    prediction_possible = True

    X_latest = None

    if feature_names is not None:

        missing_features = [
            feature
            for feature in feature_names
            if feature not in latest_row.columns
        ]

        if missing_features:

            prediction_possible = False

            st.warning(
                "Some model features are not available in the "
                "dashboard feature engineering pipeline."
            )

            with st.expander(
                "Missing Features"
            ):

                st.write(missing_features)

        else:

            X_latest = latest_row[
                feature_names
            ].copy()

    else:

        prediction_possible = False

        st.warning(
            "Feature names were not found in the model artifacts."
        )

    # ========================================================
    # REGRESSION
    # ========================================================

    if regression_model is not None and prediction_possible:

        st.subheader(
            "📈 Next-Day Return Prediction"
        )

        try:

            X_reg = X_latest.copy()

            if scaler is not None:

                try:

                    X_reg = scaler.transform(
                        X_reg
                    )

                except Exception:

                    pass

            predicted_return = (
                regression_model.predict(
                    X_reg
                )[0]
            )

            current_price = df[
                close_col
            ].iloc[-1]

            predicted_price = (
                current_price
                * (1 + predicted_return)
            )

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Predicted Next-Day Return",
                    f"{predicted_return * 100:+.3f}%"
                )

            with c2:

                st.metric(
                    "Estimated Next-Day Price",
                    f"${predicted_price:,.2f}"
                )

            st.progress(
                min(
                    max(
                        float(
                            abs(predicted_return)
                            * 100
                        ),
                        0
                    ),
                    1
                )
            )

        except Exception as e:

            st.error(
                f"Regression prediction error: {e}"
            )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if classification_model is not None and prediction_possible:

        st.subheader(
            "🎯 Next-Day Market Direction"
        )

        try:

            X_cls = X_latest.copy()

            if scaler is not None:

                try:

                    X_cls = scaler.transform(
                        X_cls
                    )

                except Exception:

                    pass

            direction = (
                classification_model.predict(
                    X_cls
                )[0]
            )

            probability = None

            if hasattr(
                classification_model,
                "predict_proba"
            ):

                probability = (
                    classification_model
                    .predict_proba(X_cls)[0]
                )

                if len(probability) > 1:

                    probability_up = (
                        probability[1] * 100
                    )

                else:

                    probability_up = (
                        probability[0] * 100
                    )

            else:

                probability_up = None

            if int(direction) == 1:

                st.markdown(
                    f"""
                    <div class="prediction-up">

                    <h2>🟢 BULLISH / UP</h2>

                    <h3>Expected Direction: UP</h3>

                    <p>
                    Probability of upward movement:
                    <strong>
                    {probability_up:.2f}%
                    </strong>
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                down_probability = (
                    100 - probability_up
                    if probability_up is not None
                    else 0
                )

                st.markdown(
                    f"""
                    <div class="prediction-down">

                    <h2>🔴 BEARISH / DOWN</h2>

                    <h3>Expected Direction: DOWN</h3>

                    <p>
                    Probability of downward movement:
                    <strong>
                    {down_probability:.2f}%
                    </strong>
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except Exception as e:

            st.error(
                f"Classification prediction error: {e}"
            )

    st.divider()

    st.markdown(
        """
        <div class="info-box">

        ⚠️ <strong>Disclaimer:</strong>
        This prediction is generated using historical BTCUSDT
        market data and machine learning models. It is intended
        for educational and research purposes only and should
        not be considered financial advice.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🏆 Model Performance":

    st.header("🏆 Model Performance")

    # Search output CSV files

    output_files = find_files(
        OUTPUT_DIR,
        [".csv"]
    )

    if output_files:

        st.subheader(
            "Saved Model Evaluation Results"
        )

        for file in output_files:

            try:

                result_df = pd.read_csv(file)

                st.markdown(
                    f"### 📄 {os.path.basename(file)}"
                )

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

            except Exception:
                pass

    else:

        st.info(
            "No evaluation CSV files found in output csv folder."
        )

    st.divider()

    st.subheader(
        "Loaded Models"
    )

    model_status = pd.DataFrame(
        {
            "Model Type": [
                "Regression",
                "Classification",
                "Scaler"
            ],
            "Status": [
                "Loaded"
                if regression_model is not None
                else "Not Found",

                "Loaded"
                if classification_model is not None
                else "Not Found",

                "Loaded"
                if scaler is not None
                else "Not Found"
            ]
        }
    )

    st.dataframe(
        model_status,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "🔍 Feature Importance":

    st.header("🔍 Feature Importance")

    model = regression_model

    if model is None:

        st.warning(
            "Regression model not found."
        )

    else:

        importance = None

        # Tree based models

        if hasattr(
            model,
            "feature_importances_"
        ):

            importance = (
                model.feature_importances_
            )

        # Linear models

        elif hasattr(
            model,
            "coef_"
        ):

            importance = np.abs(
                np.ravel(
                    model.coef_
                )
            )

        if importance is not None:

            if feature_names is not None:

                names = feature_names

            else:

                names = [
                    f"Feature {i+1}"
                    for i in range(
                        len(importance)
                    )
                ]

            min_length = min(
                len(names),
                len(importance)
            )

            importance_df = pd.DataFrame(
                {
                    "Feature": names[:min_length],
                    "Importance": importance[:min_length]
                }
            )

            importance_df = (
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(20)
            )

            fig = px.bar(
                importance_df.sort_values(
                    "Importance"
                ),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 20 Important Features"
            )

            fig.update_layout(
                template="plotly_dark",
                height=600
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                importance_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Feature importance is not directly available "
                "for this model."
            )


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "📁 Dataset Explorer":

    st.header("📁 Dataset Explorer")

    st.write(
        f"Dataset: **{os.path.basename(dataset_path)}**"
    )

    # Dataset statistics

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

    with c2:

        st.metric(
            "Columns",
            f"{df.shape[1]:,}"
        )

    with c3:

        st.metric(
            "Missing Values",
            f"{df.isna().sum().sum():,}"
        )

    with c4:

        st.metric(
            "Duplicates",
            f"{df.duplicated().sum():,}"
        )

    st.divider()

    # Search

    search = st.text_input(
        "🔎 Search dataset columns"
    )

    if search:

        matching_columns = [
            col
            for col in df.columns
            if search.lower() in col.lower()
        ]

        display_df = df[
            matching_columns
        ]

    else:

        display_df = df

    st.dataframe(
        display_df.tail(100),
        use_container_width=True
    )

    st.subheader(
        "Dataset Information"
    )

    info_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "Missing Values": [
                int(df[col].isna().sum())
                for col in df.columns
            ],
            "Unique Values": [
                int(df[col].nunique())
                for col in df.columns
            ]
        }
    )

    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#6b7280;padding:15px;">

    <strong>₿ BTCUSDT AI Prediction System</strong><br>

    Machine Learning • Time-Series Analysis • Predictive Analytics

    <br><br>

    <span class="small-text">
    Educational & Research Project — Not Financial Advice
    </span>

    </div>
    """,
    unsafe_allow_html=True
)