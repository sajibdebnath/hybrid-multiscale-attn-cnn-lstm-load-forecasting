import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load

# Lightweight Streamlit UI with lazy model loading to avoid blocking page render.
st.set_page_config(page_title="Electricity Load Forecast", layout="wide")

FEATURES = ['ERCOT', 'tmpc', 'relh', 'sped', 'feel', 'p01m', 'Max-T', 'Min-T', 'M-label', 'W-label', 'H-label']
TARGET_FEATURE = 'ERCOT'
LOOK_BACK = 24

# features excluding the prediction target (weather / exogenous inputs)
FEATURES_NO_TARGET = [f for f in FEATURES if f != TARGET_FEATURE]

MODEL_PATH = Path("Save Model/model.h5")
SCALER_PATH = Path("Save Model/scaler.pkl")

def load_model():
    """Load model and scaler when needed. Return (model, scaler) or (None, None) on failure."""
    try:
        import tensorflow as tf
    except Exception:
        return None, None

    try:
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        scaler = load(str(SCALER_PATH))
        return model, scaler
    except Exception:
        return None, None

def run_prediction():
    """Run prediction using values in `st.session_state.rows_weather`.
    Shows Streamlit messages and returns the predicted value or None on error.
    """
    model, scaler = load_model()
    if model is None or scaler is None:
        st.error("Model or scaler not available. Install TensorFlow and ensure files exist in Save Model/.")
        return None

    target_idx = FEATURES.index(TARGET_FEATURE)
    arr = np.zeros((LOOK_BACK, len(FEATURES)), dtype=float)
    weather_vals = np.array(st.session_state.rows_weather)
    if weather_vals.shape != (LOOK_BACK, len(FEATURES_NO_TARGET)):
        st.error(f"Weather input must be {LOOK_BACK} rows × {len(FEATURES_NO_TARGET)} columns")
        return None

    for j, feat in enumerate(FEATURES_NO_TARGET):
        arr[:, FEATURES.index(feat)] = weather_vals[:, j]

    try:
        arr_df = pd.DataFrame(arr, columns=FEATURES)
        if hasattr(scaler, "feature_names_in_"):
            arr_df = arr_df.loc[:, list(scaler.feature_names_in_)]
        arr_scaled = scaler.transform(arr_df)
    except Exception as e:
        st.error("Failed to scale input data with saved scaler.")
        st.exception(e)
        return None

    x = arr_scaled.reshape(1, LOOK_BACK, len(FEATURES))
    try:
        y = model.predict(x)
    except Exception as e:
        st.error("Model prediction failed. Check input shapes and model compatibility.")
        st.exception(e)
        return None

    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, target_idx] = y[0, 0]
    val = scaler.inverse_transform(dummy)[0, target_idx]
    st.success(f"Predicted {TARGET_FEATURE} load: {val:.2f}")
    return val
    


st.markdown("**UI status:** Loaded — enter data or upload CSV")
st.title("⚡ Electricity Load Forecasting")

if "rows_weather" not in st.session_state:
    # buffer for weather-only input (user provides only FEATURES_NO_TARGET)
    # initialize with empty strings so the editor shows blank cells by default
    st.session_state.rows_weather = [[""] * len(FEATURES_NO_TARGET) for _ in range(LOOK_BACK)]

col_left, col_right = st.columns([1, 3])

with col_left:
    st.header("Upload CSV (optional)")
    uploaded = st.file_uploader("CSV with weather columns: " + ", ".join(FEATURES_NO_TARGET), type=["csv"])
    if uploaded is not None:
        try:
            dfu = pd.read_csv(uploaded)
        except Exception as e:
            st.error("Cannot read CSV file")
            st.exception(e)
        else:
            # expect CSV to contain only weather columns (FEATURES_NO_TARGET)
            if all(c in dfu.columns for c in FEATURES_NO_TARGET):
                st.success(f"Loaded CSV with {len(dfu)} rows (weather-only)")
                st.dataframe(dfu[FEATURES_NO_TARGET].head())
                if st.button("Use last 24 rows from CSV"):
                    last_rows = dfu[FEATURES_NO_TARGET].values.tolist()[-LOOK_BACK:]
                    if len(last_rows) == LOOK_BACK:
                        st.session_state.rows_weather = last_rows
                        st.info("Loaded last 24 rows into the editor. Press 'Predict next hour' to compute.")
                    else:
                        st.error(f"CSV must contain at least {LOOK_BACK} rows to load last 24 rows.")
            else:
                missing = [c for c in FEATURES_NO_TARGET if c not in dfu.columns]
                st.error(f"CSV missing required weather columns: {missing}")
        # left column intentionally compact; prediction moved to right column

with col_right:
    st.header("Manual input (24 rows)")
    # show only the weather/exogenous columns for user input
    # present exactly LOOK_BACK rows (blank by default) so user can fill values
    dfw = pd.DataFrame(st.session_state.rows_weather, columns=FEATURES_NO_TARGET)
    edited_w = st.data_editor(dfw, num_rows=LOOK_BACK, use_container_width=True)
    # convert empty strings or non-numeric to 0 before saving
    edited_w = edited_w.replace("", 0)
    edited_w = edited_w.apply(pd.to_numeric, errors="coerce").fillna(0)
    st.session_state.rows_weather = edited_w.values.tolist()

    st.markdown("---")
    if st.button("Predict", key="predict_right"):
        run_prediction()

st.markdown("---")
st.markdown("Notes: this is a demo UI. For production, serve the model behind an API and secure access.")
