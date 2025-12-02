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


st.markdown("**UI status:** Loaded — enter data or upload CSV")
st.title("⚡ Electricity Load Forecasting (Demo)")

if "rows" not in st.session_state:
    # keep separate buffers for full-feature input and weather-only input
    st.session_state.rows_full = [[0.0] * len(FEATURES) for _ in range(LOOK_BACK)]
    st.session_state.rows_weather = [[0.0] * len(FEATURES_NO_TARGET) for _ in range(LOOK_BACK)]

col_left, col_right = st.columns([2, 1])

with col_left:
    st.header("Manual input (24 rows)")
    input_mode = st.radio("Input mode:", ["Weather-only (no ERCOT)", "All features (include ERCOT past)"])

    if input_mode.startswith("Weather"):
        dfw = pd.DataFrame(st.session_state.rows_weather, columns=FEATURES_NO_TARGET)
        edited_w = st.data_editor(dfw, num_rows="dynamic", use_container_width=True)
        st.session_state.rows_weather = edited_w.values.tolist()
    else:
        dff = pd.DataFrame(st.session_state.rows_full, columns=FEATURES)
        edited_f = st.data_editor(dff, num_rows="dynamic", use_container_width=True)
        st.session_state.rows_full = edited_f.values.tolist()

    st.markdown("---")
    if st.button("Predict next hour"):
        model, scaler = load_model()
        if model is None or scaler is None:
            st.error("Model or scaler not available. Install TensorFlow and ensure files exist in Save Model/.")
        else:
            target_idx = FEATURES.index(TARGET_FEATURE)
            # prepare full-shaped array (LOOK_BACK, n_features)
            if input_mode.startswith("Weather"):
                # user provided only weather features; build full array with target column zeros
                arr = np.zeros((LOOK_BACK, len(FEATURES)), dtype=float)
                weather_vals = np.array(st.session_state.rows_weather)
                if weather_vals.shape != (LOOK_BACK, len(FEATURES_NO_TARGET)):
                    st.error(f"Weather input must be {LOOK_BACK} rows × {len(FEATURES_NO_TARGET)} columns")
                    st.stop()
                # place weather columns in the correct order
                for j, feat in enumerate(FEATURES_NO_TARGET):
                    col_idx = FEATURES.index(feat)
                    arr[:, col_idx] = weather_vals[:, j]
                # target column remains zeros (or could be last-known if available)
            else:
                arr = np.array(st.session_state.rows_full)
                if arr.shape != (LOOK_BACK, len(FEATURES)):
                    st.error(f"Full input must be {LOOK_BACK} rows × {len(FEATURES)} columns")
                    st.stop()

            # scale inputs using the saved scaler (scaler expects 2D: n_samples x n_features)
            try:
                arr_scaled = scaler.transform(arr)
            except Exception as e:
                st.error("Failed to scale input data with saved scaler.")
                st.exception(e)
                st.stop()

            x = arr_scaled.reshape(1, LOOK_BACK, len(FEATURES))
            try:
                y = model.predict(x)
            except Exception as e:
                st.error("Model prediction failed. Check input shapes and model compatibility.")
                st.exception(e)
                st.stop()

            dummy = np.zeros((1, len(FEATURES)))
            dummy[0, target_idx] = y[0, 0]
            val = scaler.inverse_transform(dummy)[0, target_idx]
            st.success(f"Predicted {TARGET_FEATURE} load: {val:.2f}")

with col_right:
    st.header("Upload CSV (optional)")
    uploaded = st.file_uploader("CSV with columns: " + ", ".join(FEATURES), type=["csv"])
    if uploaded is not None:
        try:
            dfu = pd.read_csv(uploaded)
        except Exception as e:
            st.error("Cannot read CSV file")
            st.exception(e)
            st.stop()

        # allow CSVs with either full features or only weather features
        present = [c for c in FEATURES if c in dfu.columns]
        if len(present) == len(FEATURES):
            st.success(f"Loaded CSV with {len(dfu)} rows (all features)")
            st.dataframe(dfu.head())
            if st.button("Use last 24 rows (full)"):
                st.session_state.rows_full = dfu[FEATURES].values.tolist()[-LOOK_BACK:]
                st.experimental_rerun()

            if st.button("Run batch forecast (full)"):
                model, scaler = load_model()
                if model is None or scaler is None:
                    st.error("Model/scaler not available for batch forecasting.")
                else:
                    X = dfu[FEATURES].values
                    preds = []
                    target_idx = FEATURES.index(TARGET_FEATURE)
                    for i in range(LOOK_BACK, len(X) + 1):
                        win = X[i-LOOK_BACK:i]
                        try:
                            win_scaled = scaler.transform(win)
                        except Exception as e:
                            st.error("Failed to scale window for batch forecasting")
                            st.exception(e)
                            st.stop()
                        y = model.predict(win_scaled.reshape(1, LOOK_BACK, len(FEATURES)))
                        d = np.zeros((1, len(FEATURES)))
                        d[0, target_idx] = y[0, 0]
                        inv = scaler.inverse_transform(d)[0, target_idx]
                        preds.append(inv)
                    res = dfu.iloc[LOOK_BACK - 1:].copy()
                    res["forecast"] = preds
                    st.dataframe(res.head())

        elif all(c in dfu.columns for c in FEATURES_NO_TARGET):
            st.success(f"Loaded CSV with {len(dfu)} rows (weather-only)")
            st.dataframe(dfu[FEATURES_NO_TARGET].head())
            if st.button("Use last 24 rows (weather)"):
                st.session_state.rows_weather = dfu[FEATURES_NO_TARGET].values.tolist()[-LOOK_BACK:]
                st.experimental_rerun()

            if st.button("Run batch forecast (weather-only)"):
                model, scaler = load_model()
                if model is None or scaler is None:
                    st.error("Model/scaler not available for batch forecasting.")
                else:
                    Xw = dfu[FEATURES_NO_TARGET].values
                    preds = []
                    target_idx = FEATURES.index(TARGET_FEATURE)
                    for i in range(LOOK_BACK, len(Xw) + 1):
                        win_w = Xw[i-LOOK_BACK:i]
                        # build full window with zeros at target position
                        win_full = np.zeros((LOOK_BACK, len(FEATURES)))
                        for j, feat in enumerate(FEATURES_NO_TARGET):
                            win_full[:, FEATURES.index(feat)] = win_w[:, j]
                        try:
                            win_scaled = scaler.transform(win_full)
                        except Exception as e:
                            st.error("Failed to scale window for batch forecasting")
                            st.exception(e)
                            st.stop()
                        y = model.predict(win_scaled.reshape(1, LOOK_BACK, len(FEATURES)))
                        d = np.zeros((1, len(FEATURES)))
                        d[0, target_idx] = y[0, 0]
                        inv = scaler.inverse_transform(d)[0, target_idx]
                        preds.append(inv)
                    res = dfu.iloc[LOOK_BACK - 1:].copy()
                    res["forecast"] = preds
                    st.dataframe(res.head())

        else:
            missing = [c for c in FEATURES_NO_TARGET if c not in dfu.columns]
            st.error(f"CSV missing required weather columns: {missing}")

st.markdown("---")
st.markdown("Notes: this is a demo UI. For production, serve the model behind an API and secure access.")

