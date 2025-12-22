import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load

st.set_page_config(page_title="Electricity Load Forecast", layout="wide")

FEATURES = ['ERCOT', 'tmpc', 'relh', 'sped', 'feel', 'p01m',
            'Max-T', 'Min-T', 'M-label', 'W-label', 'H-label']
LOOK_BACK = 24
WEATHER_COLS = [f for f in FEATURES if f != 'ERCOT']
ACTUAL_COL = 'Actual ERCOT'
MODEL_PATH = Path("Save Model/model.h5")
SCALER_PATH = Path("Save Model/scaler.pkl")


def load_model():
    try:
        import tensorflow as tf
    except Exception as e:
        st.error("TensorFlow is not installed.")
        st.exception(e)
        return None, None
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        scaler = load(str(SCALER_PATH))
        return model, scaler
    except Exception as e:
        st.error("Failed to load model/scaler.")
        st.exception(e)
        return None, None


def predict_once(df_weather, model, scaler):
    arr = np.zeros((LOOK_BACK, len(FEATURES)))
    for j, feat in enumerate(WEATHER_COLS):
        arr[:, FEATURES.index(feat)] = df_weather[feat].astype(float).values
    arr_df = pd.DataFrame(arr, columns=FEATURES)

    if hasattr(scaler, "feature_names_in_"):
        arr_df = arr_df.loc[:, list(scaler.feature_names_in_)]

    arr_scaled = scaler.transform(arr_df)
    x = arr_scaled.reshape(1, LOOK_BACK, arr_scaled.shape[1])
    y = model.predict(x)

    if hasattr(scaler, "feature_names_in_"):
        names = list(scaler.feature_names_in_)
        pos = names.index("ERCOT")
        dummy = np.zeros((1, len(names)))
        dummy[0, pos] = y[0, 0]
        inv = scaler.inverse_transform(dummy)
        return float(inv[0, pos])
    else:
        dummy = np.zeros((1, len(FEATURES)))
        dummy[0, FEATURES.index("ERCOT")] = y[0, 0]
        inv = scaler.inverse_transform(dummy)
        return float(inv[0, FEATURES.index("ERCOT")])


# --- UI: ONLY uploader + Predict ----
st.title("⚡ Electricity Load Forecast (minimal)")

uploaded = st.file_uploader("Upload CSV file (must include weather columns)", type=["csv"], key="uploader_minimal")

# Show Predict button regardless so UI is minimal
if st.button("Predict"):
    if uploaded is None:
        st.error("Please upload a CSV first.")
    else:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error("Unable to read CSV.")
            st.exception(e)
            st.stop()

        missing = [c for c in WEATHER_COLS if c not in df.columns]
        if missing:
            st.error(f"CSV missing required columns: {missing}")
            st.stop()
        if len(df) < LOOK_BACK:
            st.error(f"CSV must contain at least {LOOK_BACK} rows.")
            st.stop()

        tail = df.tail(LOOK_BACK).reset_index(drop=True)
        df_output = pd.DataFrame({c: tail[c].values for c in WEATHER_COLS})
        df_output[ACTUAL_COL] = tail["ERCOT"].values if "ERCOT" in tail.columns else ""

        model, scaler = load_model()
        if model is None or scaler is None:
            st.stop()

        try:
            df_weather = df_output[WEATHER_COLS].replace("", np.nan).apply(pd.to_numeric, errors="coerce").fillna(0)
            pred = predict_once(df_weather, model, scaler)
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)
            st.stop()

        df_output["Predicted ERCOT"] = f"{pred:.2f}"
        ordered = WEATHER_COLS + [ACTUAL_COL, "Predicted ERCOT"]
        df_output = df_output[ordered]

        st.subheader("Prediction Output (24 rows)")
        st.dataframe(df_output, use_container_width=True)
