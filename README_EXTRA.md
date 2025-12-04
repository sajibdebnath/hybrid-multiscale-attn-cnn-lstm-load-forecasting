# Detailed Evaluation and How to Run the Demo

This file contains detailed numeric evaluation tables, model performance information, and quick run instructions for the Streamlit demo UI.

## Detailed Evaluation

### Evaluation Metrics

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | sMAPE (%) | PSNR (dB) | R² |
|-------|----------:|----------:|---------:|----------:|----------:|----:|
| LSTM | 2393.4662 | 3298.2063 | 3.8416 | 3.9473 | 28.2431 | 0.9042 |
| CNN-LSTM | 1984.1900 | 2654.4977 | 3.5111 | 3.4465 | 30.1290 | 0.9380 |
| Attention-LSTM | 2193.6967 | 2878.8981 | 3.9219 | 3.9617 | 29.4241 | 0.9270 |
| **Attention-based CNN-LSTM (Proposed / Hybrid)** | **1430.5544** | **1915.1668** | **2.5278** | **2.5063** | **32.9645** | **0.9677** |

### Model Size & Performance

| Model | Params (M) | Train Time (s/epoch) | Inference Time (ms) | Peak Mem (MB, GPU) |
|-------:|----------:|--------------------:|--------------------:|-------------------:|
| LSTM | 0.0003 | 2.132 | 71.19 | 84.7 |
| CNN-LSTM | 0.0602 | 5.034 | 66.26 | 148.9 |
| Attention-LSTM | 0.0088 | 6.995 | 80.74 | 172.2 |
| Proposed Hybrid (Attention-based CNN-LSTM) | 0.3614 | 6.327 | 66.60 | 172.8 |

## How to run the demo UI (Streamlit)

1. Ensure the trained model and scaler are present at:

   - `Save Model/model.h5`
   - `Save Model/scaler.pkl`

2. Install requirements (use your environment manager):

```powershell
pip install -r requirements.txt
# or a minimal set:
pip install streamlit tensorflow pandas numpy scikit-learn joblib
```

3. Run the local demo UI:

```powershell
streamlit run app.py
```

Open the URL shown by Streamlit (usually `http://localhost:8501`). The UI accepts a weather-only CSV (columns: `tmpc, relh, sped, feel, p01m, Max-T, Min-T, M-label, W-label, H-label`) or manual entry (24 rows). Click **Predict next hour** (right column) to get the forecast.

## Notes & Compatibility

- The scaler file (`scaler.pkl`) was saved with a particular scikit-learn version; you may see an `InconsistentVersionWarning` if your installed `scikit-learn` differs. To avoid subtle scaling differences re-save the scaler with your environment's scikit-learn or install the original version (for example `pip install scikit-learn==1.6.1`).
- Streamlit shows a `FutureWarning` about `DataFrame.replace` downcasting in some pandas versions; this does not break the app. If you prefer a warning-free run, update the small replacement call in `app.py` to use `infer_objects(copy=False)` after replacement.

If you'd like, I can merge this content into `README.md` directly (in-place), or keep it as a separate supplemental file and add a link from `README.md`.
