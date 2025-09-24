# Figure 18 collage for ERCOT demand (Actual vs Forecast + Scatter)
# ----------------------------------------------------------------------
# Requirements: numpy, matplotlib, pillow
# pip install numpy matplotlib pillow

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont


def make_figure13_like(
    ercot_actual,
    preds_dict,
    model_order=None,
    outdir=".",
    basename="Load Forecasting results and scatter plots",
    xlabel_time="Time (day)",
    ylabel_time="Demand (MW)",
    xlabel_scatter="Observed (MW)",
    ylabel_scatter="Predicted (MW)",
    dpi=300,                       # default high-res save
    # --- FONT CONTROLS ---
    panel_label_fontsize=24,       # (a,b,c,...) size
    axis_label_fontsize=14,        # x/y labels
    tick_label_fontsize=12,        # tick labels
    legend_fontsize=12,            # legend text
    annotation_fontsize=11,        # small texts inside axes
    collage_title_fontsize=26,     # stitched-image title
    # --- PANEL LABEL POSITIONS ---
    panel_label_xy_ts=(0.02, 0.02),  # time-series tag position (left-bottom)
    panel_label_xy_sc=(0.98, 0.50),  # scatter tag position (right-center)
):
    """
    Make a collage like Figure 17 with alternating time-series and scatter panels.

    ercot_actual : 1D array-like
        Actual ERCOT demand for the test window.
    preds_dict : dict[str, 1D array-like]
        Keys are model names; values are predictions aligned to ercot_actual.
        Example:
            {
              "LSTM": yhat_lstm,
              "CNN-LSTM": yhat_cnnlstm,
              "A-LSTM": yhat_alstm,
              "A-CNN-LSTM": yhat_acnnlstm
            }
    model_order : list[str] | None
        Order of rows; defaults to keys of preds_dict in insertion order.
    outdir : str
        Folder to save panels and the final collage PNG.
    basename : str
        Base filename for outputs (final image: f"{basename}.png").
    dpi : int
        DPI for saved figures.
    """

    os.makedirs(outdir, exist_ok=True)
    y_true = np.asarray(ercot_actual).astype(float).reshape(-1)
    T = y_true.shape[0]
    t = np.arange(1, T + 1)

    # order of rows
    if model_order is None:
        model_order = list(preds_dict.keys())
    assert len(model_order) >= 1

    # ---- helpers ----
    def _save(fig, path):
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def _align(a, b):
        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        assert a.shape == b.shape, "Prediction and actual must have the same length."
        m = np.isfinite(a) & np.isfinite(b)
        return a[m], b[m]

    def time_series_panel(y_obs, y_pred, model_name, tag, path):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot(t[: len(y_obs)], y_obs, label="actual")
        ax.plot(t[: len(y_pred)], y_pred, label=model_name)

        ax.set_xlabel(xlabel_time, fontsize=axis_label_fontsize)
        ax.set_ylabel(ylabel_time, fontsize=axis_label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)

        ymin, ymax = min(y_obs.min(), y_pred.min()), max(y_obs.max(), y_pred.max())
        yr = max(1.0, ymax - ymin)
        ax.set_ylim(ymin - 0.05 * yr, ymax + 0.05 * yr)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

        ax.legend(loc="upper left", frameon=False, fontsize=legend_fontsize)

        # Panel letter (a,c,e,...) — left-bottom by default
        ax.text(
            panel_label_xy_ts[0], panel_label_xy_ts[1], f"({tag})",
            transform=ax.transAxes,
            fontsize=panel_label_fontsize,
            fontweight="bold",
            va="bottom", ha="left"
        )

        _save(fig, path)

    def scatter_panel(y_obs, y_pred, model_name, tag, path):
        y_obs, y_pred = _align(y_obs, y_pred)

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        # Scatter
        ax.scatter(y_obs, y_pred, s=18, alpha=0.9)

        # 1:1 line
        x_min, x_max = float(np.min(y_obs)), float(np.max(y_obs))
        line_x = np.linspace(x_min, x_max, 200)
        ax.plot(line_x, line_x, linestyle="--", linewidth=1)

        # Regression fit (y_pred = a*x + b)
        a, b = np.polyfit(y_obs, y_pred, 1)
        ax.plot(line_x, a * line_x + b, linewidth=1.5)

        # Central observed band (10th–90th percentile)
        x1, x2 = np.percentile(y_obs, 10), np.percentile(y_obs, 90)
        ax.axvspan(x1, x2, alpha=0.12)

        ax.set_xlabel(xlabel_scatter, fontsize=axis_label_fontsize)
        ax.set_ylabel(ylabel_scatter, fontsize=axis_label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)

        ax.text(
            0.02, 0.96,
            f"Intercept = {b:,.0f}, Slope = {a:0.4f}\nModel = {model_name}",
            transform=ax.transAxes,
            va="top",
            fontsize=annotation_fontsize
        )

        # Panel letter (b,d,f,...) — RIGHT side by default
        ax.text(
            panel_label_xy_sc[0], panel_label_xy_sc[1], f"({tag})",
            transform=ax.transAxes,
            fontsize=panel_label_fontsize,
            fontweight="bold",
            va="center", ha="right"
        )

        _save(fig, path)

    # ---- generate panels (one figure per panel) ----
    panel_paths = []
    tags = list("abcdefghijklmnopqrstuvwxyz")

    for i, name in enumerate(model_order):
        y_pred = np.asarray(preds_dict[name]).astype(float).reshape(-1)
        assert y_pred.shape[0] == y_true.shape[0], f"Length mismatch for {name}"

        # (a,c,e,...) time series
        tag_ts = tags[2 * i]
        p_ts = os.path.join(outdir, f"{basename}_{i+1}_timeseries.png")
        time_series_panel(y_true, y_pred, name, tag_ts, p_ts)
        panel_paths.append(p_ts)

        # (b,d,f,...) scatter
        tag_sc = tags[2 * i + 1]
        p_sc = os.path.join(outdir, f"{basename}_{i+1}_scatter.png")
        scatter_panel(y_true, y_pred, name, tag_sc, p_sc)
        panel_paths.append(p_sc)

    # ---- stitch panels into a collage ----
    imgs = [Image.open(p).convert("RGB") for p in panel_paths]
    rows, cols = len(model_order), 2
    w, h = imgs[0].size
    pad_x, pad_y, margin = 24, 28, 28
    canvas_w = cols * w + (cols - 1) * pad_x + 2 * margin
    canvas_h = rows * h + (rows - 1) * pad_y + 2 * margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            x = margin + c * (w + pad_x)
            y = margin + r * (h + pad_y)
            canvas.paste(imgs[idx], (x, y))

    # Collage title with adjustable font size (tries DejaVu Sans Bold)
    draw = ImageDraw.Draw(canvas)
    try:
        font_path = findfont(FontProperties(family="DejaVu Sans", weight="bold"))
        title_font = ImageFont.truetype(font_path, collage_title_fontsize)
    except Exception:
        title_font = ImageFont.load_default()

    draw.text(
        (margin, 6),
        "ERCOT demand - Actual vs Forecast and Scatter (four models)",
        fill=(0, 0, 0),
        font=title_font
    )

    final_path = os.path.join(outdir, f"{basename}.png")
    canvas.save(final_path)
    return final_path


# -------------------------
# Example usage (replace with your arrays)
# -------------------------
if __name__ == "__main__":
    # Demo data (replace with your real arrays)
    np.random.seed(7)
    T = 110
    t = np.arange(1, T + 1)
    base = 330_000 + 25_000*np.sin(2*np.pi*t/28) + 12_000*np.sin(2*np.pi*t/7)
    noise = np.random.normal(0, 6000, size=T)
    ercot_actual = (base + noise).astype(float)

    preds = {
        "LSTM": ercot_actual*0.97 + 9500 + np.random.normal(0, 6000, T),
        "CNN-LSTM": ercot_actual*0.995 + 1800 + np.random.normal(0, 3800, T),
        "A-LSTM": ercot_actual*0.992 + 1000 + np.random.normal(0, 4200, T),
        "A-CNN-LSTM": ercot_actual*0.997 + 800 + np.random.normal(0, 3200, T),
    }

    out = make_figure13_like(
        ercot_actual=ercot_actual,
        preds_dict=preds,
        model_order=["LSTM", "CNN-LSTM", "A-LSTM", "A-CNN-LSTM"],
        outdir="plots",
        basename="Load Forecasting results and scatter plots",
        dpi=400,                      # high-res save
        panel_label_fontsize=28,      # bigger (a,b,c,…)
        axis_label_fontsize=16,
        tick_label_fontsize=13,
        legend_fontsize=13,
        annotation_fontsize=12,
        collage_title_fontsize=28,
        # Right-side scatter labels (default): (0.98, 0.50)
        panel_label_xy_sc=(0.98, 0.08)
    )
    print("Saved:", out)
