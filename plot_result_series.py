# =============================================================================
# File      : plot_forecast_results.py
# Author    : Felipe Munaro Lima
# Version   : 1.0.0
# Date      : 2026-06-14
#
# Descriptions
# -----------
# This script loads wind power forecasting results and generates comparison
# plots between measured and predicted values for one or more forecasting
# models.
#
# For each prediction file, an empirical 90% prediction interval is computed
# from the residual distribution:
#
#     e = y_true - y_pred
#
# The 5th and 95th percentiles of the residuals are estimated and used to
# construct uncertainty bounds around the prediction:
#
#     lower = y_pred + Q05(e)
#     upper = y_pred + Q95(e)
#
# The resulting shaded region represents an empirical prediction interval
# containing approximately 90% of the observed prediction errors.
#
# Inputs
# ------
# dates.txt / dates_cv.txt
# y_test.txt / y_test_cv.txt
# y_pred*.txt
#
# Outputs
# -------
# Comparison plots containing:
#   - Measured power (ground truth)
#   - Predicted power
#   - Empirical 90% prediction interval
#
# Dependencies
# ------------
# numpy
# pandas
# matplotlib
#
# Notes
# -----
# The uncertainty interval is residual-based and assumes that future
# prediction errors follow approximately the same distribution observed in
# the evaluation dataset.
# =============================================================================



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

plt.rcParams.update({
    'font.size': 16,        # tamanho geral da fonte
    'axes.titlesize': 20,   # título do gráfico
    'axes.labelsize': 18,   # rótulos dos eixos
    'xtick.labelsize': 14,  # ticks x
    'ytick.labelsize': 14,  # ticks y
    'legend.fontsize': 12   # legenda
})

cv = False

# =========================
# 1. Load data
# =========================

if cv:
    dates = np.loadtxt('dates_cv.txt', dtype=str)
    y_test = np.loadtxt('y_test_cv.txt')
else:
    dates = np.loadtxt('dates.txt', dtype=str)
    y_test = np.loadtxt('y_test.txt')

dates = np.asarray(dates).reshape(-1)
dates = pd.to_datetime(dates, errors='coerce')

# =========================
# 2. Load ALL predictions
# =========================
pred_files = sorted(glob.glob("y_pred*.txt"))

predictions = {}

for file in sorted(pred_files):
    filename = os.path.basename(file)

    if cv:
        # pega só arquivos com _cv
        if not filename.endswith("_cv.txt"):
            continue
    else:
        # pega só arquivos SEM _cv
        if filename.endswith("_cv.txt"):
            continue

    name = os.path.splitext(filename)[0]
    predictions[name] = np.loadtxt(file)


# =========================
# 3. Plot
# =========================

for name, y_pred in predictions.items():

    plt.figure(figsize=(12,6))

    plt.plot(y_test, label='Real', linewidth=2, color='black')

    # if len(y_pred) == len(y_test):
    plt.plot(y_pred, linestyle='--', label=name)

    erro = y_test - y_pred

    q05 = np.percentile(erro, 5)
    q95 = np.percentile(erro, 95)

    lower = y_pred + q05
    upper = y_pred + q95

    plt.fill_between(
        np.arange(len(y_pred)),
        lower,
        upper,
        alpha=0.6,
        label='90% prediction interval'
    )


    plt.title('Wind Power Forecast')
    plt.xlabel('Samples')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()