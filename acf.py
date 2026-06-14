import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []

    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)


# ==========================================================
# Leitura dos dados
# ==========================================================

df = pd.read_csv('dataset/final_la_haute_R0711.csv')

df['Date'] = pd.to_datetime(
    df['Date_time'],
    format='%m/%d/%Y %H:%M',
    errors='coerce'
)

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

new_data = df[['Month', 'Year', 'Date', 'P_avg']]
new_data = new_data[new_data['Year'] == 2017]

cap = new_data['P_avg'].max()

# ==========================================================
# Parâmetros
# ==========================================================

month_selected = [1]      # Janeiro
look_back = 6
data_partition = 0.8

# ==========================================================
# Seleção dos dados
# ==========================================================

data1 = new_data.loc[
    new_data['Month'].isin(month_selected)
]

data1 = data1.reset_index(drop=True)
data1 = data1.dropna()

datas = data1['P_avg']

s = datas.values

datasetss2 = pd.DataFrame(s)
datasets = datasetss2.values

# ==========================================================
# ACF da série que será usada no modelo
# ==========================================================

serie = datasets[:, 0]

fig, ax = plt.subplots(figsize=(12, 5))

plot_acf(
    serie,
    lags=min(100, len(serie)//2),
    ax=ax,
    fft=True,
    alpha=0.05
)

ax.set_title('Função de Autocorrelação (ACF) - P_avg')
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelação')


# Salvar figura
plt.savefig(
    'acf_pavg.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close(fig)

print("Figura salva em: acf_pavg.png")  

plt.tight_layout()
plt.show()

# ==========================================================
# Valores numéricos da ACF
# ==========================================================

nlags = min(100, len(serie)//2)

acf_values, confint = acf(
    serie,
    nlags=nlags,
    alpha=0.05,
    fft=True
)

df_acf = pd.DataFrame({
    'Lag': np.arange(len(acf_values)),
    'ACF': acf_values,
    'CI_lower': confint[:, 0],
    'CI_upper': confint[:, 1]
})

df_acf.to_csv(
    'acf_values.csv',
    index=False
)

print("Resultados salvos em: acf_values.csv")

# ==========================================================
# Lags estatisticamente significativos
# ==========================================================

print("\nLags com autocorrelação significativa (95%):")

for lag in range(1, len(acf_values)):

    lower = confint[lag, 0]
    upper = confint[lag, 1]

    if lower > 0 or upper < 0:
        print(
            f"Lag {lag:3d} | "
            f"ACF = {acf_values[lag]: .4f}"
        )


# ==========================================================
# PACF da série que será usada no modelo
# ==========================================================

fig, ax = plt.subplots(figsize=(12,5))

plot_pacf(
    serie,
    lags=50,
    ax=ax,
    method='ywm'
)

plt.tight_layout()
plt.savefig(
    'pacf_pavg.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("PACF salva em pacf_pavg.png")

