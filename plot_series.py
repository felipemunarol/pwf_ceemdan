# =============================================================================
# File      : plot_wind_power_intro.py
# Author    : Felipe Munaro Lima
# Version   : 1.0.0
# Date      : 2026-08-20
#
# Descrição
# ----------
# Gera uma única figura contendo a potência de geração eólica em diferentes
# períodos da série histórica, para utilização em apresentação.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# =============================================================================
# 1. Configuração
# =============================================================================

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 19,
    'axes.labelsize': 17,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13
})


arquivo = 'dataset/final_la_haute_R0711.csv'

pasta_saida = 'wind_power_intro'
os.makedirs(pasta_saida, exist_ok=True)

arquivo_saida = os.path.join(
    pasta_saida,
    'potencia_eolica_diferentes_periodos.png'
)


# =============================================================================
# 2. Carregar dados
# =============================================================================

df = pd.read_csv(arquivo)

df['Date_time'] = pd.to_datetime(
    df['Date_time'],
    errors='coerce'
)

df = df.dropna(
    subset=['Date_time', 'P_avg']
)

df = df.sort_values(
    'Date_time'
).reset_index(drop=True)


# =============================================================================
# 3. Definir períodos
# =============================================================================

data_inicial = df['Date_time'].min()
data_final = df['Date_time'].max()

intervalo_total = data_final - data_inicial

# Períodos distribuídos ao longo de toda a série
fracoes = [0.00, 0.33, 0.66, 0.90]

inicio_periodos = [
    data_inicial + intervalo_total * f
    for f in fracoes
]

# Duração de cada período apresentado
duracao_periodo = pd.Timedelta(days=7)


# =============================================================================
# 4. Criar figura
# =============================================================================

fig, axes = plt.subplots(
    4,
    1,
    figsize=(14, 11),
    sharey=True
)


# =============================================================================
# 5. Plotar períodos
# =============================================================================

for i, (ax, inicio) in enumerate(
    zip(axes, inicio_periodos),
    start=1
):

    fim = inicio + duracao_periodo

    dados = df[
        (df['Date_time'] >= inicio) &
        (df['Date_time'] < fim)
    ].copy()


    if dados.empty:

        ax.text(
            0.5,
            0.5,
            'Dados não disponíveis',
            transform=ax.transAxes,
            ha='center',
            va='center'
        )

        ax.set_title(
            f'Período {i}'
        )

        continue


    # -------------------------------------------------------------------------
    # Potência eólica
    # -------------------------------------------------------------------------

    ax.plot(
        dados['Date_time'],
        dados['P_avg'],
        linewidth=1.5
    )


    # -------------------------------------------------------------------------
    # Título do período
    # -------------------------------------------------------------------------

    data_inicio = dados['Date_time'].iloc[0].strftime('%d/%m/%Y')
    data_fim = dados['Date_time'].iloc[-1].strftime('%d/%m/%Y')

    ax.set_title(
        f'Período {i}: {data_inicio} – {data_fim}',
        loc='left',
        pad=8
    )


    # -------------------------------------------------------------------------
    # Grade
    # -------------------------------------------------------------------------

    ax.grid(
        True,
        alpha=0.25
    )


    # -------------------------------------------------------------------------
    # Remover bordas superiores e direitas
    # -------------------------------------------------------------------------

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    # -------------------------------------------------------------------------
    # Formatação das datas
    # -------------------------------------------------------------------------

    ax.tick_params(
        axis='x',
        rotation=0
    )


# =============================================================================
# 6. Rótulos dos eixos
# =============================================================================

axes[-1].set_xlabel(
    'Data'
)

fig.text(
    0.02,
    0.5,
    'Potência de geração (kW)',
    va='center',
    rotation='vertical',
    fontsize=18
)


# =============================================================================
# 7. Título
# =============================================================================

fig.suptitle(
    'Variabilidade temporal da geração de energia eólica',
    fontsize=23,
    y=0.995
)


# =============================================================================
# 8. Layout
# =============================================================================

plt.tight_layout(
    rect=[0.04, 0.02, 1, 0.97]
)


# =============================================================================
# 9. Salvar
# =============================================================================

plt.savefig(
    arquivo_saida,
    dpi=300,
    bbox_inches='tight'
)

plt.close()


print(
    f'Figura salva em: {arquivo_saida}'
)