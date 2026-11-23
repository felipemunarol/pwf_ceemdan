import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

PASTA_ENTRADA = "predictions"
PASTA_SAIDA = "predictions"

ARQUIVO_REAL = "y_test_cv.txt"

# Frequência de amostragem
# 1 amostra por intervalo de tempo.
FS = 1.0

os.makedirs(PASTA_SAIDA, exist_ok=True)


# ==========================================================
# IDENTIFICAÇÃO AUTOMÁTICA DOS MODELOS
# ==========================================================

arquivo_real = os.path.join(PASTA_ENTRADA, ARQUIVO_REAL)

if not os.path.isfile(arquivo_real):
    raise FileNotFoundError(
        f"Arquivo com valores reais não encontrado: {arquivo_real}"
    )

MODELOS = {}

for arquivo in os.listdir(PASTA_ENTRADA):

    if arquivo.startswith("y_pred_") and arquivo.endswith("_cv.txt"):

        modelo = arquivo[len("y_pred_"):-len("_cv.txt")]

        MODELOS[modelo] = arquivo

if not MODELOS:
    raise FileNotFoundError(
        f"Nenhum arquivo y_pred_<modelo>_cv.txt encontrado em {PASTA_ENTRADA}"
    )

print("\nModelos identificados:")

for modelo in sorted(MODELOS):
    print(f" - {modelo.upper()}")

# Carrega os valores reais uma única vez
real = np.loadtxt(arquivo_real).ravel()

if not np.all(np.isfinite(real)):
    raise ValueError(
        f"Valores inválidos no arquivo: {arquivo_real}"
    )


# ==========================================================
# ARMAZENAMENTO DOS RESULTADOS
# ==========================================================

resultados_erros = []


# ==========================================================
# FUNÇÃO PARA CALCULAR A FFT
# ==========================================================

def calcular_fft(sinal, fs):

    sinal = np.asarray(sinal, dtype=float).ravel()

    # Remove a média para destacar as oscilações
    sinal = sinal - np.mean(sinal)

    n = len(sinal)

    # FFT unilateral
    espectro = np.fft.rfft(sinal)

    frequencias = np.fft.rfftfreq(
        n,
        d=1.0 / fs
    )

    # Amplitude normalizada
    amplitude = np.abs(espectro) / n

    if n > 1:

        # Duplica as amplitudes das frequências positivas,
        # exceto a componente de Nyquist quando n é par.

        if n % 2 == 0:
            amplitude[1:-1] *= 2

        else:
            amplitude[1:] *= 2

    return frequencias, amplitude


# ==========================================================
# PROCESSAMENTO DOS MODELOS
# ==========================================================

for modelo, arquivo_previsto_nome in sorted(MODELOS.items()):

    print("\n" + "=" * 60)
    print(f"PROCESSANDO MODELO: {modelo.upper()}")
    print("=" * 60)

    arquivo_previsto = os.path.join(
        PASTA_ENTRADA,
        arquivo_previsto_nome
    )

    # ------------------------------------------------------
    # CARREGAMENTO DOS DADOS
    # ------------------------------------------------------

    previsto = np.loadtxt(arquivo_previsto).ravel()

    # Verifica valores inválidos
    if not np.all(np.isfinite(previsto)):

        print(
            f"[AVISO] Valores inválidos no arquivo: "
            f"{arquivo_previsto}"
        )

        continue

    # ------------------------------------------------------
    # GARANTE QUE OS SINAIS TENHAM O MESMO COMPRIMENTO
    # ------------------------------------------------------

    if len(real) != len(previsto):

        print(
            f"[AVISO] Comprimentos diferentes para {modelo.upper()}: "
            f"real={len(real)}, previsto={len(previsto)}"
        )

        print("Os sinais serão truncados para o menor comprimento.")

    n = min(len(real), len(previsto))

    real_modelo = real[:n]
    previsto = previsto[:n]

    if n < 2:

        print(
            f"[AVISO] Dados insuficientes para {modelo.upper()}"
        )

        continue

    # ------------------------------------------------------
    # CALCULA AS FFTs
    # ------------------------------------------------------

    freq_real, amp_real = calcular_fft(real_modelo, FS)

    freq_prev, amp_prev = calcular_fft(previsto, FS)

    # Diferença absoluta entre amplitudes espectrais
    diferenca = np.abs(amp_real - amp_prev)

    # ------------------------------------------------------
    # ERRO ESPECTRAL RELATIVO (%)
    # ------------------------------------------------------

    denominador = np.sum(amp_real)

    if denominador > 0:

        erro_relativo = (
            np.sum(diferenca) / denominador
        ) * 100

    else:

        erro_relativo = np.nan

    resultados_erros.append(
        (modelo.upper(), erro_relativo)
    )

    print(f"Amostras: {n}")

    if np.isfinite(erro_relativo):

        print(
            f"Erro espectral relativo: {erro_relativo:.4f}%"
        )

    else:

        print(
            "Erro espectral relativo: indefinido "
            "(espectro real nulo)"
        )

    # ======================================================
    # SALVAR ESPECTROS EM TXT
    # ======================================================

    dados_fft = np.column_stack((
        freq_real,
        amp_real,
        amp_prev,
        diferenca
    ))

    arquivo_txt = os.path.join(
        PASTA_SAIDA,
        f"fft_{modelo}.txt"
    )

    np.savetxt(
        arquivo_txt,
        dados_fft,
        delimiter=";",
        header="frequencia;amplitude_real;amplitude_prevista;diferenca",
        comments="",
        fmt="%.10f"
    )

    # ======================================================
    # GRÁFICO COMPARATIVO
    # ======================================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True
    )

    # ------------------------------------------------------
    # ESPECTRO REAL E PREVISTO
    # ------------------------------------------------------

    axes[0].plot(
        freq_real,
        amp_real,
        label="Real",
        linewidth=1.2
    )

    axes[0].plot(
        freq_prev,
        amp_prev,
        label="Previsto",
        linewidth=1.2
    )

    axes[0].set_title(
        f"FFT - Modelo {modelo.upper()}"
    )

    axes[0].set_ylabel("Amplitude")

    axes[0].legend()

    axes[0].grid(
        True,
        alpha=0.3
    )

    # ------------------------------------------------------
    # DIFERENÇA ENTRE OS ESPECTROS
    # ------------------------------------------------------

    axes[1].plot(
        freq_real,
        diferenca,
        label="Diferença absoluta",
        linewidth=1.2
    )

    axes[1].set_xlabel("Frequência")

    axes[1].set_ylabel(
        "|A real - A previsto|"
    )

    axes[1].legend()

    axes[1].grid(
        True,
        alpha=0.3
    )

    # ------------------------------------------------------
    # ERRO NO TÍTULO DO GRÁFICO
    # ------------------------------------------------------

    if np.isfinite(erro_relativo):

        axes[1].set_title(
            f"Erro espectral relativo: {erro_relativo:.4f}%"
        )

    plt.tight_layout()

    arquivo_png = os.path.join(
        PASTA_SAIDA,
        f"fft_{modelo}.png"
    )

    plt.savefig(
        arquivo_png,
        dpi=300
    )

    plt.close(fig)

    print(f"FFT salva em: {arquivo_txt}")

    print(f"Gráfico salvo em: {arquivo_png}")


# ==========================================================
# RESUMO FINAL DO ERRO ESPECTRAL
# ==========================================================

arquivo_resumo = os.path.join(
    PASTA_SAIDA,
    "resumo_erros.txt"
)

with open(
    arquivo_resumo,
    "w",
    encoding="utf-8"
) as f:

    f.write("RELATORIO DE ERRO ESPECTRAL\n")

    f.write("=" * 65 + "\n\n")

    # ------------------------------------------------------
    # 1. DEFINIÇÃO DA MÉTRICA
    # ------------------------------------------------------

    f.write(
        "1. O QUE E O ERRO ESPECTRAL RELATIVO?\n\n"
    )

    f.write(
        "O erro espectral relativo mede a diferenca entre as amplitudes "
        "do espectro de frequencias do sinal real e do sinal previsto "
        "pelo modelo. O resultado e expresso em porcentagem.\n\n"
    )

    f.write(
        "A metrica permite avaliar o quanto o modelo reproduz a "
        "distribuicao das oscilacoes do sinal no dominio da frequencia.\n\n"
    )

    # ------------------------------------------------------
    # 2. FÓRMULA
    # ------------------------------------------------------

    f.write(
        "2. FORMULA UTILIZADA\n\n"
    )

    f.write(
        "Erro espectral relativo (%) =\n"
        "100 * soma(|A_real(k) - A_previsto(k)|) / soma(A_real(k))\n\n"
    )

    f.write("Onde:\n")

    f.write(
        "- A_real(k): amplitude da frequencia k no espectro real.\n"
    )

    f.write(
        "- A_previsto(k): amplitude da frequencia k no espectro previsto.\n"
    )

    f.write(
        "- A diferenca absoluta e calculada para cada componente "
        "de frequencia.\n"
    )

    f.write(
        "- O denominador corresponde a soma das amplitudes do "
        "espectro real.\n\n"
    )

    # ------------------------------------------------------
    # 3. RESULTADOS
    # ------------------------------------------------------

    f.write(
        "3. RESULTADOS\n\n"
    )

    if resultados_erros:

        for modelo, erro in resultados_erros:

            if np.isfinite(erro):

                f.write(
                    f"{modelo}: erro espectral relativo = "
                    f"{erro:.6f}%\n"
                )

            else:

                f.write(
                    f"{modelo}: erro indefinido "
                    "(espectro real sem amplitude total).\n"
                )

    else:

        f.write(
            "Nenhum modelo foi avaliado.\n"
        )

    f.write("\n")

    # ------------------------------------------------------
    # 4. COMPARAÇÃO
    # ------------------------------------------------------

    f.write(
        "4. COMPARACAO ENTRE OS MODELOS\n\n"
    )

    erros_validos = [
        (modelo, erro)
        for modelo, erro in resultados_erros
        if np.isfinite(erro)
    ]

    if erros_validos:

        melhor = min(
            erros_validos,
            key=lambda x: x[1]
        )

        f.write(
            f"O menor erro espectral relativo foi obtido pelo modelo "
            f"{melhor[0]}, com {melhor[1]:.6f}%.\n\n"
        )

        f.write(
            "Um erro menor indica maior concordancia global entre as "
            "amplitudes espectrais previstas e reais, segundo esta metrica.\n\n"
        )

        if len(erros_validos) >= 2:

            ordenados = sorted(
                erros_validos,
                key=lambda x: x[1]
            )

            segundo = ordenados[1]

            f.write(
                f"O modelo {segundo[0]} apresentou erro de "
                f"{segundo[1]:.6f}%.\n\n"
            )

    else:

        f.write(
            "Nao foi possivel comparar os modelos porque nao ha "
            "resultados validos.\n\n"
        )

    # ------------------------------------------------------
    # 5. INTERPRETAÇÃO
    # ------------------------------------------------------

    f.write(
        "5. COMO INTERPRETAR O RESULTADO\n\n"
    )

    f.write(
        "Quanto menor o erro espectral relativo, maior a concordancia "
        "entre os espectros de amplitude real e previsto.\n\n"
    )

    f.write(
        "Exemplo ilustrativo: um erro de 5% indica uma diferenca "
        "espectral relativa menor do que um erro de 20%. Esses valores "
        "nao constituem limites universais de qualidade.\n\n"
    )

    f.write(
        "O erro nao significa que o modelo errou essa porcentagem em "
        "cada frequencia, nem que o erro da previsao no dominio do "
        "tempo seja igual a esse valor.\n\n"
    )

    # ------------------------------------------------------
    # 6. LIMITAÇÕES
    # ------------------------------------------------------

    f.write(
        "6. LIMITACOES DA METRICA\n\n"
    )

    f.write(
        "A FFT compara as amplitudes por frequencia, mas nao avalia "
        "diretamente se os picos e vales previstos ocorrem nos mesmos "
        "instantes que os do sinal real.\n\n"
    )

    f.write(
        "Neste codigo, a media de cada sinal e removida antes da FFT. "
        "Assim, a comparacao se concentra nas oscilacoes, sem avaliar "
        "diretamente a diferenca entre os niveis medios.\n\n"
    )

    f.write(
        "Dois sinais podem apresentar espectros semelhantes e, ainda "
        "assim, ter erros temporais significativos.\n\n"
    )

    # ------------------------------------------------------
    # 7. RECOMENDAÇÕES
    # ------------------------------------------------------

    f.write(
        "7. RECOMENDACOES PARA AVALIACAO DOS MODELOS\n\n"
    )

    f.write(
        "Recomenda-se complementar o erro espectral relativo com:\n\n"
        "- MAE: erro absoluto medio entre valores reais e previstos.\n"
        "- RMSE: raiz do erro quadratico medio.\n"
        "- R2: coeficiente de determinacao.\n\n"
    )

    f.write(
        "A combinacao dessas metricas permite avaliar tanto a "
        "concordancia espectral quanto a precisao das previsoes "
        "no dominio do tempo.\n"
    )


# ==========================================================
# FINALIZAÇÃO
# ==========================================================

print(
    f"\nRelatorio salvo em: {arquivo_resumo}"
)

print(
    f"Modelos processados: {len(resultados_erros)}"
)

print(
    "Processamento concluido."
)