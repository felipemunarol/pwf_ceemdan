import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================
PASTA_ENTRADA = "dados_previsao"
PASTA_SAIDA = "fft_previsao"

MODELOS = {
    "ann": {
        "real": "y_test_cv.txt",
        "previsto": "y_pred_ann_cv.txt",
    },
    "svr": {
        "real": "y_test_cv.txt",
        "previsto": "y_pred_svr_cv.txt",
    },
}

# Frequência de amostragem:
# 1 amostra por intervalo de tempo.
# Ajuste caso conheça o intervalo real entre amostras.
FS = 1.0

os.makedirs(PASTA_SAIDA, exist_ok=True)

# Armazena os erros para comparação final
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
    frequencias = np.fft.rfftfreq(n, d=1.0 / fs)

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
for modelo, arquivos in MODELOS.items():

    arquivo_real = os.path.join(
        PASTA_ENTRADA, arquivos["real"]
    )

    arquivo_previsto = os.path.join(
        PASTA_ENTRADA, arquivos["previsto"]
    )

    # Verifica a existência dos arquivos
    if not os.path.isfile(arquivo_real):
        print(f"[AVISO] Arquivo não encontrado: {arquivo_real}")
        continue

    if not os.path.isfile(arquivo_previsto):
        print(f"[AVISO] Arquivo não encontrado: {arquivo_previsto}")
        continue

    # Carrega os dados
    real = np.loadtxt(arquivo_real).ravel()
    previsto = np.loadtxt(arquivo_previsto).ravel()

    # Verifica valores inválidos
    if not np.all(np.isfinite(real)):
        print(f"[AVISO] Valores inválidos no arquivo real: {arquivo_real}")
        continue

    if not np.all(np.isfinite(previsto)):
        print(f"[AVISO] Valores inválidos no arquivo previsto: {arquivo_previsto}")
        continue

    # Garante que os sinais tenham o mesmo comprimento
    n = min(len(real), len(previsto))
    real = real[:n]
    previsto = previsto[:n]

    if n < 2:
        print(f"[AVISO] Dados insuficientes para {modelo.upper()}")
        continue

    # Calcula as FFTs
    freq_real, amp_real = calcular_fft(real, FS)
    freq_prev, amp_prev = calcular_fft(previsto, FS)

    # Diferença absoluta entre amplitudes espectrais
    diferenca = np.abs(amp_real - amp_prev)

    # ======================================================
    # ERRO ESPECTRAL RELATIVO (%)
    # ======================================================
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

    print(f"\nModelo: {modelo.upper()}")
    print(f"Amostras: {n}")

    if np.isfinite(erro_relativo):
        print(f"Erro espectral relativo: {erro_relativo:.4f}%")
    else:
        print("Erro espectral relativo: indefinido (espectro real nulo)")

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
        PASTA_SAIDA, f"fft_{modelo}.txt"
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
        2, 1,
        figsize=(12, 8),
        sharex=True
    )

    # Espectro real e previsto
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
    axes[0].grid(True, alpha=0.3)

    # Diferença entre os espectros
    axes[1].plot(
        freq_real,
        diferenca,
        label="Diferença absoluta",
        linewidth=1.2
    )

    axes[1].set_xlabel("Frequência")
    axes[1].set_ylabel("|A real - A previsto|")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Mostra o erro no título do gráfico inferior
    if np.isfinite(erro_relativo):
        axes[1].set_title(
            f"Erro espectral relativo: {erro_relativo:.4f}%"
        )

    plt.tight_layout()

    arquivo_png = os.path.join(
        PASTA_SAIDA, f"fft_{modelo}.png"
    )

    plt.savefig(arquivo_png, dpi=300)
    plt.close(fig)

    print(f"FFT salva em: {arquivo_txt}")
    print(f"Gráfico salvo em: {arquivo_png}")

# ==========================================================
# RESUMO FINAL DO ERRO ESPECTRAL
# ==========================================================
arquivo_resumo = os.path.join(
    PASTA_SAIDA, "resumo_erros.txt"
)

with open(arquivo_resumo, "w", encoding="utf-8") as f:

    f.write("RELATORIO DE ERRO ESPECTRAL\n")
    f.write("=" * 65 + "\n\n")

    # ------------------------------------------------------
    # 1. DEFINICAO DA METRICA
    # ------------------------------------------------------
    f.write("1. O QUE E O ERRO ESPECTRAL RELATIVO?\n\n")

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
    # 2. FORMULA
    # ------------------------------------------------------
    f.write("2. FORMULA UTILIZADA\n\n")

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
    f.write("3. RESULTADOS\n\n")

    if resultados_erros:
        for modelo, erro in resultados_erros:
            if np.isfinite(erro):
                f.write(
                    f"{modelo}: erro espectral relativo = {erro:.6f}%\n"
                )
            else:
                f.write(
                    f"{modelo}: erro indefinido "
                    "(espectro real sem amplitude total).\n"
                )
    else:
        f.write("Nenhum modelo foi avaliado.\n")

    f.write("\n")

    # ------------------------------------------------------
    # 4. COMPARACAO
    # ------------------------------------------------------
    f.write("4. COMPARACAO ENTRE OS MODELOS\n\n")

    erros_validos = [
        (modelo, erro)
        for modelo, erro in resultados_erros
        if np.isfinite(erro)
    ]

    if erros_validos:
        melhor = min(erros_validos, key=lambda x: x[1])

        f.write(
            f"O menor erro espectral relativo foi obtido pelo modelo "
            f"{melhor[0]}, com {melhor[1]:.6f}%.\n\n"
        )

        f.write(
            "Um erro menor indica maior concordancia global entre as "
            "amplitudes espectrais previstas e reais, segundo esta metrica.\n\n"
        )

        if len(erros_validos) >= 2:
            ordenados = sorted(erros_validos, key=lambda x: x[1])
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
    # 5. INTERPRETACAO
    # ------------------------------------------------------
    f.write("5. COMO INTERPRETAR O RESULTADO\n\n")

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
    # 6. LIMITACOES
    # ------------------------------------------------------
    f.write("6. LIMITACOES DA METRICA\n\n")

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
    # 7. RECOMENDACOES
    # ------------------------------------------------------
    f.write("7. RECOMENDACOES PARA AVALIACAO DOS MODELOS\n\n")

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

print(f"\nRelatorio salvo em: {arquivo_resumo}")
print("Processamento concluido.")
