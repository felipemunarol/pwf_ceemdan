import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import (
    norm,
    chi2,
    shapiro,
    jarque_bera,
    anderson,
    probplot
)

DATA_DIR = "."  # pasta onde estão os txt

# =========================
# 1. Carregar dados
# =========================
y_test = np.loadtxt("y_test.txt")

pred_files = sorted(glob(os.path.join(DATA_DIR, "y_pred*.txt")))

for pred_file in pred_files:

    try:

        # =========================
        # 2. Carregar predição
        # =========================
        y_pred = np.loadtxt(pred_file)

        # =========================
        # 3. Resíduo
        # =========================
        residuo = y_test - y_pred

        mu = np.mean(residuo)
        std = np.std(residuo)

        # ==========================================================
        # 4. Teste Qui-Quadrado (Gaussianidade)
        # ==========================================================
        k = 20  # número de bins

        # frequências observadas
        obs, bins = np.histogram(residuo, bins=k)

        # probabilidades esperadas pela normal ajustada
        cdf_vals = norm.cdf(bins, loc=mu, scale=std)
        p = np.diff(cdf_vals)

        # frequências esperadas
        exp = len(residuo) * p

        # evitar bins vazios
        valid = exp > 0

        obs = obs[valid]
        exp = exp[valid]

        # estatística chi²
        chi2_stat = np.sum((obs - exp) ** 2 / exp)

        # graus de liberdade
        dof = len(obs) - 1 - 2

        # p-value
        chi2_pvalue = 1 - chi2.cdf(chi2_stat, dof)

        # ==========================================================
        # 5. Outros testes de normalidade
        # ==========================================================

        # Shapiro-Wilk
        shapiro_stat, shapiro_p = shapiro(residuo)

        # Jarque-Bera
        jb_stat, jb_p = jarque_bera(residuo)

        # Anderson-Darling
        anderson_result = anderson(residuo)

        # ==========================================================
        # 6. Fit linear y_pred vs y_true
        # ==========================================================
        a, b = np.polyfit(y_test, y_pred, 1)

        corr = np.corrcoef(y_test, y_pred)[0, 1]

        x_fit = np.linspace(np.min(y_test), np.max(y_test), 200)
        y_fit = a * x_fit + b

        # nome base
        model_name = pred_file.split("pred_")[1].split(".")[0]

        # ==========================================================
        # Configuração visual
        # ==========================================================
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 12,
        })

        # ==========================================================
        # 7. Histograma dos resíduos
        # ==========================================================
        fig, ax = plt.subplots(figsize=(5.0, 4.0))

        counts, bins_hist, _ = ax.hist(
            residuo,
            bins=40,
            density=True,
            alpha=0.6,
        )

        # curva normal ajustada
        x = np.linspace(bins_hist.min(), bins_hist.max(), 300)
        pdf = norm.pdf(x, mu, std)

        ax.plot(
            x,
            pdf,
            linewidth=2,
            # label='Gaussian fit'
        )

        # linhas de referência
        ax.axvline(mu, linestyle='--', linewidth=1)
        ax.axvline(0, linestyle=':', linewidth=1)

        ax.set_title(model_name)

        ax.set_xlabel("Residuals")
        ax.set_ylabel("Density")

        # ==========================================================
        # Texto com estatísticas
        # ==========================================================
        textstr = (
            f"$\\mu={mu:.3f}$\n"
            f"$\\sigma={std:.3f}$\n"
            f"$\\chi^2={chi2_stat:.2f}$\n"
            f"$p_{{\\chi^2}}={chi2_pvalue:.3f}$\n"
            f"$p_{{SW}}={shapiro_p:.3f}$\n"
            f"$p_{{JB}}={jb_p:.3f}$"
        )

        ax.text(
            0.97,
            0.97,
            textstr,
            transform=ax.transAxes,
            ha='right',
            va='top',
            bbox=dict(alpha=0.1)
        )

        ax.grid(alpha=0.3)

        plt.tight_layout()

        # salvar
        plt.savefig(
            f"residuals_histogram_{model_name}.png",
            dpi=300
        )

        plt.savefig(
            f"residuals_histogram_{model_name}.pdf"
        )

        # plt.show()

        # ==========================================================
        # 8. QQ-Plot
        # ==========================================================
        fig, ax = plt.subplots(figsize=(4.5, 4.5))

        probplot(
            residuo,
            dist="norm",
            plot=ax
        )

        ax.set_title(f"QQ-Plot - {model_name}")

        ax.grid(alpha=0.3)

        plt.tight_layout()

        # salvar
        plt.savefig(
            f"qqplot_{model_name}.png",
            dpi=300
        )

        plt.savefig(
            f"qqplot_{model_name}.pdf"
        )

        # plt.show()

        # ==========================================================
        # 9. Scatter y_pred vs y_true
        # ==========================================================
        fig, ax = plt.subplots(figsize=(4.5, 4.0))

        ax.scatter(
            y_test,
            y_pred,
            s=15,
            alpha=0.7
        )

        # reta ajustada
        ax.plot(
            x_fit,
            y_fit,
            linewidth=2,
            label=fr"$y={a:.3f}x + {b:.3f}$"
        )

        # reta ideal
        ax.plot(
            x_fit,
            x_fit,
            linestyle='--',
            linewidth=1,
            label=r"$y=x$"
        )

        ax.set_title(model_name)

        ax.set_xlabel(r"$y_{true}$")
        ax.set_ylabel(r"$y_{pred}$")

        ax.text(
            0.05,
            0.95,
            f"Slope = {a:.4f}\n"
            f"Intercept = {b:.4f}\n"
            f"Corr = {corr:.4f}",
            transform=ax.transAxes,
            ha='right',
            va='top'
        )

        ax.grid(alpha=0.3)

        ax.legend()

        plt.tight_layout()

        # salvar
        plt.savefig(
            f"scatter_{model_name}.png",
            dpi=300
        )

        plt.savefig(
            f"scatter_{model_name}.pdf"
        )

        # plt.show()

        # ==========================================================
        # 9. Resultados
        # ==========================================================
        print("\n====================================")
        print(f"Modelo: {model_name}")
        print("====================================")

        print(f"Slope      : {a}")
        print(f"Intercept  : {b}")
        print(f"Correlation: {corr}")

        print("\n--- Gaussianity Tests ---")

        print(f"Chi² statistic : {chi2_stat}")
        print(f"Chi² p-value   : {chi2_pvalue}")

        print(f"\nShapiro p-value: {shapiro_p}")

        print(f"\nJarque-Bera p-value: {jb_p}")

        print(f"\nAnderson-Darling statistic: "
              f"{anderson_result.statistic}")

        print("Critical values:")
        for cv, sl in zip(
            anderson_result.critical_values,
            anderson_result.significance_level
        ):
            print(f"{sl}% : {cv}")

    except Exception as e:

        print(f"Erro em {pred_file}")
        print(e)