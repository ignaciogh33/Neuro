"""
graficar.py
===========
Genera gráficas de influencia de parámetros a partir de los CSVs
producidos por generar_csv.py.

Para cada parámetro variado y cada red genera:
  - Línea de Precisión Test (media ± std)
  - Línea de Precisión Train (media ± std)
  - Línea de ECM Test (eje Y secundario)

Salida: carpeta 'graficas/' con un PNG por cada (red, parámetro).

Ejecutar desde: Neuro/p1/src/
  ../.venv/bin/python graficar.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ──────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────
CSV_PERC    = "resultados_perceptron.csv"
CSV_ADALINE = "resultados_adaline.csv"
SALIDA_DIR  = "graficas"
os.makedirs(SALIDA_DIR, exist_ok=True)

# Etiquetas legibles para los parámetros
ETIQUETAS = {
    "alfa":       "Tasa de aprendizaje (α)",
    "umbral":     "Umbral (θ)",
    "sesgo":      "Sesgo inicial (b)",
    "tolerancia": "Tolerancia de convergencia",
}

# Paleta de colores consistente
COLOR_TRAIN = "#2196F3"   # azul
COLOR_TEST  = "#4CAF50"   # verde
COLOR_ECM   = "#FF5722"   # naranja

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.grid":    True,
    "grid.alpha":   0.3,
    "figure.dpi":   130,
})


# ──────────────────────────────────────────────
# Función principal de plot
# ──────────────────────────────────────────────
def graficar_parametro(df_param, red, param, ax_prec, ax_ecm):
    """
    Dibuja en ax_prec (precisión train/test) y ax_ecm (ECM) las curvas
    con banda de error (±1 std) para un único parámetro variado.
    """
    x = df_param["valor_parametro"].values

    # ── Precisión entrenamiento ──
    ax_prec.plot(x, df_param["precision_train"],
                 "o-", color=COLOR_TRAIN, label="Precisión entrenamiento", linewidth=2, markersize=5)
    ax_prec.fill_between(x,
                         df_param["precision_train"] - df_param["precision_train_std"],
                         df_param["precision_train"] + df_param["precision_train_std"],
                         alpha=0.15, color=COLOR_TRAIN)

    # ── Precisión test ──
    ax_prec.plot(x, df_param["precision_test"],
                 "s-", color=COLOR_TEST, label="Precisión test", linewidth=2, markersize=5)
    ax_prec.fill_between(x,
                         df_param["precision_test"] - df_param["precision_test_std"],
                         df_param["precision_test"] + df_param["precision_test_std"],
                         alpha=0.15, color=COLOR_TEST)

    ax_prec.set_ylabel("Precisión (%)", color="black")
    ax_prec.set_ylim(bottom=max(0, df_param["precision_test"].min() - 15), top=101)
    ax_prec.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax_prec.legend(loc="lower left", fontsize=9)

    # ── ECM ──
    ax_ecm.plot(x, df_param["ecm_test"],
                "^--", color=COLOR_ECM, label="ECM test", linewidth=1.8, markersize=5)
    ax_ecm.fill_between(x,
                        df_param["ecm_test"] - df_param["ecm_test_std"],
                        df_param["ecm_test"] + df_param["ecm_test_std"],
                        alpha=0.1, color=COLOR_ECM)
    ax_ecm.set_ylabel("ECM test", color=COLOR_ECM)
    ax_ecm.tick_params(axis="y", labelcolor=COLOR_ECM)
    ax_ecm.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax_ecm.legend(loc="upper right", fontsize=9)


def generar_graficas_red(csv_path, nombre_red):
    df = pd.read_csv(csv_path)
    params = df["parametro_variado"].unique()

    for param in params:
        df_p = df[df["parametro_variado"] == param].copy()
        df_p = df_p.sort_values("valor_parametro")

        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax2 = ax1.twinx()

        graficar_parametro(df_p, nombre_red, param, ax1, ax2)

        etiqueta = ETIQUETAS.get(param, param)
        ax1.set_xlabel(etiqueta)
        fig.suptitle(f"{nombre_red} — Influencia de {etiqueta}", fontsize=13, fontweight="bold", y=1.01)

        # Escala logarítmica en X si los valores varían más de 2 órdenes de magnitud
        rango = df_p["valor_parametro"].values
        positivos = rango[rango > 0]
        if len(positivos) == len(rango) and positivos.max() / positivos.min() > 50:
            ax1.set_xscale("log")

        fig.tight_layout()
        nombre_archivo = f"{SALIDA_DIR}/{nombre_red.lower()}_{param}.png"
        fig.savefig(nombre_archivo, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✔ {nombre_archivo}")


# ──────────────────────────────────────────────
# Figura resumen: todos los parámetros juntos
# ──────────────────────────────────────────────
def generar_resumen(csv_path, nombre_red):
    df = pd.read_csv(csv_path)
    params = list(df["parametro_variado"].unique())
    n = len(params)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        df_p = df[df["parametro_variado"] == param].sort_values("valor_parametro")
        x = df_p["valor_parametro"].values

        ax.plot(x, df_p["precision_train"], "o-", color=COLOR_TRAIN,
                label="Train", linewidth=2, markersize=4)
        ax.fill_between(x,
                        df_p["precision_train"] - df_p["precision_train_std"],
                        df_p["precision_train"] + df_p["precision_train_std"],
                        alpha=0.15, color=COLOR_TRAIN)

        ax.plot(x, df_p["precision_test"], "s-", color=COLOR_TEST,
                label="Test", linewidth=2, markersize=4)
        ax.fill_between(x,
                        df_p["precision_test"] - df_p["precision_test_std"],
                        df_p["precision_test"] + df_p["precision_test_std"],
                        alpha=0.15, color=COLOR_TEST)

        etiqueta = ETIQUETAS.get(param, param)
        ax.set_xlabel(etiqueta, fontsize=10)
        ax.set_ylabel("Precisión (%)" if ax == axes[0] else "")
        ax.set_title(etiqueta, fontsize=10, fontweight="bold")
        ax.set_ylim(bottom=max(0, df_p["precision_test"].min() - 20), top=101)
        ax.grid(True, alpha=0.3)

        positivos = x[x > 0]
        if len(positivos) == len(x) and positivos.max() / positivos.min() > 50:
            ax.set_xscale("log")

    # Leyenda compartida
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.04))

    fig.suptitle(f"{nombre_red} — Influencia de parámetros (Precisión)",
                 fontsize=13, fontweight="bold", y=1.08)
    fig.tight_layout()
    nombre_archivo = f"{SALIDA_DIR}/{nombre_red.lower()}_resumen.png"
    fig.savefig(nombre_archivo, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ {nombre_archivo}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("▶ Generando gráficas individuales del Perceptrón...")
    generar_graficas_red(CSV_PERC, "Perceptron")
    generar_resumen(CSV_PERC, "Perceptron")

    print("▶ Generando gráficas individuales del Adaline...")
    generar_graficas_red(CSV_ADALINE, "Adaline")
    generar_resumen(CSV_ADALINE, "Adaline")

    print(f"\n✔ Todas las gráficas guardadas en '{SALIDA_DIR}/'")
