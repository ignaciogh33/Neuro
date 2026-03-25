import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

COLOR_TRAIN = "#2196F3"
COLOR_TEST  = "#4CAF50"
COLOR_ECM   = "#FF5722"

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.grid":    True,
    "grid.alpha":   0.3,
    "figure.dpi":   130,
})

def graficar_influencia_parametro(df, param, nombre_param, directorio):
    # Agrupar colapsando el resto de parámetros y archivos para sacar media y STD global
    df_agg = df.groupby(param).agg(
        prec_train_mean=('Prec_Train', 'mean'),
        prec_train_std=('Prec_Train', 'std'),
        prec_test_mean=('Prec_Test', 'mean'),
        prec_test_std=('Prec_Test', 'std'),
        ecm_test_mean=('ECM_Test_Medio', 'mean'),
        ecm_test_std=('ECM_Test_Medio', 'std')
    ).reset_index().fillna(0)

    fig, ax_prec = plt.subplots(figsize=(7, 4.5))
    ax_ecm = ax_prec.twinx()

    x = df_agg[param].values

    # Precisión entrenamiento
    ax_prec.plot(x, df_agg["prec_train_mean"], "o-", color=COLOR_TRAIN, label="Prec. Train", linewidth=2, markersize=5)
    ax_prec.fill_between(x, df_agg["prec_train_mean"] - df_agg["prec_train_std"],
                         df_agg["prec_train_mean"] + df_agg["prec_train_std"], alpha=0.15, color=COLOR_TRAIN)

    # Precisión test
    ax_prec.plot(x, df_agg["prec_test_mean"], "s-", color=COLOR_TEST, label="Prec. Test", linewidth=2, markersize=5)
    ax_prec.fill_between(x, df_agg["prec_test_mean"] - df_agg["prec_test_std"],
                         df_agg["prec_test_mean"] + df_agg["prec_test_std"], alpha=0.15, color=COLOR_TEST)

    ax_prec.set_ylabel("Precisión (%)", color="black")
    min_prec = max(0, df_agg["prec_test_mean"].min() - 15)
    ax_prec.set_ylim(bottom=min_prec, top=101)
    ax_prec.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax_prec.legend(loc="lower left", fontsize=9)

    # ECM
    ax_ecm.plot(x, df_agg["ecm_test_mean"], "^--", color=COLOR_ECM, label="ECM Test", linewidth=1.8, markersize=5)
    ax_ecm.fill_between(x, df_agg["ecm_test_mean"] - df_agg["ecm_test_std"],
                        df_agg["ecm_test_mean"] + df_agg["ecm_test_std"], alpha=0.1, color=COLOR_ECM)
    ax_ecm.set_ylabel("ECM Test", color=COLOR_ECM)
    ax_ecm.tick_params(axis="y", labelcolor=COLOR_ECM)
    ax_ecm.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax_ecm.legend(loc="upper right", fontsize=9)

    ax_prec.set_xlabel(nombre_param)
    fig.suptitle(f"MLP — Influencia de {nombre_param}", fontsize=13, fontweight="bold", y=1.01)

    positivos = x[x > 0]
    if len(positivos) == len(x) and positivos.max() / positivos.min() > 50:
        ax_prec.set_xscale("log")

    fig.tight_layout()
    nom_arch = os.path.join(directorio, f"mlp_influencia_{param.lower()}.png")
    fig.savefig(nom_arch, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ {nom_arch}")

def graficar_resumen_multi(df, param, nombre_param, directorio):
    # Obtener la lista de problemas/archivos
    archivos = sorted(df['Archivo'].unique())
    n = len(archivos)
    
    # Configuramos para 2 filas y 2 columnas
    filas = 2
    cols = 2
    
    # Crear la figura 2x2
    fig, axes = plt.subplots(filas, cols, figsize=(12, 10), sharey=False)
    
    # Aplanamos axes para poder iterar fácilmente con un solo índice (0, 1, 2, 3)
    axes_flat = axes.flatten()

    ax_ecms = []

    for i, arc in enumerate(archivos):
        if i >= len(axes_flat): break # Por si hay más de 4 archivos
        
        ax_prec = axes_flat[i]
        
        # Filtrar solo por el archivo actual
        df_archivo = df[df['Archivo'] == arc]
        
        # Agrupar colapsando el resto de parámetros  
        df_agg = df_archivo.groupby(param).agg(
            prec_train_mean=('Prec_Train', 'mean'),
            prec_train_std=('Prec_Train', 'std'),
            prec_test_mean=('Prec_Test', 'mean'),
            prec_test_std=('Prec_Test', 'std'),
            ecm_test_mean=('ECM_Test_Medio', 'mean'),
            ecm_test_std=('ECM_Test_Medio', 'std')
        ).reset_index().fillna(0)

        ax_ecm = ax_prec.twinx()
        ax_ecms.append(ax_ecm)

        x = df_agg[param].values

        # Precisión entrenamiento
        ax_prec.plot(x, df_agg["prec_train_mean"], "o-", color=COLOR_TRAIN, label="Prec. Train", linewidth=2, markersize=4)
        ax_prec.fill_between(x, df_agg["prec_train_mean"] - df_agg["prec_train_std"],
                             df_agg["prec_train_mean"] + df_agg["prec_train_std"], alpha=0.15, color=COLOR_TRAIN)

        # Precisión test
        ax_prec.plot(x, df_agg["prec_test_mean"], "s-", color=COLOR_TEST, label="Prec. Test", linewidth=2, markersize=4)
        ax_prec.fill_between(x, df_agg["prec_test_mean"] - df_agg["prec_test_std"],
                             df_agg["prec_test_mean"] + df_agg["prec_test_std"], alpha=0.15, color=COLOR_TEST)

        # Configurar Y de precisión
        ax_prec.set_ylabel("Precisión (%)", color="black")
        min_prec = max(0, df_agg["prec_test_mean"].min() - 15)
        ax_prec.set_ylim(bottom=min_prec, top=101)
        
        # Títulos y X
        ax_prec.set_title(arc.replace('.txt', ''), fontsize=11, fontweight="bold")
        ax_prec.set_xlabel(nombre_param)
        ax_prec.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

        # ECM
        ax_ecm.plot(x, df_agg["ecm_test_mean"], "^--", color=COLOR_ECM, label="ECM Test", linewidth=1.8, markersize=4)
        ax_ecm.fill_between(x, df_agg["ecm_test_mean"] - df_agg["ecm_test_std"],
                            df_agg["ecm_test_mean"] + df_agg["ecm_test_std"], alpha=0.1, color=COLOR_ECM)
        
        # Configurar Y de ECM
        ax_ecm.set_ylabel("ECM Test", color=COLOR_ECM)
        ax_ecm.tick_params(axis="y", labelcolor=COLOR_ECM)
        ax_ecm.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        # Escala logarítmica si varianza de x es grande
        positivos = x[x > 0]
        if len(positivos) == len(x) and positivos.max() / positivos.min() > 50:
            ax_prec.set_xscale("log")

    # Si hay huecos vacíos (ej. solo 3 archivos), ocultar el cuarto eje
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    # Leyenda unificada
    h1, l1 = axes_flat[0].get_legend_handles_labels()
    h2, l2 = ax_ecms[0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=11)

    fig.suptitle(f"MLP — Influencia de {nombre_param} por Problema", fontsize=16, fontweight="bold", y=1.05)
    
    fig.tight_layout()
    nom_arch = os.path.join(directorio, f"mlp_resumen_multi_{param.lower()}.png")
    fig.savefig(nom_arch, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ {nom_arch}")
    
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    archivo_csv = os.path.join(script_dir, "resultados_mlp_reales.csv")
    
    if not os.path.exists(archivo_csv):
        print(f"Error: No se encuentra el archivo {archivo_csv}")
        return

    df = pd.read_csv(archivo_csv)
    
    metricas = ['ECM_Train_Medio', 'Tasa_Train_Medio', 'ECM_Test_Medio', 'Tasa_Test_Medio']
    for m in metricas:
        df[m] = pd.to_numeric(df[m], errors='coerce')
        
    df['Prec_Train'] = df['Tasa_Train_Medio'] * 100
    df['Prec_Test']  = df['Tasa_Test_Medio'] * 100

    directorio_graficas = os.path.join(script_dir, "graficas_mlp")
    os.makedirs(directorio_graficas, exist_ok=True)

    print("\nGenerando gráficas de influencia de hiperparámetros (Globales)...")
    graficar_influencia_parametro(df, "N_Ocultas", "Neuronas en Capa Oculta", directorio_graficas)
    graficar_influencia_parametro(df, "Alfa", "Tasa de aprendizaje (α)", directorio_graficas)
    graficar_influencia_parametro(df, "Epocas", "Número de Épocas", directorio_graficas)

    print("\nGenerando gráficas desglosadas por problema en paralelo...")
    graficar_resumen_multi(df, "N_Ocultas", "Neuronas en Capa Oculta", directorio_graficas)
    graficar_resumen_multi(df, "Alfa", "Tasa de aprendizaje (α)", directorio_graficas)
    graficar_resumen_multi(df, "Epocas", "Número de Épocas", directorio_graficas)

    print("\nGenerando gráficas complementarias...")
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Archivo', y='Tasa_Test_Medio', errorbar=None, palette="viridis")
    plt.title('Precisión Media en Test por Problema')
    plt.ylabel('Tasa de Aciertos (Test)')
    plt.xlabel('Problema')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(directorio_graficas, 'grafica_precision_por_problema.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='ECM_Test_Medio', y='Tasa_Test_Medio', hue='Archivo', style='Archivo', s=100)
    plt.title('ECM vs Precisión (Test)')
    plt.xlabel('ECM (Test)')
    plt.ylabel('Tasa de Aciertos (Test)')
    plt.tight_layout()
    plt.savefig(os.path.join(directorio_graficas, 'grafica_ecm_vs_precision.png'))
    plt.close()

    print(f"\n✔ Todas las gráficas guardadas en '{directorio_graficas}/'\n")

    print("--- MEJORES CONFIGURACIONES POR PROBLEMA ---")
    idx_mejores = df.groupby('Archivo')['Tasa_Test_Medio'].idxmax()
    for _, row in df.loc[idx_mejores].iterrows():
        print(f"\n🏆 {row['Archivo']}")
        print(f"   Hiperparámetros -> N_Ocultas: {row['N_Ocultas']} | Alfa: {row['Alfa']} | Epocas: {row['Epocas']}")
        print(f"   Precisión Test  -> {row['Prec_Test']:.2f}% | ECM Test: {row['ECM_Test_Medio']:.6f}")

if __name__ == "__main__":
    main()
