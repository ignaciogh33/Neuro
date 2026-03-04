"""
hiperparametros_pr2.py
======================
Grid search combinatorio de hiperparámetros para Perceptrón y Adaline
sobre problema_real2.txt usando Modo 1 (80% train / 20% test).

Repite cada combinación N veces para promediar la aleatoriedad del split.
Genera:
  - hiperparametros_perceptron_pr2.csv
  - hiperparametros_adaline_pr2.csv

Ejecutar desde: Neuro/p1/src/
  ../.venv/bin/python hiperparametros_pr2.py
"""

import subprocess
import re
import csv
import sys
import statistics
from itertools import product

# ──────────────────────────────────────────────
# Configuración general
# ──────────────────────────────────────────────
PYTHON      = sys.executable   # usa el mismo intérprete que ejecuta este script
FICHERO     = "P1/problema_real2.txt"
PORCENTAJE  = 0.8              # 80% train / 20% test
N_REPS      = 5                # repeticiones por combinación

# Timeouts
TIMEOUT_PERC    = 10
TIMEOUT_ADALINE = 15

# Tiempo máximo del subprocess (timeout_script * n_clases + margen)
SUBPROCESS_TIMEOUT_PERC    = TIMEOUT_PERC    * 2 + 10
SUBPROCESS_TIMEOUT_ADALINE = TIMEOUT_ADALINE * 2 + 10

# ──────────────────────────────────────────────
# Grids de búsqueda (combinatorio)
# ──────────────────────────────────────────────
GRID_PERC = {
    "alfa":   [0.01, 0.05, 0.1, 0.3, 0.5],
    "umbral": [0.05, 0.1, 0.2, 0.5],
    "sesgo":  [-0.5, 0.0, 0.5],
}

GRID_ADALINE = {
    "alfa":       [0.001, 0.005, 0.01, 0.05, 0.1],
    "tolerancia": [0.001, 0.005, 0.01, 0.05],
    "sesgo":      [-0.5, 0.0, 0.5],
}

# ──────────────────────────────────────────────
# Barra de progreso
# ──────────────────────────────────────────────
def progreso(actual, total, prefijo=""):
    porcentaje = int(100 * actual / total) if total > 0 else 100
    barra = "█" * (porcentaje // 5) + "░" * (20 - porcentaje // 5)
    print(f"\r{prefijo} [{barra}] {porcentaje:3d}% ({actual}/{total})", end="", flush=True)
    if actual == total:
        print()


# ──────────────────────────────────────────────
# Parseo de la salida del script
# ──────────────────────────────────────────────
RE_TRAIN = re.compile(r"Precision entrenamiento:\s*([\d.]+)%")
RE_TEST  = re.compile(r"Precision test:\s*([\d.]+)%")
RE_ECM   = re.compile(r"ECM test:\s*([\d.]+)")

def parsear_salida(stdout: str):
    """Extrae métricas de la salida. Devuelve (train%, test%, ecm) o None."""
    m_train = RE_TRAIN.search(stdout)
    m_test  = RE_TEST.search(stdout)
    m_ecm   = RE_ECM.search(stdout)
    if m_train and m_test and m_ecm:
        return float(m_train.group(1)), float(m_test.group(1)), float(m_ecm.group(1))
    return None


# ──────────────────────────────────────────────
# Ejecutar un experimento
# ──────────────────────────────────────────────
def ejecutar(script: str, params: dict, subprocess_timeout: int):
    """Lanza el script con los params y devuelve (train, test, ecm) o None."""
    cmd = [PYTHON, script, "--fichero", FICHERO, "--porcentaje", str(PORCENTAJE)]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]

    try:
        resultado = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=subprocess_timeout,
        )
        return parsear_salida(resultado.stdout)
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def experimento_n_veces(script, params, n, subprocess_timeout):
    """Repite el experimento n veces y devuelve (media, std) de cada métrica."""
    trains, tests, ecms = [], [], []
    for _ in range(n):
        res = ejecutar(script, params, subprocess_timeout)
        if res is not None:
            t, te, e = res
            trains.append(t)
            tests.append(te)
            ecms.append(e)

    if not trains:
        return None

    def stats(lst):
        media = statistics.mean(lst)
        std   = statistics.stdev(lst) if len(lst) > 1 else 0.0
        return media, std

    return stats(trains), stats(tests), stats(ecms)


# ──────────────────────────────────────────────
# Generar todas las combinaciones de un grid
# ──────────────────────────────────────────────
def generar_combinaciones(grid: dict):
    """Devuelve lista de dicts con todas las combinaciones."""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for combo in product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


# ──────────────────────────────────────────────
# Grid search completo para una red
# ──────────────────────────────────────────────
def grid_search(script, nombre_red, grid, timeout_script, subprocess_timeout, csv_path):
    combinaciones = generar_combinaciones(grid)
    total = len(combinaciones) * N_REPS
    realizados = 0

    filas = []
    mejor = None  # (precision_test_media, fila_dict)

    param_keys = list(grid.keys())

    for combo in combinaciones:
        # Añadir timeout al dict de params
        params = {**combo, "timeout": timeout_script}

        desc = ", ".join([f"{k}={combo[k]}" for k in param_keys])
        progreso(realizados, total, prefijo=f"{nombre_red} [{desc}]")

        res = experimento_n_veces(script, params, N_REPS, subprocess_timeout)
        realizados += N_REPS
        progreso(realizados, total, prefijo=f"{nombre_red} [{desc}]")

        if res is None:
            continue

        (pt_media, pt_std), (pte_media, pte_std), (ecm_media, ecm_std) = res

        fila = {}
        for k in param_keys:
            fila[k] = combo[k]

        fila.update({
            "precision_train":     round(pt_media, 4),
            "precision_train_std": round(pt_std, 4),
            "precision_test":      round(pte_media, 4),
            "precision_test_std":  round(pte_std, 4),
            "ecm_test":            round(ecm_media, 6),
            "ecm_test_std":        round(ecm_std, 6),
        })
        filas.append(fila)

        # Rastrear mejor combinación (mayor precision_test)
        if mejor is None or pte_media > mejor[0]:
            mejor = (pte_media, fila)

    # Escribir CSV
    if filas:
        campos = param_keys + [
            "precision_train", "precision_train_std",
            "precision_test", "precision_test_std",
            "ecm_test", "ecm_test_std",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(filas)

        print(f"\n✔ CSV guardado en: {csv_path}  ({len(filas)} filas)\n")
    else:
        print(f"\n⚠ No se obtuvieron resultados para {nombre_red}\n")

    return mejor


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print(f"  Grid Search de Hiperparámetros — problema_real2.txt")
    print(f"  Porcentaje train: {PORCENTAJE*100:.0f}% | Repeticiones: {N_REPS}")
    print("=" * 65)

    n_perc = 1
    for v in GRID_PERC.values():
        n_perc *= len(v)
    n_adal = 1
    for v in GRID_ADALINE.values():
        n_adal *= len(v)
    print(f"\n  Perceptrón: {n_perc} combinaciones × {N_REPS} reps = {n_perc*N_REPS} ejecuciones")
    print(f"  Adaline:    {n_adal} combinaciones × {N_REPS} reps = {n_adal*N_REPS} ejecuciones")
    print(f"  Total:      {(n_perc+n_adal)*N_REPS} ejecuciones\n")

    # ── Perceptrón ──
    print("▶ Perceptrón")
    mejor_perc = grid_search(
        script="perceptron.py",
        nombre_red="Perceptron",
        grid=GRID_PERC,
        timeout_script=TIMEOUT_PERC,
        subprocess_timeout=SUBPROCESS_TIMEOUT_PERC,
        csv_path="hiperparametros_perceptron_pr2.csv",
    )

    # ── Adaline ──
    print("▶ Adaline")
    mejor_adal = grid_search(
        script="adaline.py",
        nombre_red="Adaline",
        grid=GRID_ADALINE,
        timeout_script=TIMEOUT_ADALINE,
        subprocess_timeout=SUBPROCESS_TIMEOUT_ADALINE,
        csv_path="hiperparametros_adaline_pr2.csv",
    )

    # ── Resumen ──
    print("=" * 65)
    print("  MEJORES CONFIGURACIONES ENCONTRADAS")
    print("=" * 65)

    if mejor_perc:
        _, fila = mejor_perc
        print(f"\n  🏆 Perceptrón:")
        for k in GRID_PERC.keys():
            print(f"     {k}: {fila[k]}")
        print(f"     Precisión test: {fila['precision_test']:.2f}% (±{fila['precision_test_std']:.2f})")
        print(f"     ECM test:       {fila['ecm_test']:.6f}")
    else:
        print("\n  ⚠ Perceptrón: sin resultados válidos")

    if mejor_adal:
        _, fila = mejor_adal
        print(f"\n  🏆 Adaline:")
        for k in GRID_ADALINE.keys():
            print(f"     {k}: {fila[k]}")
        print(f"     Precisión test: {fila['precision_test']:.2f}% (±{fila['precision_test_std']:.2f})")
        print(f"     ECM test:       {fila['ecm_test']:.6f}")
    else:
        print("\n  ⚠ Adaline: sin resultados válidos")

    print(f"\n✔ Grid search completado.\n")
