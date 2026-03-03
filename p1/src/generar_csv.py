"""
generar_csv.py
==============
Ejecuta perceptron.py y adaline.py variando un parámetro a la vez
(principio de un solo parámetro), repitiendo N veces cada experimento
para promediar la aleatoriedad de leer1 (Modo 1).

Genera:
  - resultados_perceptron.csv
  - resultados_adaline.csv

Ejecutar desde: Neuro/p1/src/
  ../.venv/bin/python generar_csv.py
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
PYTHON      = "../.venv/bin/python"
FICHERO     = "P1/problema_real1.txt"
PORCENTAJE  = 0.7
N_REPS      = 5          # repeticiones por experimento (promedia aleatoriedad)

# Timeout (segundos) pasado AL SCRIPT como argumento --timeout
# (el script ya usa signal.alarm internamente con ese valor)
TIMEOUT_PERC   = 10
TIMEOUT_ADALINE = 15

# Tiempo máximo que esperamos a que el subprocess termine
# = timeout_del_script * n_clases + margen
SUBPROCESS_TIMEOUT_PERC    = TIMEOUT_PERC    * 3 + 10
SUBPROCESS_TIMEOUT_ADALINE = TIMEOUT_ADALINE * 3 + 10

# ──────────────────────────────────────────────
# Parámetros por defecto (base para barridos)
# ──────────────────────────────────────────────
DEFAULTS_PERC = {
    "alfa":    0.1,
    "umbral":  0.2,
    "sesgo":   0.0,
    "timeout": TIMEOUT_PERC,
}

DEFAULTS_ADALINE = {
    "alfa":       0.01,
    "tolerancia": 0.01,
    "sesgo":      0.0,
    "timeout":    TIMEOUT_ADALINE,
}

# ──────────────────────────────────────────────
# Rangos de barrido (un parámetro a la vez)
# ──────────────────────────────────────────────
SWEEP_PERC = {
    "alfa":   [0.01, 0.05, 0.1, 0.3, 0.5],
    "umbral": [0.05, 0.1, 0.2, 0.5, 1.0],
    "sesgo":  [-1.0, -0.5, 0.0, 0.5, 1.0],
}

SWEEP_ADALINE = {
    "alfa":       [0.001, 0.005, 0.01, 0.05, 0.1],
    "tolerancia": [0.001, 0.005, 0.01, 0.05, 0.1],
    "sesgo":      [-1.0, -0.5, 0.0, 0.5, 1.0],
}

# ──────────────────────────────────────────────
# Barra de progreso simple (sin dependencias)
# ──────────────────────────────────────────────
def progreso(actual, total, prefijo=""):
    porcentaje = int(100 * actual / total)
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
    """Extrae métricas de la salida del script. Devuelve (train%, test%, ecm) o None."""
    m_train = RE_TRAIN.search(stdout)
    m_test  = RE_TEST.search(stdout)
    m_ecm   = RE_ECM.search(stdout)
    if m_train and m_test and m_ecm:
        return float(m_train.group(1)), float(m_test.group(1)), float(m_ecm.group(1))
    return None


# ──────────────────────────────────────────────
# Ejecutar un experimento (N repeticiones)
# ──────────────────────────────────────────────
def ejecutar(script: str, params: dict, subprocess_timeout: int):
    """
    Lanza el script con los params dados y devuelve (train, test, ecm).
    Devuelve None si falla el parseo.
    """
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
        print(f"\n  ⚠ Subprocess timeout ({subprocess_timeout}s) en: {' '.join(cmd)}")
        return None
    except Exception as e:
        print(f"\n  ⚠ Error ejecutando {script}: {e}")
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
        return None  # todos fallaron

    def stats(lst):
        media = statistics.mean(lst)
        std   = statistics.stdev(lst) if len(lst) > 1 else 0.0
        return media, std

    return stats(trains), stats(tests), stats(ecms)


# ──────────────────────────────────────────────
# Barrido completo para una red
# ──────────────────────────────────────────────
def barrer(script, nombre_red, defaults, sweep, subprocess_timeout, csv_path):
    # Calcular total de experimentos para la barra de progreso
    total = sum(len(vals) for vals in sweep.values()) * N_REPS
    realizados = 0

    filas = []
    for param_variado, valores in sweep.items():
        for valor in valores:
            # Parámetros: defauts + override del parámetro variado
            params = {**defaults, param_variado: valor}

            progreso(realizados, total, prefijo=f"{nombre_red} [{param_variado}={valor}]")

            res = experimento_n_veces(script, params, N_REPS, subprocess_timeout)
            realizados += N_REPS
            progreso(realizados, total, prefijo=f"{nombre_red} [{param_variado}={valor}]")

            if res is None:
                print(f"  ⚠ Sin resultados válidos para {param_variado}={valor}")
                continue

            (pt_media, pt_std), (pte_media, pte_std), (ecm_media, ecm_std) = res

            filas.append({
                "red":               nombre_red,
                "parametro_variado": param_variado,
                "valor_parametro":   valor,
                "precision_train":   round(pt_media, 4),
                "precision_train_std": round(pt_std, 4),
                "precision_test":    round(pte_media, 4),
                "precision_test_std":  round(pte_std, 4),
                "ecm_test":          round(ecm_media, 6),
                "ecm_test_std":      round(ecm_std, 6),
            })

    # Escribir CSV
    campos = [
        "red", "parametro_variado", "valor_parametro",
        "precision_train", "precision_train_std",
        "precision_test", "precision_test_std",
        "ecm_test", "ecm_test_std",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)

    print(f"\n✔ CSV guardado en: {csv_path}  ({len(filas)} filas)\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"  Generando CSVs  (N={N_REPS} repeticiones por experimento)")
    print("=" * 60)

    print("\n▶ Perceptrón")
    barrer(
        script="perceptron.py",
        nombre_red="Perceptron",
        defaults=DEFAULTS_PERC,
        sweep=SWEEP_PERC,
        subprocess_timeout=SUBPROCESS_TIMEOUT_PERC,
        csv_path="resultados_perceptron.csv",
    )

    print("▶ Adaline")
    barrer(
        script="adaline.py",
        nombre_red="Adaline",
        defaults=DEFAULTS_ADALINE,
        sweep=SWEEP_ADALINE,
        subprocess_timeout=SUBPROCESS_TIMEOUT_ADALINE,
        csv_path="resultados_adaline.csv",
    )

    print("✔ Todos los CSVs generados.")
