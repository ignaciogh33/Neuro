import sys
import os
import csv
import numpy as np
import signal
from redes_neuronales import leer2, NeuronaPerceptron, NeuronaAdaline, NeuronaEntrada

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, handler)

def guardar_frontera_csv(filepath, nombre_problema, red, w1, w2, sesgo, entradas, salidas_deseadas, salidas_obtenidas, converge):
    """Guarda la frontera de decisión y los datos en un CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        # Metadatos de la frontera
        writer.writerow(['problema', 'red', 'w1', 'w2', 'sesgo', 'converge'])
        writer.writerow([nombre_problema, red, f'{w1:.6f}', f'{w2:.6f}', f'{sesgo:.6f}', converge])
        writer.writerow([])  # línea vacía separadora
        # Datos de los patrones
        writer.writerow(['x1', 'x2', 'esperado', 'obtenido'])
        for i in range(len(entradas)):
            x1, x2 = entradas[i][0], entradas[i][1]
            t = salidas_deseadas[i]
            y = salidas_obtenidas[i] if salidas_obtenidas is not None else ''
            writer.writerow([x1, x2, t, y])

def guardar_no_converge_csv(filepath, nombre_problema, red, entradas, salidas_deseadas):
    """Guarda un CSV indicando que no hubo convergencia."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['problema', 'red', 'w1', 'w2', 'sesgo', 'converge'])
        writer.writerow([nombre_problema, red, '', '', '', False])
        writer.writerow([])
        writer.writerow(['x1', 'x2', 'esperado', 'obtenido'])
        for i in range(len(entradas)):
            x1, x2 = entradas[i][0], entradas[i][1]
            t = salidas_deseadas[i]
            writer.writerow([x1, x2, t, ''])

def main():
    # Crear carpeta de salida
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logicos_files')
    os.makedirs(output_dir, exist_ok=True)

    # Hiperparámetros
    TASA_APRENDIZAJE_P = 1.0
    TASA_APRENDIZAJE_A = 0.01
    TOLERANCIA_A = 0.01

    problemas = [
        ("AND", "P1/and.txt"),
        ("OR", "P1/or.txt"),
        ("NAND", "P1/nand.txt"),
        ("XOR", "P1/xor.txt")
    ]

    for nombre, ruta in problemas:
        print(f"=== {nombre} ===")
        entradas_datos, salidas_datos = leer2(ruta)
        salidas_deseadas = salidas_datos.flatten()
        n_entradas = entradas_datos.shape[1]

        print("--- Perceptrón ---")
        p = NeuronaPerceptron(0.2, 0.0)
        neuronas_entrada_p = [NeuronaEntrada() for _ in range(n_entradas)]
        for np_e in neuronas_entrada_p:
            np_e.Conectar(p, 0.0)

        try:
            signal.alarm(2)
            p.Aprender(entradas_datos, salidas_deseadas, TASA_APRENDIZAJE_P)
            signal.alarm(0)

            pesos_p = [c.peso for c in p.conexiones_entrantes]
            print(f"Frontera de decisión: {pesos_p[0]:.4f} * x1 + {pesos_p[1]:.4f} * x2 + ({p.sesgo:.4f}) = 0")

            p_correcto = True
            salidas_obtenidas_p = []
            for i in range(len(entradas_datos)):
                patron = entradas_datos[i]
                for j, c in enumerate(p.conexiones_entrantes):
                    c.neurona_origen.Inicializar(patron[j])
                    c.neurona_origen.Propagar()
                p.Disparar()
                y = p.valor
                t = salidas_deseadas[i]
                salidas_obtenidas_p.append(y)
                marca = "✓" if y == t else "✗"
                if y != t:
                    p_correcto = False
                print(f"[{patron[0]}, {patron[1]}] → esperado: {t}, obtenido: {y} {marca}")
            
            if p_correcto:
                print("Resultado: CORRECTO\n")
            else:
                print("Resultado: NO CONVERGE\n")

            # Guardar frontera de decisión del Perceptrón
            csv_path_p = os.path.join(output_dir, f'{nombre.lower()}_perceptron.csv')
            guardar_frontera_csv(csv_path_p, nombre, 'perceptron',
                                 pesos_p[0], pesos_p[1], p.sesgo,
                                 entradas_datos, salidas_deseadas, salidas_obtenidas_p, p_correcto)
            print(f"  → CSV guardado: {csv_path_p}")

        except TimeoutException:
            signal.alarm(0)
            msg = "Resultado: NO CONVERGE (tiempo agotado)"
            if nombre == "XOR":
                msg += " (esperado, XOR no es linealmente separable)"
            print(msg + "\n")
            # Extraer pesos parciales y predicciones actuales
            pesos_p = [c.peso for c in p.conexiones_entrantes]
            salidas_obtenidas_p = []
            for i in range(len(entradas_datos)):
                patron = entradas_datos[i]
                for j, c in enumerate(p.conexiones_entrantes):
                    c.neurona_origen.Inicializar(patron[j])
                    c.neurona_origen.Propagar()
                p.Disparar()
                salidas_obtenidas_p.append(p.valor)
            csv_path_p = os.path.join(output_dir, f'{nombre.lower()}_perceptron.csv')
            guardar_frontera_csv(csv_path_p, nombre, 'perceptron',
                                 pesos_p[0], pesos_p[1], p.sesgo,
                                 entradas_datos, salidas_deseadas, salidas_obtenidas_p, False)
            print(f"  → CSV guardado: {csv_path_p}")


        # --- Adaline ---
        print("--- Adaline ---")
        a = NeuronaAdaline(0.0)
        neuronas_entrada_a = [NeuronaEntrada() for _ in range(n_entradas)]
        for na_e in neuronas_entrada_a:
            na_e.Conectar(a, 0.0)

        try:
            signal.alarm(2)
            a.Aprender(entradas_datos, salidas_deseadas, TASA_APRENDIZAJE_A, TOLERANCIA_A)
            signal.alarm(0)

            pesos_a = [c.peso for c in a.conexiones_entrantes]
            print(f"Frontera de decisión: {pesos_a[0]:.4f} * x1 + {pesos_a[1]:.4f} * x2 + ({a.sesgo:.4f}) = 0")

            a_correcto = True
            salidas_obtenidas_a = []
            for i in range(len(entradas_datos)):
                patron = entradas_datos[i]
                for j, c in enumerate(a.conexiones_entrantes):
                    c.neurona_origen.Inicializar(patron[j])
                    c.neurona_origen.Propagar()
                a.Disparar()
                y = a.valor
                t = salidas_deseadas[i]
                salidas_obtenidas_a.append(y)
                marca = "✓" if y == t else "✗"
                if y != t:
                    a_correcto = False
                print(f"[{patron[0]}, {patron[1]}] → esperado: {t}, obtenido: {y} {marca}")

            if a_correcto:
                print("Resultado: CORRECTO\n")
            else:
                print("Resultado: NO CONVERGE\n")

            # Guardar frontera de decisión del Adaline
            csv_path_a = os.path.join(output_dir, f'{nombre.lower()}_adaline.csv')
            guardar_frontera_csv(csv_path_a, nombre, 'adaline',
                                 pesos_a[0], pesos_a[1], a.sesgo,
                                 entradas_datos, salidas_deseadas, salidas_obtenidas_a, a_correcto)
            print(f"  → CSV guardado: {csv_path_a}")

        except TimeoutException:
            signal.alarm(0)
            msg = "Resultado: NO CONVERGE (tiempo agotado)"
            if nombre == "XOR":
                msg += " (esperado, XOR no es linealmente separable)"
            print(msg + "\n")
            # Extraer pesos parciales y predicciones actuales
            pesos_a = [c.peso for c in a.conexiones_entrantes]
            salidas_obtenidas_a = []
            for i in range(len(entradas_datos)):
                patron = entradas_datos[i]
                for j, c in enumerate(a.conexiones_entrantes):
                    c.neurona_origen.Inicializar(patron[j])
                    c.neurona_origen.Propagar()
                a.Disparar()
                salidas_obtenidas_a.append(a.valor)
            csv_path_a = os.path.join(output_dir, f'{nombre.lower()}_adaline.csv')
            guardar_frontera_csv(csv_path_a, nombre, 'adaline',
                                 pesos_a[0], pesos_a[1], a.sesgo,
                                 entradas_datos, salidas_deseadas, salidas_obtenidas_a, False)
            print(f"  → CSV guardado: {csv_path_a}")

if __name__ == "__main__":
    main()
