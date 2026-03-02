import sys
import numpy as np
import signal
from redes_neuronales import leer2, NeuronaPerceptron, NeuronaAdaline, NeuronaEntrada

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, handler)

def main():
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
            for i in range(len(entradas_datos)):
                patron = entradas_datos[i]
                for j, c in enumerate(p.conexiones_entrantes):
                    c.neurona_origen.Inicializar(patron[j])
                    c.neurona_origen.Propagar()
                p.Disparar()
                y = p.valor
                t = salidas_deseadas[i]
                marca = "✓" if y == t else "✗"
                if y != t:
                    p_correcto = False
                print(f"[{patron[0]}, {patron[1]}] → esperado: {t}, obtenido: {y} {marca}")
            
            if p_correcto:
                print("Resultado: CORRECTO\n")
            else:
                print("Resultado: NO CONVERGE\n")
        except TimeoutException:
            msg = "Resultado: NO CONVERGE (tiempo agotado)"
            if nombre == "XOR":
                msg += " (esperado, XOR no es linealmente separable)"
            print(msg + "\n")


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
            for i in range(len(entradas_datos)):
                patron = entradas_datos[i]
                for j, c in enumerate(a.conexiones_entrantes):
                    c.neurona_origen.Inicializar(patron[j])
                    c.neurona_origen.Propagar()
                a.Disparar()
                y = a.valor
                t = salidas_deseadas[i]
                marca = "✓" if y == t else "✗"
                if y != t:
                    a_correcto = False
                print(f"[{patron[0]}, {patron[1]}] → esperado: {t}, obtenido: {y} {marca}")

            if a_correcto:
                print("Resultado: CORRECTO\n")
            else:
                print("Resultado: NO CONVERGE\n")
        except TimeoutException:
            msg = "Resultado: NO CONVERGE (tiempo agotado)"
            if nombre == "XOR":
                msg += " (esperado, XOR no es linealmente separable)"
            print(msg + "\n")

if __name__ == "__main__":
    main()
