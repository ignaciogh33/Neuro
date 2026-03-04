import argparse
import os
import signal
import numpy as np
from redes_neuronales import Red, Capa, leer3, NeuronaPerceptron, NeuronaEntrada


class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, handler)


def main():
    parser = argparse.ArgumentParser(description="Perceptron - Problema Real 2 (Modo 3)")
    parser.add_argument("--fichero_train", type=str, default="P1/problema_real2.txt")
    parser.add_argument("--fichero_test", type=str, default="P1/problema_real2_no_etiquetados.txt")
    parser.add_argument("--alfa", type=float, default=0.3)
    parser.add_argument("--umbral", type=float, default=0.1)
    parser.add_argument("--sesgo", type=float, default=0.5)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--salida", type=str, default="predicciones/prediccion_perceptron.txt")
    args = parser.parse_args()

    # Modo 3: fichero de entrenamiento + fichero de test separados
    ent_train, sal_train, ent_test, sal_test = leer3(args.fichero_train, args.fichero_test)
    n_clases = sal_train.shape[1]
    n_entradas = ent_train.shape[1]

    print(f"Entrenamiento: {len(ent_train)} patrones | Test (no etiquetados): {len(ent_test)} patrones")
    print(f"Atributos: {n_entradas} | Clases: {n_clases}")
    print(f"Alfa: {args.alfa} | Umbral: {args.umbral} | Sesgo: {args.sesgo}")
    print()

    # 1. Crear neuronas
    neuronas_entrada = [NeuronaEntrada() for _ in range(n_entradas)]
    neuronas = [NeuronaPerceptron(args.umbral, args.sesgo) for _ in range(n_clases)]

    # 2. Conectar: cada entrada → todas las salidas (fully connected)
    for ne in neuronas_entrada:
        for p in neuronas:
            ne.Conectar(p, 0.0)

    # 3. Crear red y capas
    red = Red()

    capa_entrada = Capa()
    for ne in neuronas_entrada:
        capa_entrada.Añadir(ne)

    capa_salida = Capa()
    for p in neuronas:
        capa_salida.Añadir(p)

    red.Añadir(capa_entrada)
    red.Añadir(capa_salida)

    red.Inicializar(0)

    # 4. Entrenamiento con TODOS los datos etiquetados
    for c in range(n_clases):
        p = neuronas[c]
        salidas_clase = sal_train[:, c]

        try:
            signal.alarm(args.timeout)
            p.Aprender(ent_train, salidas_clase, args.alfa)
            signal.alarm(0)
            print(f"Clase {c+1}: Convergencia alcanzada")
        except TimeoutException:
            print(f"Clase {c+1}: No converge (timeout {args.timeout}s)")

        pesos = [con.peso for con in p.conexiones_entrantes]
        terms = " + ".join([f"{pesos[i]:.4f} * x{i+1}" for i in range(len(pesos))])
        print(f"  Frontera: {terms} + ({p.sesgo:.4f}) = 0")

    print()

    # 5. Evaluación entrenamiento
    aciertos_train = 0
    for idx in range(len(ent_train)):
        patron = ent_train[idx]
        for i, ne in enumerate(neuronas_entrada):
            ne.Inicializar(patron[i])
        red.Propagar()
        red.Disparar()
        correcto = True
        for c in range(n_clases):
            if neuronas[c].valor != sal_train[idx, c]:
                correcto = False
        if correcto:
            aciertos_train += 1
    print(f"Precision entrenamiento: {aciertos_train / len(ent_train) * 100:.2f}%")
    print()

    # 6. Predicción sobre los datos NO etiquetados
    os.makedirs(os.path.dirname(args.salida), exist_ok=True)

    predicciones = []
    for idx in range(len(ent_test)):
        patron = ent_test[idx]
        for i, ne in enumerate(neuronas_entrada):
            ne.Inicializar(patron[i])
        red.Propagar()
        red.Disparar()
        pred = [int(neuronas[c].valor) for c in range(n_clases)]
        predicciones.append(pred)

    with open(args.salida, 'w') as f:
        for pred in predicciones:
            f.write(" ".join(map(str, pred)) + "\n")

    print(f"Predicciones guardadas en: {args.salida}")
    print(f"Total predicciones: {len(predicciones)}")

    # Resumen de predicciones
    preds_array = np.array(predicciones)
    for c in range(n_clases):
        conteo_1 = np.sum(preds_array[:, c] == 1)
        conteo_m1 = np.sum(preds_array[:, c] == -1)
        conteo_0 = np.sum(preds_array[:, c] == 0)
        print(f"  Clase {c+1}: {conteo_1} (+1), {conteo_m1} (-1), {conteo_0} (0)")


if __name__ == "__main__":
    main()
