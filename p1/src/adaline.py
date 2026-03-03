import argparse
import signal
import numpy as np
from redes_neuronales import Red, Capa, leer1, NeuronaAdaline, NeuronaEntrada


class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, handler)


def main():
    parser = argparse.ArgumentParser(description="Adaline - Problema Real 1")
    parser.add_argument("--fichero", type=str, default="P1/problema_real1.txt")
    parser.add_argument("--porcentaje", type=float, default=0.7)
    parser.add_argument("--alfa", type=float, default=0.01)
    parser.add_argument("--tolerancia", type=float, default=0.01)
    parser.add_argument("--sesgo", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()

    ent_train, sal_train, ent_test, sal_test = leer1(args.fichero, args.porcentaje)
    n_clases = sal_train.shape[1]
    n_entradas = ent_train.shape[1]

    print(f"Entrenamiento: {len(ent_train)} patrones | Test: {len(ent_test)} patrones")
    print(f"Atributos: {n_entradas} | Clases: {n_clases}")
    print(f"Alfa: {args.alfa} | Tolerancia: {args.tolerancia} | Sesgo: {args.sesgo}")
    print()

    # 1. Crear neuronas
    neuronas_entrada = [NeuronaEntrada() for _ in range(n_entradas)]
    neuronas = [NeuronaAdaline(args.sesgo) for _ in range(n_clases)]

    # 2. Conectar: cada entrada → todas las salidas (fully connected)
    for ne in neuronas_entrada:
        for a in neuronas:
            ne.Conectar(a, np.random.uniform(-0.5, 0.5))

    # 3. Crear red y capas
    red = Red()

    capa_entrada = Capa()
    for ne in neuronas_entrada:
        capa_entrada.Añadir(ne)

    capa_salida = Capa()
    for a in neuronas:
        capa_salida.Añadir(a)

    red.Añadir(capa_entrada)
    red.Añadir(capa_salida)

    red.Inicializar(0)

    # 4. Entrenamiento
    for c in range(n_clases):
        a = neuronas[c]
        salidas_clase = sal_train[:, c]

        try:
            signal.alarm(args.timeout)
            a.Aprender(ent_train, salidas_clase, args.alfa, args.tolerancia)
            signal.alarm(0)
            print(f"Clase {c+1}: Convergencia alcanzada")
        except TimeoutException:
            a.en_entrenamiento = False  # garantizar modo binario en evaluación
            print(f"Clase {c+1}: No converge (timeout {args.timeout}s)")

        pesos = [con.peso for con in a.conexiones_entrantes]
        terms = " + ".join([f"{pesos[i]:.4f} * x{i+1}" for i in range(len(pesos))])
        print(f"  Frontera: {terms} + ({a.sesgo:.4f}) = 0")

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

    # 6. Evaluación test
    if len(ent_test) > 0:
        aciertos_test = 0
        ecm_total = 0.0
        for idx in range(len(ent_test)):
            patron = ent_test[idx]
            for i, ne in enumerate(neuronas_entrada):
                ne.Inicializar(patron[i])
            red.Propagar()
            red.Disparar()
            correcto = True
            for c in range(n_clases):
                error = sal_test[idx, c] - neuronas[c].valor
                ecm_total += error ** 2
                if neuronas[c].valor != sal_test[idx, c]:
                    correcto = False
            if correcto:
                aciertos_test += 1
        ecm = ecm_total / (len(ent_test) * n_clases)
        print(f"Precision test: {aciertos_test / len(ent_test) * 100:.2f}%")
        print(f"ECM test: {ecm:.6f}")


if __name__ == "__main__":
    main()
