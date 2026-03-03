"""
mp.py - Programa principal que construye y ejecuta una red neuronal
McCulloch-Pitts de 4 capas para procesar patrones desde un fichero.

Arquitectura de la red:
    Capa entrada:  x1, x2       (neuronas de entrada)
    Capa oculta 1: z2           (McCulloch-Pitts, umbral=2)
    Capa oculta 2: z1           (McCulloch-Pitts, umbral=2)
    Capa salida:   y1, y2       (McCulloch-Pitts, umbral=2)

Conexiones (neurona_origen -> neurona_destino, peso):
    x1 -> y1 (peso=2)
    x2 -> z1 (peso=-1, inhibidora)
    x2 -> z2 (peso=2)
    x2 -> y2 (peso=1)
    z2 -> z1 (peso=2)
    z2 -> y2 (peso=1)
    z1 -> y1 (peso=2)
"""

from redes_neuronales import Red, Capa, NeuronaEntrada, NeuronaMcCullochPitts
import sys
import csv
import pandas as pd
import numpy as np


def main():
    """
    Función principal que:
    1. Lee los argumentos (fichero entrada y salida).
    2. Crea las neuronas y establece las conexiones.
    3. Organiza las neuronas en capas y forma la red.
    4. Procesa cada patrón de entrada propagando y disparando la red.
    5. Ejecuta ciclos adicionales para vaciar la red tras el último patrón.
    6. Escribe los estados de todas las neuronas en el fichero de salida.
    """
    if len(sys.argv) != 3:
        print("Uso: python mp.py <fichero_entrada> <fichero_salida>")
        sys.exit(1)

    entrada = sys.argv[1]  # ruta al fichero con los patrones de entrada
    salida = sys.argv[2]   # ruta al fichero donde se escribirán los resultados

    # Neuronas de entrada: transfieren su entrada directamente como salida
    x1 = NeuronaEntrada()
    x2 = NeuronaEntrada()

    # Neuronas McCulloch-Pitts con umbral 2 para capas ocultas y salida
    z1 = NeuronaMcCullochPitts(2)
    z2 = NeuronaMcCullochPitts(2)
    y1 = NeuronaMcCullochPitts(2)
    y2 = NeuronaMcCullochPitts(2)

    # Establecimiento de las conexiones entre neuronas
    x1.Conectar(y1, 2)    # x1 -> y1, peso 2 (excitadora directa)
    x2.Conectar(z1, -1)   # x2 -> z1, peso -1 (inhibidora)
    x2.Conectar(z2, 2)    # x2 -> z2, peso 2 (excitadora)
    x2.Conectar(y2, 1)    # x2 -> y2, peso 1 (excitadora)
    z2.Conectar(z1, 2)    # z2 -> z1, peso 2 (excitadora)
    z2.Conectar(y2, 1)    # z2 -> y2, peso 1 (excitadora)
    z1.Conectar(y1, 2)    # z1 -> y1, peso 2 (excitadora)

    # Construcción de la red: se organizan las neuronas en capas
    red = Red()

    capa_entrada = Capa()  # capa de entrada con x1 y x2
    capa_entrada.Añadir(x1)
    capa_entrada.Añadir(x2)

    capa_oculta1 = Capa()  # primera capa oculta: z2 se procesa antes que z1
    capa_oculta1.Añadir(z2)

    capa_oculta2 = Capa()  # segunda capa oculta: z1 depende de z2
    capa_oculta2.Añadir(z1)

    capa_salida = Capa()  # capa de salida con y1 e y2
    capa_salida.Añadir(y1)
    capa_salida.Añadir(y2)

    # Añade las capas en orden de procesamiento (entrada -> ocultas -> salida)
    red.Añadir(capa_entrada)
    red.Añadir(capa_oculta1)
    red.Añadir(capa_oculta2)
    red.Añadir(capa_salida)

    red.Inicializar(0)  # inicializa todas las neuronas a 0

    # Lectura de patrones: fichero CSV separado por espacios, cada fila = un patrón
    patrones = pd.read_csv(entrada, sep=r'\s+',
                           header=None, engine='c').to_numpy()

    resultados = []  # almacena los estados de la red en cada ciclo

    # Procesamiento de cada patrón
    for patron in patrones:
        x1.Inicializar(patron[0])  # carga valor de x1
        x2.Inicializar(patron[1])  # carga valor de x2

        # Registra el estado ACTUAL (antes de propagar/disparar)
        resultados.append([x1.valor, x2.valor,
                           z1.valor, z2.valor, y1.valor, y2.valor])

        red.Propagar()  # propaga señales por las conexiones
        red.Disparar()  # cada neurona calcula su salida

    # Propagación residual: (nº_capas - 1) ciclos adicionales para que
    # la señal del último patrón recorra toda la red hasta la salida
    for _ in range(len(red.capas) - 1):
        resultados.append([x1.valor, x2.valor,
                           z1.valor, z2.valor, y1.valor, y2.valor])
        red.Propagar()
        red.Disparar()

    # Escritura de resultados en el fichero de salida
    with open(salida, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(["x1", "x2", "z1", "z2", "y1", "y2"])  # cabecera
        writer.writerows(resultados)  # un estado por fila


if __name__ == '__main__':
    main()
