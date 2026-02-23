from redes_neuronales import Red, Capa, NeuronaEntrada, NeuronaMcCullochPitts
import sys
import csv
import pandas as pd
import numpy as np


def main():
    if len(sys.argv) != 3:
        print("Uso: python mp.py <fichero_entrada> <fichero_salida>")
        sys.exit(1)

    entrada = sys.argv[1]
    salida = sys.argv[2]

    x1 = NeuronaEntrada()
    x2 = NeuronaEntrada()
    z1 = NeuronaMcCullochPitts(2)
    z2 = NeuronaMcCullochPitts(2)
    y1 = NeuronaMcCullochPitts(2)
    y2 = NeuronaMcCullochPitts(2)

    x1.Conectar(y1, 2)
    x2.Conectar(z1, -1)
    x2.Conectar(z2, 2)
    x2.Conectar(y2, 1)
    z2.Conectar(z1, 2)
    z2.Conectar(y2, 1)
    z1.Conectar(y1, 2)

    red = Red()

    capa_entrada = Capa()
    capa_entrada.Añadir(x1)
    capa_entrada.Añadir(x2)

    capa_oculta1 = Capa()
    capa_oculta1.Añadir(z2)

    capa_oculta2 = Capa()
    capa_oculta2.Añadir(z1)

    capa_salida = Capa()
    capa_salida.Añadir(y1)
    capa_salida.Añadir(y2)

    red.Añadir(capa_entrada)
    red.Añadir(capa_oculta1)
    red.Añadir(capa_oculta2)
    red.Añadir(capa_salida)

    red.Inicializar(0)

    patrones = pd.read_csv(entrada, sep=r'\s+',
                           header=None, engine='c').to_numpy()

    resultados = []

    for patron in patrones:
        x1.entrada = patron[0]
        x2.entrada = patron[1]
        red.Disparar()
        resultados.append([x1.valor, x2.valor,
                           z1.valor, z2.valor, y1.valor, y2.valor])
        red.Propagar()

    while True:
        x1.entrada = 0
        x2.entrada = 0
        red.Disparar()
        if z1.valor == 0 and z2.valor == 0 and y1.valor == 0 and y2.valor == 0:
            break
        resultados.append([x1.valor, x2.valor,
                           z1.valor, z2.valor, y1.valor, y2.valor])
        red.Propagar()

    with open(salida, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(["x1", "x2", "z1", "z2", "y1", "y2"])
        writer.writerows(resultados)


if __name__ == '__main__':
    main()
