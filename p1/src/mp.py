from redes_neuronales import Red, Capa, NeuronaEntrada, NeuronaMcCullochPitts
import sys


def main():
    if len(sys.argv) != 3:
        print("Uso: python mp.py <fichero_entrada> <fichero_salida>")
        sys.exit(1)

    entrada = sys.argv[1]
    salida = sys.argv[2]

    x1 = NeuronaEntrada()
    x2 = NeuronaEntrada()
    z1 = NeuronaMcCullochPitts(1)
    z2 = NeuronaMcCullochPitts(1)
    y1 = NeuronaMcCullochPitts(1)
    y2 = NeuronaMcCullochPitts(1)

    x2.Conectar(z2, 2)
    z2.Conectar(z1, 2)
    z1.Conectar(y1, 2)
    z2.Conectar(y2, 1)
    x2.Conectar(y2, 1)

    red = Red()

    capa_entrada = Capa()
    capa_entrada.Añadir(x1)
    capa_entrada.Añadir(x2)

    capa_oculta = Capa()
    capa_oculta.Añadir(z2)
    capa_oculta.Añadir(z1)

    capa_salida = Capa()
    capa_salida.Añadir(y1)
    capa_salida.Añadir(y2)

    red.Añadir(capa_entrada)
    red.Añadir(capa_oculta)
    red.Añadir(capa_salida)

    red.Inicializar(0)

    with open(entrada, 'r') as f:
        lineas = f.readlines()

    patrones = []
    for linea in lineas:
        linea = linea.strip()
        if linea:
            vals = list(map(int, linea.split()))
            patrones.append(vals)

    resultados = []

    for patron in patrones:
        x1.entrada = patron[0]
        x2.entrada = patron[1]
        red.Disparar()
        resultados.append([int(x1.valor), int(x2.valor), int(
            z1.valor), int(z2.valor), int(y1.valor), int(y2.valor)])
        red.Propagar()

    while True:
        x1.entrada = 0
        x2.entrada = 0
        red.Disparar()

        if z1.valor == 0 and z2.valor == 0 and y1.valor == 0 and y2.valor == 0:
            break

        resultados.append([int(x1.valor), int(x2.valor), int(
            z1.valor), int(z2.valor), int(y1.valor), int(y2.valor)])
        red.Propagar()

    with open(salida, 'w') as f:
        f.write("x1 x2 z1 z2 y1 y2\n")
        for r in resultados:
            f.write(" ".join(map(str, r)) + "\n")


if __name__ == '__main__':
    main()
