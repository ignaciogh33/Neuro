import random
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Conexion:

    def __init__(self, peso: float, neurona: 'Neurona'):
        self.Crear(peso, neurona)

    def Crear(self, peso: float, neurona: 'Neurona'):
        self.peso = peso
        self.neurona_destino = neurona
        self.neurona_origen = None

    def Liberar(self):
        self.neurona_destino = None
        self.neurona_origen = None


class Neurona(ABC):

    def __init__(self, *args, **kwargs):
        self.Crear(*args, **kwargs)

    def Crear(self, *args, **kwargs):
        self.conexiones = []
        self.conexiones_entrantes = []
        self.entrada = 0
        self.valor = 0

    def Liberar(self):
        for c in self.conexiones:
            c.Liberar()
        self.conexiones.clear()
        self.conexiones_entrantes.clear()

    def Conectar(self, neurona: 'Neurona', peso: float):
        conexion = Conexion(peso, neurona)
        conexion.neurona_origen = self
        self.conexiones.append(conexion)
        neurona.conexiones_entrantes.append(conexion)

    def Inicializar(self, valor: float):
        self.valor = valor
        self.entrada = 0

    @abstractmethod
    def Disparar(self):
        pass

    def Propagar(self):
        for c in self.conexiones:
            c.neurona_destino.entrada += self.valor * c.peso


class Capa:

    def __init__(self):
        self.Crear()

    def Crear(self):
        self.neuronas = []

    def Liberar(self):
        for n in self.neuronas:
            n.Liberar()
        self.neuronas.clear()

    def Añadir(self, neurona: Neurona):
        self.neuronas.append(neurona)

    def Conectar(self, destino, peso_min: float, peso_max: float):
        if isinstance(destino, Capa):
            for n_origen in self.neuronas:
                for n_destino in destino.neuronas:
                    peso = random.uniform(peso_min, peso_max)
                    n_origen.Conectar(n_destino, peso)
        elif isinstance(destino, Neurona):
            for n_origen in self.neuronas:
                peso = random.uniform(peso_min, peso_max)
                n_origen.Conectar(destino, peso)

    def Inicializar(self, valor: float):
        for n in self.neuronas:
            n.Inicializar(valor)

    def Disparar(self):
        for n in self.neuronas:
            n.Disparar()

    def Propagar(self):
        for n in self.neuronas:
            n.Propagar()


class Red:

    def __init__(self):
        self.Crear()

    def Crear(self):
        self.capas = []

    def Liberar(self):
        for capa in self.capas:
            capa.Liberar()
        self.capas.clear()

    def Añadir(self, capa: Capa):
        self.capas.append(capa)

    def Inicializar(self, valor: float):
        for capa in self.capas:
            capa.Inicializar(valor)

    def Disparar(self):
        for capa in self.capas:
            capa.Disparar()

    def Propagar(self):
        for capa in self.capas:
            capa.Propagar()


class NeuronaEntrada(Neurona):

    def Crear(self):
        super().Crear()

    def Disparar(self):
        self.valor = self.entrada
        self.entrada = 0


class NeuronaMcCullochPitts(Neurona):

    def Crear(self, umbral: float):
        super().Crear()
        self.umbral = umbral

    def Disparar(self):
        inhibida = False

        for c in self.conexiones_entrantes:
            if c.peso < 0 and c.neurona_origen.valor > 0:
                inhibida = True
                break

        if inhibida:
            self.valor = 0
        elif self.entrada >= self.umbral:
            self.valor = 1
        else:
            self.valor = 0

        self.entrada = 0.0


class NeuronaPerceptron(Neurona):

    def Crear(self, umbral: float, sesgo: float):
        super().Crear()
        self.umbral = umbral
        self.sesgo = sesgo

    def Disparar(self):
        y_in = self.entrada + self.sesgo

        if y_in > self.umbral:
            self.valor = 1
        elif -self.umbral <= y_in:
            self.valor = 0
        else:
            self.valor = -1

        self.entrada = 0.0

    def Aprender(self, entradas: np.ndarray, salidas_deseadas: np.ndarray, alfa: float):
        self.sesgo = 0.0
        for c in self.conexiones_entrantes:
            c.peso = 0.0

        while True:
            hubo_cambios = False

            for p in range(len(entradas)):
                patron = entradas[p]

                for i, c in enumerate(self.conexiones_entrantes):
                    c.neurona_origen.Inicializar(patron[i])
                    c.neurona_origen.Propagar()

                self.Disparar()
                y = self.valor

                t = salidas_deseadas[p]
                if y != t:
                    if alfa == 0 or t == 0:
                        break
                    hubo_cambios = True
                    alfa_t = alfa * t
                    self.sesgo += alfa_t
                    for c in self.conexiones_entrantes:
                        c.peso += alfa_t * c.neurona_origen.valor

            if not hubo_cambios:
                break


class NeuronaAdaline(Neurona):

    def Crear(self, sesgo: float):
        super().Crear()
        self.sesgo = sesgo
        self.en_entrenamiento = True

    def Disparar(self):
        y_in = self.entrada + self.sesgo

        if self.en_entrenamiento:
            self.valor = y_in
        else:
            self.valor = 1 if y_in >= 0 else -1

        self.entrada = 0.0

    def Aprender(self, entradas: np.ndarray, salidas_deseadas: np.ndarray, alfa: float, tolerancia: float):
        self.sesgo = random.uniform(-0.5, 0.5)
        for c in self.conexiones_entrantes:
            c.peso = random.uniform(-0.5, 0.5)

        self.en_entrenamiento = True

        while True:
            cambio_max_epoca = 0.0

            for p in range(len(entradas)):
                patron = entradas[p]

                for i, c in enumerate(self.conexiones_entrantes):
                    c.neurona_origen.Inicializar(patron[i])
                    c.neurona_origen.Propagar()

                self.Disparar()
                y_in = self.valor

                t = salidas_deseadas[p]
                error = t - y_in

                alfa_error = alfa * error
                self.sesgo += alfa_error
                abs_delta_b = abs(alfa_error)
                if abs_delta_b > cambio_max_epoca:
                    cambio_max_epoca = abs_delta_b

                for c in self.conexiones_entrantes:
                    delta_w = alfa_error * c.neurona_origen.valor
                    c.peso += delta_w
                    abs_delta_w = abs(delta_w)
                    if abs_delta_w > cambio_max_epoca:
                        cambio_max_epoca = abs_delta_w

            if cambio_max_epoca < tolerancia:
                break

        self.en_entrenamiento = False


def parsear_fichero(fichero: str):
    with open(fichero, 'r') as f:
        m, n = map(int, f.readline().split())
        datos = pd.read_csv(f, sep=r'\s+', header=None, engine='c').to_numpy()

    entradas = datos[:, :m]
    salidas = datos[:, m:m+n].astype(np.int32)

    return entradas, salidas


def leer1(fichero_de_datos: str, por: float):
    entradas, salidas = parsear_fichero(fichero_de_datos)

    indices = np.random.permutation(len(entradas))
    entradas = entradas[indices]
    salidas = salidas[indices]

    corte = int(len(entradas) * por)

    return entradas[:corte], salidas[:corte], entradas[corte:], salidas[corte:]


def leer2(fichero_de_datos: str):
    entradas_datos, salidas_datos = parsear_fichero(fichero_de_datos)
    return entradas_datos, salidas_datos


def leer3(fichero_de_entrenamiento: str, fichero_de_test: str):
    entradas_entrenamiento, salidas_entrenamiento = parsear_fichero(
        fichero_de_entrenamiento)
    entradas_test, salidas_test = parsear_fichero(fichero_de_test)

    return entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test
