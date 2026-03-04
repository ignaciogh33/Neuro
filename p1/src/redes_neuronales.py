import random
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Conexion:
    """
    Representa una conexión (sinapsis) entre dos neuronas.
    Cada conexión tiene un peso asociado y conoce su neurona de origen y destino.
    """

    def __init__(self, peso: float, neurona: 'Neurona'):
        self.Crear(peso, neurona)

    def Crear(self, peso: float, neurona: 'Neurona'):
        """
        Inicializa la conexión con el peso sináptico y la neurona destino.
        La neurona de origen se asigna después, al establecer la conexión.
        """
        self.peso = peso
        self.neurona_destino = neurona
        self.neurona_origen = None

    def Liberar(self):
        """Elimina las referencias a las neuronas para liberar recursos."""
        self.neurona_destino = None
        self.neurona_origen = None


class Neurona(ABC):
    """
    Clase base abstracta para todos los tipos de neuronas.
    Define la interfaz común: conectar, inicializar, disparar y propagar.
    Atributos principales:
        - conexiones: lista de conexiones salientes (hacia otras neuronas).
        - conexiones_entrantes: lista de conexiones entrantes (desde otras neuronas).
        - entrada: acumulador de la suma ponderada de señales recibidas.
        - valor: valor de salida actual de la neurona.
    """

    def __init__(self, *args, **kwargs):
        self.Crear(*args, **kwargs)

    def Crear(self, *args, **kwargs):
        """Inicializa las listas de conexiones y los valores de entrada/salida a 0."""
        self.conexiones = []
        self.conexiones_entrantes = []
        self.entrada = 0
        self.valor = 0

    def Liberar(self):
        """Libera todas las conexiones salientes y limpia ambas listas."""
        for c in self.conexiones:
            c.Liberar()
        self.conexiones.clear()
        self.conexiones_entrantes.clear()

    def Conectar(self, neurona: 'Neurona', peso: float):
        """
        Crea una conexión desde esta neurona hacia otra con un peso dado.
        La conexión se registra tanto en la lista de salientes de esta neurona
        como en la lista de entrantes de la neurona destino.
        """
        conexion = Conexion(peso, neurona)
        conexion.neurona_origen = self
        self.conexiones.append(conexion)
        neurona.conexiones_entrantes.append(conexion)

    def Inicializar(self, valor: float):
        """Establece el valor de salida de la neurona y reinicia la entrada a 0."""
        self.valor = valor
        self.entrada = 0

    @abstractmethod
    def Disparar(self):
        """
        Método abstracto: cada tipo de neurona define cómo calcula su salida
        a partir de la entrada acumulada.
        """
        pass

    def Propagar(self):
        """
        Envía la señal de salida (self.valor) a todas las neuronas conectadas,
        multiplicándola por el peso de cada conexión.
        El resultado se suma al acumulador 'entrada' de la neurona destino.
        """
        for c in self.conexiones:
            c.neurona_destino.entrada += self.valor * c.peso


class Capa:
    """
    Agrupa un conjunto de neuronas que operan en paralelo.
    Permite disparar y propagar todas las neuronas de la capa a la vez.
    """

    def __init__(self):
        self.Crear()

    def Crear(self):
        """Inicializa la lista de neuronas vacía."""
        self.neuronas = []

    def Liberar(self):
        """Libera todas las neuronas de la capa."""
        for n in self.neuronas:
            n.Liberar()
        self.neuronas.clear()

    def Añadir(self, neurona: Neurona):
        """Añade una neurona a la capa."""
        self.neuronas.append(neurona)

    def Conectar(self, destino, peso_min: float, peso_max: float):
        """
        Conecta todas las neuronas de esta capa con el destino.
        El destino puede ser:
            - Otra Capa: conexión 'todos con todos' (fully connected).
            - Una Neurona individual.
        Los pesos se asignan aleatoriamente en el rango [peso_min, peso_max].
        """
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
        """Inicializa todas las neuronas de la capa con el valor dado."""
        for n in self.neuronas:
            n.Inicializar(valor)

    def Disparar(self):
        """Dispara (calcula la salida de) todas las neuronas de la capa."""
        for n in self.neuronas:
            n.Disparar()

    def Propagar(self):
        """Propaga la señal de salida de todas las neuronas de la capa."""
        for n in self.neuronas:
            n.Propagar()


class Red:
    """
    Representa una red neuronal completa formada por capas.
    Gestiona el ciclo Inicializar -> Disparar -> Propagar sobre todas las capas.
    Las capas se almacenan en orden: entrada -> ocultas -> salida.
    """

    def __init__(self):
        self.Crear()

    def Crear(self):
        """Inicializa la lista ordenada de capas vacía."""
        self.capas = []

    def Liberar(self):
        """Libera todas las capas y sus neuronas."""
        for capa in self.capas:
            capa.Liberar()
        self.capas.clear()

    def Añadir(self, capa: Capa):
        """Añade una capa a la red (el orden de adición importa)."""
        self.capas.append(capa)

    def Inicializar(self, valor: float):
        """Inicializa todas las neuronas de todas las capas."""
        for capa in self.capas:
            capa.Inicializar(valor)

    def Disparar(self):
        """Dispara todas las capas de la red secuencialmente."""
        for capa in self.capas:
            capa.Disparar()

    def Propagar(self):
        """Propaga las señales de todas las capas de la red secuencialmente."""
        for capa in self.capas:
            capa.Propagar()


class NeuronaEntrada(Neurona):
    """
    Neurona especial de la capa de entrada.
    Simplemente transfiere su entrada directamente como valor de salida.
    No realiza ningún cálculo adicional.
    """

    def Disparar(self):
        """Copia la entrada acumulada como valor de salida y reinicia la entrada."""
        self.valor = self.entrada
        self.entrada = 0


class NeuronaMcCullochPitts(Neurona):
    """
    Modelo clásico de neurona binaria McCulloch-Pitts con umbral.
    Características:
        - Salida binaria: 0 o 1.
        - Se activa (salida=1) si la entrada excitadora >= umbral.
        - Se inhibe (salida=0) si alguna conexión inhibidora (peso negativo)
          proviene de una neurona activa (valor > 0).
    """

    def Crear(self, umbral: float):
        """Inicializa la neurona con el umbral de activación dado."""
        super().Crear()
        self.umbral = umbral

    def Disparar(self):
        """
        Calcula la salida de la neurona:
        1. Comprueba si hay alguna conexión inhibidora activa.
        2. Si está inhibida, la salida es 0.
        3. Si no está inhibida y la entrada >= umbral, la salida es 1.
        4. En caso contrario, la salida es 0.
        """
        inhibida = False

        for c in self.conexiones_entrantes:
            if c.peso < 0 and c.neurona_origen.valor > 0:  # conexión inhibidora activa
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
    """
    Modelo de perceptrón con aprendizaje supervisado.
    Características:
        - Salida ternaria: -1, 0 o 1.
        - Función de activación con zona muerta alrededor de 0
          (entre -umbral y +umbral).
        - Incluye un sesgo (bias) que se suma a la entrada.
        - Método Aprender implementa la regla de aprendizaje del perceptrón.
    """

    def Crear(self, umbral: float, sesgo: float):
        """
        Inicializa la neurona con un umbral para la función de activación
        y un sesgo (bias) que se suma siempre a la entrada neta.
        """
        super().Crear()
        self.umbral = umbral
        self.sesgo = sesgo

    def Disparar(self):
        """
        Calcula la entrada neta (suma ponderada + sesgo) y aplica la función
        de activación con tres regiones:
            - y_in > umbral      -> salida = 1  (activación positiva)
            - -umbral <= y_in    -> salida = 0  (zona muerta)
            - y_in < -umbral     -> salida = -1 (activación negativa)
        """
        y_in = self.entrada + self.sesgo

        if y_in > self.umbral:
            self.valor = 1
        elif -self.umbral <= y_in:
            self.valor = 0
        else:
            self.valor = -1

        self.entrada = 0.0

    def Aprender(self, entradas: np.ndarray, salidas_deseadas: np.ndarray, alfa: float):
        """
        Entrena la neurona con la regla de aprendizaje del perceptrón.
        Inicializa pesos y sesgo a 0, luego itera sobre los patrones
        ajustando los pesos cuando la salida no coincide con la deseada:
            sesgo += alfa * t
            peso  += alfa * t * entrada_correspondiente
        Se repite hasta que no haya cambios en una época completa (convergencia).
        """
        self.sesgo = 0.0
        for c in self.conexiones_entrantes:
            c.peso = 0.0

        while True:
            hubo_cambios = False

            for p in range(len(entradas)):
                patron = entradas[p]

                # Carga el patrón en las neuronas de entrada y propaga
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
                    self.sesgo += alfa_t  # actualiza sesgo
                    for c in self.conexiones_entrantes:
                        c.peso += alfa_t * c.neurona_origen.valor  # actualiza pesos

            if not hubo_cambios:
                break


class NeuronaAdaline(Neurona):
    """
    Modelo ADALINE (ADAptive LINear Element).
    Características:
        - Durante el entrenamiento: salida lineal (y_in directo).
        - En uso (test): salida bipolar (-1 o 1) según signo de y_in.
        - Incluye sesgo (bias).
        - Método Aprender implementa la regla delta (LMS - Least Mean Squares).
        - Converge cuando el máximo cambio en pesos/sesgo < tolerancia.
    """

    def Crear(self, sesgo: float):
        """
        Inicializa la neurona con un sesgo dado.
        en_entrenamiento controla si la salida es lineal (True) o bipolar (False).
        """
        super().Crear()
        self.sesgo = sesgo
        self.en_entrenamiento = True

    def Disparar(self):
        """
        Calcula la entrada neta (suma ponderada + sesgo).
        - En entrenamiento: la salida es el valor lineal y_in,
          lo que permite calcular el error continuo para la regla delta.
        - En test: aplica función de activación bipolar (1 si y_in >= 0, -1 si no).
        """
        y_in = self.entrada + self.sesgo

        if self.en_entrenamiento:
            self.valor = y_in
        else:
            self.valor = 1 if y_in >= 0 else -1

        self.entrada = 0.0

    def Aprender(self, entradas: np.ndarray, salidas_deseadas: np.ndarray, alfa: float, tolerancia: float):
        """
        Entrena la neurona con la regla delta (LMS).
        1. Inicializa sesgo y pesos con valores aleatorios en [-0.5, 0.5].
        2. Para cada patrón calcula el error (t - y_in) y actualiza:
               sesgo += alfa * error
               peso  += alfa * error * entrada_correspondiente
        3. Registra el mayor cambio absoluto en la época.
        4. Se detiene cuando el mayor cambio < tolerancia (convergencia).
        5. Desactiva el modo entrenamiento al finalizar.
        """
        self.sesgo = random.uniform(-0.5, 0.5)
        for c in self.conexiones_entrantes:
            c.peso = random.uniform(-0.5, 0.5)

        self.en_entrenamiento = True

        while True:
            cambio_max_epoca = 0.0  # mayor cambio absoluto en esta época

            for p in range(len(entradas)):
                patron = entradas[p]

                # Carga el patrón en las neuronas de entrada y propaga
                for i, c in enumerate(self.conexiones_entrantes):
                    c.neurona_origen.Inicializar(patron[i])
                    c.neurona_origen.Propagar()

                self.Disparar()
                y_in = self.valor

                t = salidas_deseadas[p]
                error = t - y_in  # error entre salida deseada y obtenida

                alfa_error = alfa * error
                self.sesgo += alfa_error  # actualiza sesgo con regla delta
                abs_delta_b = abs(alfa_error)
                if abs_delta_b > cambio_max_epoca:
                    cambio_max_epoca = abs_delta_b

                for c in self.conexiones_entrantes:
                    delta_w = alfa_error * c.neurona_origen.valor  # actualiza peso
                    c.peso += delta_w
                    abs_delta_w = abs(delta_w)
                    if abs_delta_w > cambio_max_epoca:
                        cambio_max_epoca = abs_delta_w

            if cambio_max_epoca < tolerancia:  # criterio de convergencia
                break

        self.en_entrenamiento = False


def parsear_fichero(fichero: str):
    """
    Lee un fichero de datos con el formato:
        - Primera línea: m n (número de entradas y número de salidas).
        - Resto de líneas: valores separados por espacios (m entradas + n salidas).
    Devuelve dos arrays NumPy: entradas (float) y salidas (int32).
    """
    with open(fichero, 'r') as f:
        # lee dimensiones m (entradas) y n (salidas)
        m, n = map(int, f.readline().split())
        datos = pd.read_csv(f, sep=r'\s+', header=None, engine='c').to_numpy()

    entradas = datos[:, :m]                    # primeras m columnas
    salidas = datos[:, m:m+n].astype(np.int32)  # siguientes n columnas

    return entradas, salidas


def leer1(fichero_de_datos: str, por: float):
    """
    Modo de lectura 1: Lee un único fichero y lo divide en dos conjuntos
    tras barajar aleatoriamente:
        - Entrenamiento: primeros por*100 % de los patrones.
        - Test: el resto.
    Devuelve: entradas_train, salidas_train, entradas_test, salidas_test.
    """
    entradas, salidas = parsear_fichero(fichero_de_datos)

    # baraja para evitar sesgos de orden
    indices = np.random.permutation(len(entradas))
    entradas = entradas[indices]
    salidas = salidas[indices]

    corte = int(len(entradas) * por)  # punto de corte según el porcentaje

    return entradas[:corte], salidas[:corte], entradas[corte:], salidas[corte:]


def leer2(fichero_de_datos: str):
    """
    Modo de lectura 2: Lee un único fichero y devuelve todos los datos
    sin dividir (se usan tanto para entrenamiento como para test).
    Devuelve: entradas, salidas.
    """
    entradas_datos, salidas_datos = parsear_fichero(fichero_de_datos)
    return entradas_datos, salidas_datos


def leer3(fichero_de_entrenamiento: str, fichero_de_test: str):
    """
    Modo de lectura 3: Lee dos ficheros separados, uno para entrenamiento
    y otro para test.
    Devuelve: entradas_train, salidas_train, entradas_test, salidas_test.
    """
    entradas_entrenamiento, salidas_entrenamiento = parsear_fichero(
        fichero_de_entrenamiento)
    entradas_test, salidas_test = parsear_fichero(fichero_de_test)

    return entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test
