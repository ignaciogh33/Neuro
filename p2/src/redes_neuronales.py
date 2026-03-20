import math
import random
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def sigmoide_bipolar(x: float) -> float:
    """Función de activación sigmoide bipolar. Devuelve valores en el rango (-1, 1)."""
    return 2.0 / (1.0 + math.exp(-x)) - 1.0


def derivada_sigmoide_bipolar(fx: float) -> float:
    """Derivada de la sigmoide bipolar expresada en función de su propia salida fx.
    Permite calcular el gradiente de forma eficiente sin recalcular la exponencial.
    """
    return 0.5 * (1.0 + fx) * (1.0 - fx)


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

    def CalcularSalida(self, patron: np.ndarray) -> np.ndarray:
        """
        Realiza el paso feedforward completo para un patrón de entrada.
        1. Inicializa cada neurona de la capa de entrada con el valor del patrón.
        2. Propaga y dispara capa a capa hasta la capa de salida.
        Devuelve un array NumPy con los valores de las neuronas de salida.
        """
        for i, neurona in enumerate(self.capas[0].neuronas):
            neurona.Inicializar(patron[i])
        for c in range(len(self.capas) - 1):
            self.capas[c].Propagar()
            self.capas[c + 1].Disparar()
        return np.array([n.valor for n in self.capas[-1].neuronas])


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


class NeuronaSigmoide(Neurona):
    """
    Neurona con función de activación sigmoide bipolar.
    Produce salidas continuas en el rango (-1, 1).
    Se utiliza en capas ocultas y de salida del MLP con retropropagación.
    """

    def Crear(self, sesgo: float):
        """Inicializa la neurona con el sesgo (bias) dado."""
        super().Crear()
        self.sesgo = sesgo

    def Disparar(self):
        """
        Calcula la entrada neta (suma ponderada + sesgo) y aplica la función
        sigmoide bipolar para obtener la salida en el rango (-1, 1).
        Reinicia el acumulador de entrada tras disparar.
        """
        self.valor = sigmoide_bipolar(self.entrada + self.sesgo)
        self.entrada = 0.0


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


def _evaluar_conjunto(red: Red, entradas: np.ndarray,
                      salidas_deseadas: np.ndarray,
                      calcular_matriz: bool = False) -> tuple:
    """
    Evalúa la red sobre un conjunto de patrones en un único recorrido y calcula:
        - ECM (Error Cuadrático Medio).
        - Tasa de aciertos.
        - Matriz de confusión (opcional, filas: clase real, columnas: clase predicha).
    Si calcular_matriz es False devuelve: (ecm, tasa_aciertos).
    Si calcular_matriz es True  devuelve: (ecm, tasa_aciertos, matriz_confusion).
    """
    n_salidas = salidas_deseadas.shape[1]
    n_patrones = len(entradas)
    ecm_total = 0.0
    aciertos = 0
    mc = np.zeros((n_salidas, n_salidas),
                  dtype=np.int32) if calcular_matriz else None

    for p in range(n_patrones):
        salida = red.CalcularSalida(entradas[p])
        t = salidas_deseadas[p]

        for k in range(n_salidas):
            ecm_total += (t[k] - salida[k]) ** 2

        clase_pred = int(np.argmax(salida))
        clase_real = int(np.argmax(t))
        if clase_pred == clase_real:
            aciertos += 1
        if calcular_matriz:
            mc[clase_real, clase_pred] += 1

    ecm = ecm_total / (n_patrones * n_salidas)
    tasa = aciertos / n_patrones
    return (ecm, tasa, mc) if calcular_matriz else (ecm, tasa)


def retropropagacion(red: Red, entradas_train: np.ndarray,
                     salidas_deseadas_train: np.ndarray,
                     entradas_test: np.ndarray,
                     salidas_deseadas_test: np.ndarray,
                     alfa: float, epocas: int) -> tuple:
    """
    Entrena una red MLP de tres capas mediante el algoritmo de retropropagación
    del error (backpropagation) con descenso por gradiente en línea (patrón a patrón).

    Por cada época:
      1. Para cada patrón de entrenamiento realiza feedforward y calcula:
           - delta_k para cada neurona de salida: (t_k - y_k) * f'(y_k)
           - Actualiza pesos y sesgos de la capa de salida.
           - delta_j para cada neurona oculta: (Σ delta_k * w_jk) * f'(z_j)
           - Actualiza pesos y sesgos de la capa oculta.
      2. Evalúa ECM y tasa de aciertos sobre train y test,
         almacenando los resultados en los historiales.

    Al finalizar calcula las matrices de confusión finales sobre ambos conjuntos.

    Devuelve: (historial_train, historial_test, mc_train, mc_test)
      donde cada historial es una lista de tuplas (ecm, tasa) por época.
    """
    historial_train = []
    historial_test = []

    neuronas_ocultas = red.capas[1].neuronas
    neuronas_salida = red.capas[2].neuronas

    n_patrones = len(entradas_train)
    n_ocultas = len(neuronas_ocultas)
    n_salidas = len(neuronas_salida)

    delta_k = np.empty(n_salidas)
    idx_salida = {n: k for k, n in enumerate(neuronas_salida)}

    for epoca in range(epocas):

        for p in range(n_patrones):
            patron = entradas_train[p]
            t = salidas_deseadas_train[p]

            red.CalcularSalida(patron)

            for k in range(n_salidas):
                n_k = neuronas_salida[k]
                y_k = n_k.valor
                delta_k[k] = (t[k] - y_k) * derivada_sigmoide_bipolar(y_k)
                alfa_dk = alfa * delta_k[k]
                n_k.sesgo += alfa_dk
                for c in n_k.conexiones_entrantes:
                    c.peso += alfa_dk * c.neurona_origen.valor

            for j in range(n_ocultas):
                delta_in_j = 0.0
                for c in neuronas_ocultas[j].conexiones:
                    delta_in_j += delta_k[idx_salida[c.neurona_destino]] * c.peso
                z_j = neuronas_ocultas[j].valor
                delta_j = delta_in_j * derivada_sigmoide_bipolar(z_j)
                alfa_dj = alfa * delta_j
                neuronas_ocultas[j].sesgo += alfa_dj
                for c in neuronas_ocultas[j].conexiones_entrantes:
                    c.peso += alfa_dj * c.neurona_origen.valor

        if epoca < epocas - 1:
            historial_train.append(_evaluar_conjunto(
                red, entradas_train, salidas_deseadas_train
            ))
            historial_test.append(_evaluar_conjunto(
                red, entradas_test, salidas_deseadas_test
            ))

    # Última época: ECM, tasa y matriz de confusión en un único recorrido
    ecm_tr, tasa_tr, mc_train = _evaluar_conjunto(
        red, entradas_train, salidas_deseadas_train, calcular_matriz=True)
    ecm_te, tasa_te, mc_test = _evaluar_conjunto(
        red, entradas_test,  salidas_deseadas_test,  calcular_matriz=True)
    historial_train.append((ecm_tr, tasa_tr))
    historial_test.append((ecm_te, tasa_te))

    return historial_train, historial_test, mc_train, mc_test


def ejecutar_retropropagacion(datos: tuple, n_ocultas: int,
                              alfa: float, epocas: int,
                              peso_inicial_min: float = -0.5,
                              peso_inicial_max: float = 0.5) -> tuple:
    """
    Construye y entrena una red MLP con retropropagación a partir de los datos
    y los hiperparámetros indicados.

    Arquitectura creada:
        - Capa de entrada: tantas NeuronaEntrada como características en entradas_train.
        - Capa oculta:     n_ocultas NeuronaSigmoide con sesgo aleatorio en
                           [peso_inicial_min, peso_inicial_max].
        - Capa de salida:  tantas NeuronaSigmoide como clases en salidas_train.
    Los pesos de todas las conexiones se inicializan aleatoriamente en
    [peso_inicial_min, peso_inicial_max].

    Parámetros:
        datos:             tupla (entradas_train, salidas_train, entradas_test, salidas_test).
        n_ocultas:         número de neuronas en la capa oculta.
        alfa:              tasa de aprendizaje.
        epocas:            número máximo de épocas de entrenamiento.
        peso_inicial_min:  límite inferior para la inicialización aleatoria de pesos/sesgos.
        peso_inicial_max:  límite superior para la inicialización aleatoria de pesos/sesgos.

    Devuelve: (historial_train, historial_test, mc_train, mc_test)
    """
    entradas_train, salidas_train, entradas_test, salidas_test = datos

    n_entradas = entradas_train.shape[1]
    n_clases = salidas_train.shape[1]

    capa_entrada = Capa()
    for _ in range(n_entradas):
        capa_entrada.Añadir(NeuronaEntrada())

    capa_oculta = Capa()
    for _ in range(n_ocultas):
        capa_oculta.Añadir(NeuronaSigmoide(
            random.uniform(peso_inicial_min, peso_inicial_max)))

    capa_salida = Capa()
    for _ in range(n_clases):
        capa_salida.Añadir(NeuronaSigmoide(
            random.uniform(peso_inicial_min, peso_inicial_max)))

    capa_entrada.Conectar(capa_oculta, peso_inicial_min, peso_inicial_max)
    capa_oculta.Conectar(capa_salida, peso_inicial_min, peso_inicial_max)

    red = Red()
    red.Añadir(capa_entrada)
    red.Añadir(capa_oculta)
    red.Añadir(capa_salida)

    historial_train, historial_test, mc_train, mc_test = retropropagacion(
        red, entradas_train, salidas_train,
        entradas_test, salidas_test, alfa, epocas
    )

    return historial_train, historial_test, mc_train, mc_test
