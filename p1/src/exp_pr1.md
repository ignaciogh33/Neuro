# Implementación de Redes Neuronales: Adaline y Perceptrón

Ambas implementaciones (`adaline.py` y `perceptron.py`) resuelven de manera análoga el clásico problema de clasificación multiclase, apoyándose en la infraestructura orientada a objetos contenida en `redes_neuronales.py`. 

A nivel de arquitectura de código, ambos siguen una **estructura idéntica de 6 etapas**, actuando como "archivos cliente" que ensamblan y dirigen el flujo de la red:

### 1. Parámetros y Lectura de Datos (`argparse` y `leer1`)
Ambos scripts exponen parámetros por línea de comandos utilizando la librería `argparse` (`--alfa`, `--sesgo`, `--porcentaje`, `--timeout`, etc.). Para el Perceptrón se añade `--umbral` y para el Adaline se añade `--tolerancia`.
Los datos se cargan usando la función `leer1(...)`, que corresponde al "Modo 1", encargada de barajar (mezclar aleatoriamente) el dataset y partirlo en dos subconjuntos: Entrenamiento y Validación (Test), todo de manera dinámica en función al porcentaje parametrizado.

### 2. Definición Topológica (Nodos)
Dependiendo del número de atributos del dataset y del número de clases (leídas de los ficheros), el flujo de trabajo inicializa las entidades subyacentes:
* **Entradas:** Crea una lista de $N$ objetos `NeuronaEntrada` donde $N$ es la cantidad de características del problema.
* **Salidas:** Crea otra lista de las neuronas del modelo propiamente tal: objetos `NeuronaPerceptron` en el primer script, o `NeuronaAdaline` en el segundo.

### 3. Conectividad (Fully Connected)
Paso seguido, conectan físicamente las neuronas en una malla completamente conectada. Mediante un doble bucle, desde cada neurona de entrada se llama al método `Conectar()` apuntando hacia cada neurona de salida. 
* El script del Perceptrón inicializa estos pesos a `0.0` (los ajustará luego dentro de su `Aprender()`).
* El script del Adaline pasa pesos aleatorios entre `[-0.5, 0.5]`.

### 4. Empaquetado en Capas y Red
Para manejar las propagaciones conjuntas de forma elegante, las listas de neuronas se insertan dentro de objetos de tipo `Capa` (`capa_entrada` y `capa_salida`). Finalmente, para cerrar la abstracción, estas capas se añaden al contenedor maestro `Red()`.

### 5. Entrenamiento Supervisado y Límite de Tiempo (Timeout)
Aquí reside una de las mecánicas más importantes en tu código. Se itera clase por clase (simulando una clasificación "Uno contra Resto") y se llama al método `Aprender()` de la neurona correspondiente.
Al enfrentarse con problemas inherentemente *no linealmente separables* (donde ninguno de estos algoritmos básicos convergería nunca), los scripts integran una **defensa anti-bloqueo**:
* Importan el módulo de sistema operativo `signal`.
* Utilizan `signal.alarm(timeout)` para lanzar una interrupción silenciosa pasados X segundos definidos.
* Si el método `Aprender()` excede este límite, la red captura un `TimeoutException` personalizado, finaliza por fuerza el entrenamiento, y reporta *"No converge (timeout Xs)"*. En el caso de Adaline, adicionalmente, se fuerza el estado interno `en_entrenamiento = False` para habilitar el disparo predictivo binario posterior.
* El script imprime amigablemente con formato matemático la función y frontera decodificando los pesos de la neurona una vez parada.

### 6. Fase Forward y Evaluación de Rendimientos
Finalizado el ajuste, se evalúan tanto los patrones de entrenamiento como los de Test.
* El bucle itera los patrones para embutirlos dentro de las neuronas por medio de la función `Inicializar(patron[i])`.
* Se invoca explícitamente al flujo de alimentación hacia delante: `capa_entrada.Propagar()` que envía los valores multiplicados por su peso, y `capa_salida.Disparar()`, que integra y genera la salida de los perceptrones/adalines.
* **Métricas:** Compara lo resuelto por las neuronas (`neurona.valor`) contra las etiquetas reales. Contabiliza el número general de Aciertos y presenta la variable **Precisión Test (%)**. Adicionalmente, computa la magnitud del fallo cuadrático acumulando `error ** 2`, presentando finalmente el valor de **Error Cuadrático Medio (ECM)** de la red.
