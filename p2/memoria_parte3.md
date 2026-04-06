# 3. Problemas reales y normalización

## 3.1. Análisis del problema

Los problemas reales 4 y 6 presentan atributos cuyos valores tienen escalas muy diferentes o
magnitudes muy elevadas, lo que impide obtener buenos resultados con el perceptrón multicapa
sin un preprocesamiento adecuado.

### Problema real 4

El problema 4 tiene 699 patrones con 9 atributos y 2 clases. Al calcular la media y la desviación
estándar de cada atributo se observan escalas radicalmente diferentes:

| Atributo | Media       | Desv. estándar |
|----------|-------------|----------------|
| 1        | 4.42e+03    | 2.81e+03       |
| 2        | 3.13e-02    | 3.05e-02       |
| 3        | 3.21e-02    | 2.97e-02       |
| 4        | 2.81e+01    | 2.85e+01       |
| 5        | 6.43e-01    | 4.43e-01       |
| 6        | 3.54e-02    | 3.60e-02       |
| 7        | 3.44e-02    | 2.44e-02       |
| 8        | 2.87e-02    | 3.05e-02       |
| 9        | 1.59e+03    | 1.71e+03       |

Los atributos 1 y 9 tienen valores en los miles, mientras que los atributos 2, 3, 6, 7 y 8 tienen
valores del orden de 0.01–0.03. Esto causa dos problemas:

- Los atributos con valores grandes dominan la suma ponderada de entrada a cada neurona,
  haciendo que los atributos pequeños sean prácticamente irrelevantes.
- Las entradas netas resultantes son tan grandes que la función sigmoide bipolar satura
  (devuelve valores cercanos a -1 o 1), produciendo derivadas próximas a cero y anulando
  el gradiente durante la retropropagación.

### Problema real 6

El problema 6 tiene 14980 patrones con 14 atributos y 2 clases. Todos los atributos tienen
valores en el rango de miles (medias entre 4009 y 4644), pero con desviaciones estándar que
varían enormemente:

| Atributo | Media    | Desv. estándar |
|----------|----------|----------------|
| 1        | 4321.92  | 2491.99        |
| 2        | 4009.77  | 45.94          |
| 3        | 4264.02  | 44.43          |
| 4        | 4164.95  | 5216.23        |
| 5        | 4341.74  | 34.74          |
| 6        | 4644.02  | 2924.69        |
| 7        | 4110.40  | 4600.77        |
| 8        | 4616.06  | 29.29          |
| 9        | 4218.83  | 2136.34        |
| 10       | 4231.32  | 38.05          |
| 11       | 4202.46  | 37.78          |
| 12       | 4279.23  | 41.54          |
| 13       | 4615.21  | 1208.33        |
| 14       | 4416.44  | 5891.09        |

Los valores absolutos son todos muy elevados (media ~4000), lo que satura la sigmoide
desde el primer instante del entrenamiento. Además, las diferencias entre desviaciones
estándar (desde 29 hasta 5891) hacen que ciertos atributos dominen sobre otros.

## 3.2. Normalización aplicada

Se ha implementado la normalización z-score (también llamada estandarización), que
transforma cada atributo para que tenga media 0 y desviación estándar 1:

$$x'_j = \frac{x_j - \mu_j}{\sigma_j}$$

donde $\mu_j$ y $\sigma_j$ son la media y la desviación estándar del atributo $j$, calculadas
**únicamente sobre el conjunto de entrenamiento**.

### Procedimiento

1. Se calcula la media y la desviación estándar de cada atributo usando exclusivamente
   los datos de entrenamiento.
2. Se aplica la transformación a los datos de entrenamiento: cada valor se centra restando
   la media y se escala dividiendo por la desviación estándar.
3. Se aplica **la misma transformación** (con la media y desviación del entrenamiento) a los
   datos de test. Esto es fundamental para que la red reciba datos en la misma escala con la
   que fue entrenada, evitando fugas de información del conjunto de test.

Para atributos con desviación estándar igual a cero (constantes), se evita la división por cero
manteniendo el valor centrado en la media sin escalar.

### Por qué se hace

- **Evita la saturación de la sigmoide**: Al centrar los datos en torno a 0 con dispersión
  unitaria, las entradas netas de las neuronas caen en la zona activa de la sigmoide bipolar,
  donde la derivada es significativa y el gradiente puede fluir durante la retropropagación.
- **Equipara la contribución de todos los atributos**: Ningún atributo domina sobre los demás
  por tener una escala mayor, permitiendo que la red aprenda la importancia relativa de cada
  uno a través de los pesos.
- **Mejora la convergencia**: El descenso por gradiente converge más rápido cuando los datos
  están en escalas similares, ya que la superficie de error es más simétrica.

## 3.3. Resultados

### Problema real 4 (500 épocas, 20 neuronas ocultas, α=0.1, 70% entrenamiento)

| Configuración    | ECM Train | Tasa Train | ECM Test | Tasa Test |
|------------------|-----------|------------|----------|-----------|
| Sin normalización | 0.2085   | 67.48%     | 0.2063   | 71.43%    |
| Con normalización | 0.0325   | 96.32%     | 0.0187   | 98.57%    |

Matriz de confusión (test, con normalización):

|              | Pred. clase 0 | Pred. clase 1 |
|--------------|---------------|---------------|
| Real clase 0 | 148           | 2             |
| Real clase 1 | 1             | 59            |

La normalización mejora la tasa de aciertos en test de 71% a 99%, demostrando que sin ella
la red no puede aprovechar correctamente los atributos con escalas tan dispares.

### Problema real 6 (5000 épocas, 20 neuronas ocultas, α=0.1, 70% entrenamiento)

| Configuración    | ECM Train | Tasa Train | ECM Test | Tasa Test |
|------------------|-----------|------------|----------|-----------|
| Sin normalización | 0.2473   | 55.33%     | 0.2483   | 54.63%    |
| Con normalización | 0.1533   | 78.91%     | 0.1572   | 78.66%    |

Matriz de confusión (test, con normalización):

|              | Pred. clase 0 | Pred. clase 1 |
|--------------|---------------|---------------|
| Real clase 0 | 2234          | 239           |
| Real clase 1 | 720           | 1301          |

Sin normalización la red no supera el 55% (prácticamente azar para 2 clases). Con
normalización alcanza el 79%, confirmando que el preprocesamiento de los atributos es
imprescindible para que la retropropagación funcione correctamente en este dataset.
