import os
import csv
import itertools
import argparse
from redes_neuronales import leer1, ejecutar_retropropagacion, ejecutar_retropropagacion_matricial, normalizar

# Configuración de los archivos de datos
archivos = [
    "problema_real1.txt",
    "problema_real2.txt",
    "problema_real3.txt",
    "problema_real5.txt"
]

# Configuración de los hiperparámetros a probar por DEFECTO
# (Se usarán estos si no se pasan parámetros por terminal)
DEFAULT_N_OCULTAS = [10]
DEFAULT_ALFA = [0.05]
DEFAULT_EPOCAS = [100]

n_ejecuciones = 5
script_dir = os.path.dirname(os.path.abspath(__file__))
directorio_datos = os.path.join(script_dir, "P2", "data")
archivo_csv_salida = os.path.join(script_dir, "resultados_mlp_reales.csv")

def main():
    parser = argparse.ArgumentParser(description='Entrenar Perceptrón Multicapa con distintos hiperparámetros.')
    parser.add_argument('-no', '--n_ocultas', type=int, nargs='+', default=None, help='Lista de neuronas en capa oculta')
    parser.add_argument('-a', '--alfa', type=float, nargs='+', default=None, help='Lista de tasas de aprendizaje')
    parser.add_argument('-e', '--epocas', type=int, nargs='+', default=None, help='Lista de épocas')
    parser.add_argument('--archivo', type=str, default=None, help='Forzar ejecución sobre un solo archivo')
    parser.add_argument('--prediccion', nargs=2, metavar=('TRAIN', 'TEST'), help='Modo predicción: usar lectura 3 con [archivo_train] [archivo_test_no_etiquetado]')
    parser.add_argument('--normalizacion', action='store_true', default=False, help='Aplicar normalización z-score a los atributos de entrada')
    parser.add_argument('-p', '--porcentaje', type=float, default=0.8, help='Porcentaje de datos para entrenamiento (defecto: 0.8)')
    parser.add_argument('-n', '--n_ejecuciones', type=int, default=5, help='Número de ejecuciones para promediar (defecto: 5)')
    args = parser.parse_args()

    n_ocultas_list = args.n_ocultas if args.n_ocultas else DEFAULT_N_OCULTAS
    alfa_list = args.alfa if args.alfa else DEFAULT_ALFA
    epocas_list = args.epocas if args.epocas else DEFAULT_EPOCAS

    archivos_ejecutar = [args.archivo] if args.archivo else archivos
    
    # MODO PREDICCION (Lectura 3)
    if args.prediccion:
        from redes_neuronales import leer3
        from redes_neuronales import Capa, NeuronaEntrada, NeuronaSigmoide, Red
        import random
        import numpy as np

        train_file = args.prediccion[0]
        test_file = args.prediccion[1]
        
        ruta_train = os.path.join(directorio_datos, train_file)
        ruta_test = os.path.join(directorio_datos, test_file)
        
        datos_completos = leer3(ruta_train, ruta_test)
        entradas_train, salidas_train, entradas_test, _ = datos_completos # salidas_test no importa aquí

        if args.normalizacion:
            entradas_train, entradas_test = normalizar(entradas_train, entradas_test)
            datos_completos = (entradas_train, salidas_train, entradas_test, _)

        # Coger el primer hiperparámetro introducido (o default)
        n_ocultas = n_ocultas_list[0]
        alfa = alfa_list[0]
        epocas = epocas_list[0]
        
        print(f"Modo Predicción -> Entrenando con {train_file}, Prediciendo {test_file}")
        print(f"Hiperparámetros -> N_ocultas: {n_ocultas} | Alfa: {alfa} | Epocas: {epocas}")

        _, _, _, _ = ejecutar_retropropagacion(
            datos_completos, n_ocultas, alfa, epocas
        )
        # NOTA: Para predecir y escribir en fichero correctamente, la red debe ser accesible 
        # tras ejecutar retropropagacion. Vamos a reconstruir y entrenar manualmente rápido o 
        # necesitas que 'ejecutar_retropropagacion' devuelva la red instanciada.

        # Entrenamiento completo adaptado para guardar las predicciones al final:
        peso_min, peso_max = -0.5, 0.5
        n_entradas = entradas_train.shape[1]
        n_clases = salidas_train.shape[1]

        # Crear arquitectura
        c_entrada = Capa()
        for _ in range(n_entradas): c_entrada.Añadir(NeuronaEntrada())

        c_oculta = Capa()
        for _ in range(n_ocultas): c_oculta.Añadir(NeuronaSigmoide(random.uniform(peso_min, peso_max)))

        c_salida = Capa()
        for _ in range(n_clases): c_salida.Añadir(NeuronaSigmoide(random.uniform(peso_min, peso_max)))

        c_entrada.Conectar(c_oculta, peso_min, peso_max)
        c_oculta.Conectar(c_salida, peso_min, peso_max)

        red = Red()
        red.Añadir(c_entrada)
        red.Añadir(c_oculta)
        red.Añadir(c_salida)

        from redes_neuronales import retropropagacion
        retropropagacion(
            red, entradas_train, salidas_train,
            entradas_test, np.zeros((len(entradas_test), n_clases)), # Fake salidas_test
            alfa, epocas
        )

        archivo_predicciones = os.path.join(script_dir, "..", f"prediccion_{train_file}")
        with open(archivo_predicciones, 'w') as f_out:
            for i, p in enumerate(entradas_test):
                salida_pura = red.CalcularSalida(p)
                idx_ganador = int(np.argmax(salida_pura))
                
                # Formato solicitado: 1 para la clase ganadora, -1 para el resto
                prediccion = ["1" if j == idx_ganador else "-1" for j in range(len(salida_pura))]
                f_out.write(" ".join(prediccion) + "\n")
                
        print(f"✔ Predicciones guardadas en: {archivo_predicciones}")
        return
    
    # MODO ESTÁNDAR (Lectura 1 - CSV de resultados)
    print(f"Iniciando evaluación. Los resultados se guardarán en: {archivo_csv_salida}")
    print(f"Hiperparámetros a probar -> Ocultas: {n_ocultas_list} | Alfa: {alfa_list} | Epocas: {epocas_list}")
    
    with open(archivo_csv_salida, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Archivo", "N_Ocultas", "Alfa", "Epocas", 
            "ECM_Train_Medio", "Tasa_Train_Medio", 
            "ECM_Test_Medio", "Tasa_Test_Medio"
        ])
        
        for archivo in archivos_ejecutar:
            ruta_archivo = os.path.join(directorio_datos, archivo)
            
            if not os.path.exists(ruta_archivo):
                print(f"Advertencia: No se encontró el archivo {ruta_archivo}")
                continue
                
            combinaciones = list(itertools.product(n_ocultas_list, alfa_list, epocas_list))
            
            print(f"\n======================================")
            print(f"Procesando {archivo}...")
            print(f"======================================")
            
            for n_ocultas, alfa, epocas in combinaciones:
                ecm_train_acum = 0.0
                tasa_train_acum = 0.0
                ecm_test_acum = 0.0
                tasa_test_acum = 0.0
                
                print(f"\n  Configuración: N_ocultas={n_ocultas}, Alfa={alfa}, Epocas={epocas}")
                
                for i in range(args.n_ejecuciones):
                    datos = leer1(ruta_archivo, args.porcentaje)

                    if args.normalizacion:
                        entradas_tr, salidas_tr, entradas_te, salidas_te = datos
                        entradas_tr, entradas_te = normalizar(entradas_tr, entradas_te)
                        datos = (entradas_tr, salidas_tr, entradas_te, salidas_te)

                    historial_train, historial_test, mc_train, mc_test = ejecutar_retropropagacion_matricial(
                        datos, n_ocultas, alfa, epocas
                    )
                    
                    # Extraer resultados de la última época
                    ecm_train, tasa_train = historial_train[-1]
                    ecm_test, tasa_test = historial_test[-1]
                    
                    ecm_train_acum += ecm_train
                    tasa_train_acum += tasa_train
                    ecm_test_acum += ecm_test
                    tasa_test_acum += tasa_test
                
                # Promediar
                ecm_train_medio = ecm_train_acum / args.n_ejecuciones
                tasa_train_medio = tasa_train_acum / args.n_ejecuciones
                ecm_test_medio = ecm_test_acum / args.n_ejecuciones
                tasa_test_medio = tasa_test_acum / args.n_ejecuciones
                
                # Guardar fila en CSV
                writer.writerow([
                    archivo, n_ocultas, alfa, epocas,
                    f"{ecm_train_medio:.4f}", f"{tasa_train_medio:.4f}",
                    f"{ecm_test_medio:.4f}", f"{tasa_test_medio:.4f}"
                ])
                f.flush() # Asegurar que se escriba en disco
                
                # Imprimir resultados por terminal
                print(f"    -> ECM Train: {ecm_train_medio:.4f} | Tasa Train: {tasa_train_medio:.4f}")
                print(f"    -> ECM Test:  {ecm_test_medio:.4f} | Tasa Test:  {tasa_test_medio:.4f}")

if __name__ == "__main__":
    main()
