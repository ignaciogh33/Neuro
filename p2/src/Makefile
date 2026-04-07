compilar:
	@echo "No es necesario compilar (Python)."

ejecuta_problema_real2:
	python perceptrones_multicapa.py --archivo problema_real2.txt -no 10 -a 0.05 -e 500
	
ejecuta_problema_real2_prediccion:
	python perceptrones_multicapa.py -no 10 -a 0.05 -e 500 --prediccion problema_real2.txt problema_real2_no_etiquetados.txt

ejecuta_problema_real6:
	python perceptrones_multicapa.py --archivo problema_real6.txt -no 20 -a 0.1 -e 5000 -p 0.7 --normalizacion

ayuda_retro_funcion:
	@echo "Funcion: ejecutar_retropropagacion(datos, n_ocultas, alfa, epocas, peso_inicial_min=-0.5, peso_inicial_max=0.5)"
	@echo ""
	@echo "Parametros:"
	@echo "  datos              Tupla de 4 elementos: (entradas_train, salidas_train, entradas_test, salidas_test)"
	@echo "  n_ocultas          Numero de neuronas en la capa oculta"
	@echo "  alfa               Tasa de aprendizaje (alpha)"
	@echo "  epocas             Numero de epocas de entrenamiento"
	@echo "  peso_inicial_min   Limite inferior para inicializacion de pesos/sesgos (defecto: -0.5)"
	@echo "  peso_inicial_max   Limite superior para inicializacion de pesos/sesgos (defecto:  0.5)"
	@echo ""
	@echo "Retorno: (historial_train, historial_test, mc_train, mc_test)"
	@echo "  historial_train   Lista de tuplas (ecm, tasa_aciertos) por epoca sobre el conjunto de entrenamiento"
	@echo "  historial_test    Lista de tuplas (ecm, tasa_aciertos) por epoca sobre el conjunto de test"
	@echo "  mc_train          Matriz de confusion final sobre el conjunto de entrenamiento"
	@echo "  mc_test           Matriz de confusion final sobre el conjunto de test"

ayuda_retro:
	@echo "Uso: python perceptrones_multicapa.py [opciones]"
	@echo ""
	@echo "Comportamiento por defecto (sin flags):"
	@echo "  Ejecuta iterativamente los archivos: problema_real1.txt, problema_real2.txt, problema_real3.txt, problema_real5.txt"
	@echo "  Aplica hiperparametros base: 10 ocultas, alfa=0.05, 100 epocas, 5 ejecuciones, 80% train."
	@echo "  Guarda resultados agregados (y medias) en: resultados_mlp_reales.csv"
	@echo ""
	@echo "Modos de ejecucion:"
	@echo "  Modo estandar (lectura 1): Entrena y evalua dividiendo los datos en train/test (permite multiples hiperparametros para barrido)."
	@echo "  Modo prediccion (lectura 3): Entrena con un fichero de TRAIN, y predice sobre un fichero de TEST sin etiquetas guardando la salida txt."
	@echo ""
	@echo "Argumentos completados:"
	@echo "  -no, --n_ocultas N [N ...]    Lista de neuronas en la capa oculta a iterar (defecto: 10)"
	@echo "  -a, --alfa A [A ...]          Lista de tasas de aprendizaje a iterar (defecto: 0.05)"
	@echo "  -e, --epocas E [E ...]        Lista de epocas de entrenamiento a iterar (defecto: 100)"
	@echo "  -n, --n_ejecuciones NUM       Numero de ejecuciones consecutivas para promediar resultados (defecto: 5)"
	@echo "  --archivo FICHERO             Forzar ejecucion sobre un solo fichero en vez de toda la lista por defecto."
	@echo "  --prediccion TRAIN TEST       Activar modo prediccion. Requiere fichero train y fichero test."
	@echo "  --normalizacion               Aplicar normalizacion iterativa z-score a los atributos (media=0, desviacion=1)."
	@echo "  -p, --porcentaje P            Proporcion de datos usados para entrenamiento (defecto: 0.8)."
	@echo ""
	@echo "Ejemplos:"
	@echo "  # 1) Comportamiento sin flags en iteracion a todas las BBDDs reales base:"
	@echo "  python perceptrones_multicapa.py"
	@echo ""
	@echo "  # 2) Barrido de parametros en un fichero concreto:"
	@echo "  python perceptrones_multicapa.py --archivo problema_real2.txt -no 5 10 20 -a 0.05 0.1 -e 100 500"
	@echo ""
	@echo "  # 3) Ejecucion con normalizacion, solo 1 ejecucion y 70% train:"
	@echo "  python perceptrones_multicapa.py --archivo problema_real6.txt -no 20 -a 0.1 -e 5000 -p 0.7 -n 1 --normalizacion"
	@echo ""
	@echo "  # 4) Modo prediccion (se ignorara 'archivo', la lista o el flag '-n', y coge los primeros hiperparametros):"
	@echo "  python perceptrones_multicapa.py --prediccion problema_real2.txt problema_real2_no_etiquetados.txt -no 10 -a 0.05 -e 500"
