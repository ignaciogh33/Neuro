compilar:
	@echo "No es necesario compilar (Python)."

ejecuta_problema_real2:
	python perceptrones_multicapa.py --archivo problema_real2.txt -no 10 -a 0.05 -e 500
	
ejecuta_problema_real2_prediccion:
	python perceptrones_multicapa.py -no 10 -a 0.05 -e 500 --prediccion problema_real2.txt problema_real2_no_etiquetados.txt

ejecuta_problema_real6:
	python perceptrones_multicapa.py --archivo problema_real6.txt -no 20 -a 0.1 -e 5000 -p 0.7 -n 1 --normalizacion

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
	@echo "Modos de ejecucion:"
	@echo "  Modo estandar (lectura 1): entrena y evalua sobre un fichero dividido en train/test"
	@echo "  Modo prediccion (lectura 3): entrena con un fichero y predice sobre otro sin etiquetas"
	@echo ""
	@echo "Argumentos:"
	@echo "  -no, --n_ocultas N [N ...]    Numero de neuronas en la capa oculta (defecto: 10)"
	@echo "  -a, --alfa A [A ...]          Tasa de aprendizaje (defecto: 0.05)"
	@echo "  -e, --epocas E [E ...]        Numero de epocas de entrenamiento (defecto: 100)"
	@echo "  --archivo FICHERO             Ejecutar sobre un solo fichero (modo estandar)"
	@echo "  --prediccion TRAIN TEST       Modo prediccion: entrena con TRAIN, predice TEST"
	@echo "  --normalizacion               Aplicar normalizacion z-score a los atributos"
	@echo "  -p, --porcentaje P            Porcentaje de datos para entrenamiento (defecto: 0.8)"
	@echo ""
	@echo "Ejemplos:"
	@echo "  python perceptrones_multicapa.py --archivo problema_real2.txt -no 10 -a 0.05 -e 500"
	@echo "  python perceptrones_multicapa.py --archivo problema_real6.txt -no 20 -a 0.1 -e 5000 -p 0.7 --normalizacion"
	@echo "  python perceptrones_multicapa.py --prediccion problema_real2.txt problema_real2_no_etiquetados.txt -no 10 -a 0.05 -e 500"
