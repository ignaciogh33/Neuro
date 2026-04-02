compilar:
	@echo "No es necesario compilar (Python)."

ejecuta_problema_real2:
	python perceptrones_multicapa.py --archivo problema_real2.txt -no 10 -a 0.05 -e 500
	
ejecuta_problema_real2_prediccion:
	python perceptrones_multicapa.py -no 10 -a 0.05 -e 500 --prediccion problema_real2.txt problema_real2_no_etiquetados.txt

ejecuta_problema_real6:
	python perceptrones_multicapa.py --archivo problema_real6.txt -no 20 -a 0.1 -e 5000 -p 0.7 -n 1 --normalizacion

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
