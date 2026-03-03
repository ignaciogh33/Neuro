PYTHON = python

ayuda_mp:
	@echo "Uso: $(PYTHON) mp.py <fichero_entrada> <fichero_salida>"
	@echo ""
	@echo "Argumentos:"
	@echo "  fichero_entrada  Fichero con los impulsos (dos valores por linea: x1 x2)"
	@echo "  fichero_salida   Fichero donde se guardaran los valores de todas las neuronas"

ejecuta_mp:
	$(PYTHON) mp.py mp_files/entrada.txt mp_files/salida.csv
	@echo "Resultado guardado en mp_files/salida.csv:"
	@cat mp_files/salida.csv

ejecuta_logicos:
	$(PYTHON) entrenar_logicos.py

compilar:
	@echo "Python no requiere compilacion."

ayuda_perceptron:
	@echo "Uso: $(PYTHON) perceptron.py [opciones]"
	@echo ""
	@echo "Opciones:"
	@echo "  --fichero      Ruta al fichero de datos (default: P1/problema_real1.txt)"
	@echo "  --porcentaje   Porcentaje de datos para entrenamiento (default: 0.7)"
	@echo "  --alfa         Tasa de aprendizaje (default: 0.1)"
	@echo "  --umbral       Umbral de activacion (default: 0.2)"
	@echo "  --sesgo        Sesgo inicial (default: 0.0)"
	@echo "  --timeout      Tiempo maximo de entrenamiento en segundos (default: 10)"

ejecuta_perceptron:
	$(PYTHON) perceptron.py --fichero P1/problema_real1.txt --porcentaje 0.7 --alfa 0.1 --umbral 0.2 --sesgo 0.0 --timeout 5

ayuda_adaline:
	@echo "Uso: $(PYTHON) adaline.py [opciones]"
	@echo ""
	@echo "Opciones:"
	@echo "  --fichero      Ruta al fichero de datos (default: P1/problema_real1.txt)"
	@echo "  --porcentaje   Porcentaje de datos para entrenamiento (default: 0.7)"
	@echo "  --alfa         Tasa de aprendizaje (default: 0.01)"
	@echo "  --tolerancia   Tolerancia de convergencia (default: 0.01)"
	@echo "  --sesgo        Sesgo inicial (default: 0.0)"
	@echo "  --timeout      Tiempo maximo de entrenamiento en segundos (default: 10)"

ejecuta_adaline:
	$(PYTHON) adaline.py --fichero P1/problema_real1.txt --porcentaje 0.7 --alfa 0.01 --tolerancia 0.01 --sesgo 0.0 --timeout 10
