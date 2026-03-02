ayuda_mp:
	@echo "Uso: python mp.py <fichero_entrada> <fichero_salida>"
	@echo ""
	@echo "Argumentos:"
	@echo "  fichero_entrada  Fichero con los impulsos (dos valores por linea: x1 x2)"
	@echo "  fichero_salida   Fichero donde se guardaran los valores de todas las neuronas"

ejecuta_mp:
	python mp.py mp_files/entrada.txt mp_files/salida.csv
	@echo "Resultado guardado en mp_files/salida.csv:"
	@cat mp_files/salida.csv

ejecuta_logicos:
	python entrenar_logicos.py

