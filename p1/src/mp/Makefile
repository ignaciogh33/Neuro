ayuda_mp:
	@echo "Uso: python -m mp.mp <fichero_entrada> <fichero_salida>"
	@echo ""
	@echo "Argumentos:"
	@echo "  fichero_entrada  Fichero con los impulsos (dos valores por linea: x1 x2)"
	@echo "  fichero_salida   Fichero donde se guardaran los valores de todas las neuronas"
	@echo ""
	@echo "Ejecutar desde el directorio p1, salvo si ejecutamos el makefile"

ejecuta_mp:
	cd ../..
	python -m mp.mp /mp/entrada.txt mp/salida.txt
	@echo "Resultado guardado en salida.txt:"
	@cat mp/salida.txt
