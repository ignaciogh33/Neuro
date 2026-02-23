ayuda_mp:
	@echo "Uso: python -m src.mp.mp <fichero_entrada> <fichero_salida>"
	@echo ""
	@echo "Argumentos:"
	@echo "  fichero_entrada  Fichero con los impulsos (dos valores por linea: x1 x2)"
	@echo "  fichero_salida   Fichero donde se guardaran los valores de todas las neuronas"
	@echo ""
	@echo "Ejecutar desde el directorio raíz del proyecto (padre de src), salvo si ejecutamos el makefile, que es desde src"

ejecuta_mp:
	cd ../.. && python -m src.mp.mp src/mp_files/entrada.txt src/mp_files/salida.txt
	@echo "Resultado guardado en salida.txt:"
	@cat mp_files/salida.txt
