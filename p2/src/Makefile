ejecuta_problema_real2:
	python perceptrones_multicapa.py --archivo problema_real2.txt -no 10 -a 0.05 -e 500
	
ejecuta_problema_real2_prediccion:
	python perceptrones_multicapa.py -no 10 -a 0.05 -e 500 --prediccion problema_real2.txt problema_real2_no_etiquetados.txt

ayuda_retro:
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
