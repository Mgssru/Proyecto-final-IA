# Proyecto final IA
***

## Titulo del trabajo

**Predicción de gente propensa a ser fumadora a partir de sus datos médicos**

## Lista de los integrantes del grupo.

<p align = "center">
Simón Sierra Ruiz
</p>

## Introducción del proyecto (donde se consigue el dataset! cuantas muestras tiene que caracteristicas y que etiquetas.... etc... )

El Dataset fue conseguido en: https://www.kaggle.com/datasets/mirichoi0218/insurance, donde nos proporcionan un dataset bastante basico con los datos médicos de 1338 personas, estos datos son: 

  * Edad
  * Sexo
  * Índice de masa corporal
  * Cantidad de hijos
  * Fumador
  * Región
  * Costos médicos individuales facturados por el seguro de salud

## Desarrollo:

 * Tipo de problema:

De las caracteristicas anteriores, se desea predecir lo propensa que es una persona a ser fumadora, es decir, un problema de supervisado de clasificación, donde 0 -> No Fumador y 1 -> Fumador.

 * Preprocesamiento del conjunto de datos:
   * Limpieza:
     
     - **Eliminación de valores faltantes o NaN**

      Al usar el comando: *data.isnull().sum()*, notamos que el dataset no tiene ningun valor nulo, así que este paso es innecesario.

     - **Pasamos los valores categóricos a numéricos**

      El dataset cuenta con 3 valores categoricos: Sexo, Fumador y Región. Se le asigno valores númericos empezando desde el 0 hasta la cantidad (n-1) de diferentes tipos de datos existentes. Por ejemplo: Región tiene 4 diferentes ('southwest', 'southeast', 'northwest', 'northeast'), de modo que para cada uno de ellos asignamos los valores 0 , 1 , 2 , 3 respectivamente.

     - **Creamos un archivo csv con nuestros datos limpios de tal manera que por columnas tenemos nuestras características y por filas cada muestra las etiqueta(s) “Y" las tenemos en la última(s) columnas**

      Como anteriormente mencionamos, deseamos predecir la probabilidad de que una persona sea propensa a fumar dependiendo de las caracteristicas de su registro de salud, y dependiendo del resultado, este se clasifique en 0 -> no fumador y 1 -> Fumador, de modo que para el nuevo archivo csv seleccionamos los datos de la columna de fumador y lo separamos del conjunto X, en su propio vector Y.

   * Selección del metodo de evaluación:

     - Métodos supervisados:
       - **Vamos a usar un ##% para entrenar y un ##% para validar,seleccionamos estos a partir de la cantidad de datos, de tal manera que el conjunto de validación que suele ser el más pequeño de los dos tenga por lo menos una cantidad significativa de representantes de todas las clases (10 mínimo)**

      Para ello usamos la función: *X_train, X_test, Y_train, Y_test = train_test_split( S, Y, test_size=0.1, train_size=0.9, random_state=0 )* donde determinamos un conjunto de entrenamiento del 90% y un conjunto de validación del 10%, el cual es de 134 muestras (muchisimo mayor al mínimo de 10)

   * Normalización:

     - Usando el conjunto de entrenamiento (o todos los datos en caso de que se esté haciendo cross-validación) normalizar usando:
       - El valor min y max (minmaxscaler).

   * Representación y reducción dimensional:

     - Hacer una descomposición en componentes principales del dataset (PCA)
     - Evaluar si este se viera beneficiado por reducir su dimensionalidad:
       - Tomar los valores propios y dividirlos por la suma de valores propios para hallar la varianza explicada





para ello se utilizo 3 tipos de métodos:

  * Maquina de soporte Vectorial
  * KNN
  * Regresión Logistica

### Maquina de soporte vectorial:

Para usar maquinas de soporte vectorial se usa la función: svm.SVC(), donde variaran los parametros de "kernel" y "gamma", para luego entrenarlo con el conjunto de entrenamiento que determinemos

    # Se definen diferentes Kernel para evaluar
    kernels = [ 'linear' , 'poly' , 'rbf' , 'sigmoid' ]

    Kernel = 2

    # Se crea la maquina de soporte vectorial
    msv = svm.SVC( kernel = kernels[ Kernel ] , gamma='auto' , random_state=895 )
    msv.fit( Xn_t , Y_train )

    # Se predice con la prueba
    Y_test_predicted = msv.predict( Xn_te )
    Y_test_scores = msv.decision_function( Xn_te )

    # Crea los valores de la curva ROC
    fpr , tpr , thresholds = roc_curve( Yn_te , Y_test_scores )
    roc_auc = roc_auc_score( Y_test , Y_test_scores )

    # Calculo del coeficiente de correlación de Matthew
    MCC = matthews_corrcoef( Y_test , Y_test_predicted )
    print("")
    print( "Coeficiente de correlación de Matthew: " , MCC )

    # Puntuación de precisión
    ACC = accuracy_score( Y_test , Y_test_predicted )
    print( "Precisión: " , ACC )

## Resultados (puede poner unas imagenes de una tabla de resultados en el README.md)

## Conclusiones

## Link al video de youtube en el readme.md
