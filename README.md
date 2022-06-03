# Proyecto final IA
***

## Titulo del trabajo

**Predicción de gente propensa a ser fumadora**

## Lista de los integrantes del grupo.

<p align = "center">
Simón Sierra Ruiz
</p>

## Introducción del proyecto

El Dataset fue conseguido en: https://www.kaggle.com/datasets/mirichoi0218/insurance, donde nos proporcionan un dataset bastante basico con los datos médicos de 1338 personas, estos datos son: 

  * Edad
  * Sexo
  * Índice de masa corporal
  * Cantidad de hijos
  * Fumador
  * Región
  * Costos médicos individuales facturados por el seguro de salud

## Desarrollo

### Tipo de problema

De las caracteristicas anteriores, se desea predecir lo propensa que es una persona a ser fumadora, es decir, un problema de supervisado de clasificación, donde 0 -> No Fumador y 1 -> Fumador.

### Preprocesamiento del grupo de datos
   * Limpieza:
     
     - **Eliminación de valores faltantes o NaN**

      Al usar el comando: *data.isnull().sum()*, notamos que el dataset no tiene ningun valor nulo, así que este paso es innecesario.
 
        ![imagen](https://user-images.githubusercontent.com/86111238/171550074-720ecba9-07b2-4839-ae5a-4e6a6475a386.png)

     - **Pasamos los valores categóricos a numéricos**

      El dataset cuenta con 3 valores categoricos: Sexo, Fumador y Región. Se le asigno valores númericos empezando desde el 0 hasta la cantidad (n-1) de diferentes tipos de datos existentes. Por ejemplo: Región tiene 4 diferentes ('southwest', 'southeast', 'northwest', 'northeast'), de modo que para cada uno de ellos asignamos los valores 0 , 1 , 2 , 3 respectivamente.

        ![imagen](https://user-images.githubusercontent.com/86111238/171550176-3349abc5-1522-4e5e-9ef2-66d9797e5000.png)


     - **Creamos un archivo csv con nuestros datos limpios de tal manera que por columnas tenemos nuestras características y por filas cada muestra las etiqueta(s) “Y" las tenemos en la última(s) columnas**

      Como anteriormente mencionamos, deseamos predecir la probabilidad de que una persona sea propensa a fumar dependiendo de las caracteristicas de su registro de salud, y dependiendo del resultado, este se clasifique en 0 -> no fumador y 1 -> Fumador, de modo que para el nuevo archivo csv seleccionamos los datos de la columna de fumador y lo separamos del conjunto X, en su propio vector Y.

        ![imagen](https://user-images.githubusercontent.com/86111238/171550434-11838726-7fd8-4b61-9f78-924bb3f33ac5.png)


   * Selección de evaluación:

     - **Dado a que el metodo es supervisado se uso un ##% para entrenar y un ##% para validar,seleccionamos estos a partir de la cantidad de datos, de tal manera que el conjunto de validación que suele ser el más pequeño de los dos tenga por lo menos una cantidad significativa de representantes de todas las clases (10 mínimo)**

      Para ello se uso la función: *X_train, X_test, Y_train, Y_test = train_test_split( S, Y, test_size=0.1, train_size=0.9, random_state=0 )* donde determinamos un conjunto de entrenamiento del 90% y un conjunto de validación del 10%, el cual es de 134 muestras (muchisimo mayor al mínimo de 10)

        ![imagen](https://user-images.githubusercontent.com/86111238/171550492-a63de7c4-868b-4747-ad2d-dd2f28107840.png)

   * Normalización:

     - Usando el conjunto de entrenamiento, normalizar usando: **El valor min y max (minmaxscaler)**

      Para ello se uso la función: *scaler = MinMaxScaler()* el cual dota al objeto scaler de la función *fit()*, encargada de transformar a la base de datos de entrenamiento con sus valores normalizados.

         scaler = MinMaxScaler()
         
         scaler.fit( X_train )
         Xn_t = scaler.transform( X_train )

        ![imagen](https://user-images.githubusercontent.com/86111238/171551211-22f811ba-50ed-481a-85cd-8fcc769a6303.png)


     - Normalizar también el conjunto de validación con los valores obtenidos del conjunto de prueba (min, max, media, desviación, etc...)

      Repetimos el proceso para el conjunto de entrenamiento "Y" que definimos anteriormente, el cual sólo es la columna de *Fumador*, aunque a la larga esto no tiene mucho sentido dado a que los datos oscilan entre 0 y 1, así que al normalizar tendran el mismo valor.
      
         scaler.fit( Y_train )
         Yn_t = scaler.transform( Y_train )

        ![imagen](https://user-images.githubusercontent.com/86111238/171551324-36f1d9bc-0eb3-4481-94cd-12952f22a8e9.png)

   * Representación y reducción dimensional:

     - Hacer una descomposición en componentes principales del dataset (PCA)

      Para implementarla basta con usar la función: *decomposition.PCA()*, la cual dota a un objeto para que pueda hacer descomposiciones en componentes principales, tal y como se ve en el codigo:

         pca = decomposition.PCA(n_components = x , whiten = True , svd_solver = 'auto')
         pca.fit( Xn_t )
         PCA_Xn_t = pca.transform( Xn_t )

      Siendo x la cantidad de componentes principales que deseamos.

     - Evaluar si este se viera beneficiado por reducir su dimensionalidad: "Tomar los valores propios y dividirlos por la suma de valores propios para hallar la varianza explicada"

      Para hallar la varianza explicada usamos la caracteristica del objeto "pca" anteriormente definido: *pca.explained_variance_ratio_* los cuales al sumarlos nos da la varianza explicada:
      
        ![imagen](https://user-images.githubusercontent.com/86111238/171552513-b00a0f8e-a014-4218-8e33-aeb9a74361b5.png)

      En el enunciado se nos menciona: "usualmente mayor a 97% pero puede subir hasta 99.9% en algunos casos.", lo que viene a decir que lo óptimo es que se haga reducción dimensional para valores por encima del 97%, un valor bastante cercano al que usualmente entra como regla para hacer una reducción dimensional, de modo que aplicamos PCA para 5 componentes.

### Implementación de los métodos

Para resolver el problema se utilizó 3 tipos de métodos:

  * Maquina de soporte Vectorial
  * KNN
  * Regresión Logistica

#### Maquina de soporte vectorial:

Para usar maquinas de soporte vectorial se usa la función: *svm.SVC()*, donde variaran los parametros de "kernel" y "gamma", para luego entrenarlo con el conjunto de entrenamiento que determinemos.

    #  Definición del método de maquina de soporte vectorial
    msv = svm.SVC( kernel = 'linear' , gamma='auto' , random_state=895 )
    
    # Entrenamiento del método
    msv.fit( Xn_t , Y_train )

#### KNN:

Para KNN usamos la función: *KNeighborsClassifier()*, la cual dota a un objeto de las caracteristicas del método, cuyo parametro mas importante es de la cantidad de "K vecinos mas cercanos" para el entrenamiento.

    # Definición del método KNN
    knn = KNeighborsClassifier( 10 , weights = 'uniform' , metric = 'euclidean' , metric_params=None , algorithm='auto' )

    # Entrenamiento del método
    knn.fit( Xn_t , Y_train )

#### Regresión logísticas:

Para la regresión logística usamos la función: *KNeighborsClassifier()*, la cual dota a un objeto de las caracteristicas del método

    # Definición del método de regresión logistica
    LR = LogisticRegression(penalty='l2',max_iter=1000, C=10000,random_state=0)
    
    # Entrenamiento del método
    LR.fit(Xn_t, Y_train)

### Selección de métricas de evaluación y optimización de hiperparámetros:

Antes de entrar en la evaluación, determinamos las predicciones:

  * Maquina de soporte vectorial:

        Y_test_predicted_msv = msv.predict( Xn_te )

  * KNN
        
        Y_test_predicted_knn = knn.predict( Xn_te )

  * Regresión logistica

        Y_test_predicted_LR = LR.predict( Xn_te )

#### Selección de metricas de evaluación

   * MCC
   * F1
   * AUC-ROC

##### MCC

La metrica de evaluación MCC se usa con la función: *matthews_corrcoef()*, donde "Y_test_predicted" son las predicciones hechas por el modelo dado por el método, el cual comparamos con "Y_test", cuales son las respuestas correctas para los datos de prueba dados

    MCC_msv = matthews_corrcoef( Y_test , Y_test_predicted_msv )

##### F1

La metrica de evaluación F1 se usa con la función: *f1_score()*, donde "Y_test_predicted" son las predicciones hechas por el modelo dado por el método, el cual comparamos con "Y_test", cuales son las respuestas correctas para los datos de prueba dados

    f1_score_msv = f1_score(Y_test, Y_test_predicted_msv, average='micro')

##### AUC-ROC

Para esta métrica de evaluación si existe una serie de pasos mayor, primeramente se obtienen los puntajes a partir de la caracteristica: *.decision_function()* de cada uno de los metodos (KNN no posee esta característica).

    Y_test_scores_msv = msv.decision_function( Xn_te )
    
Dado que a partir de estos se crea los valores de la curva ROC con la función: *roc_curve()*, la cual trabaja con los falsos positivos y los verdaderos positivos para crear la grafica que se vera en resultados.

#### Optimización de hiperparametros

   * Grid Search

La implementación de Grid Search no es complicada, se buscaran los mejores hiperparametros para los métodos de **Maquina de soporte vectorial** y **KNN**, dado a que en palabras de la guia: "La regresión logística tampoco tiene mucho que iterar aparte de la regularización.".
Para ello primeramente definimos un "param_grid", el cual contiene las caracteristicas de las funciones, las cuales Grid search nos dira cual es la mejor combinación posible.

    param_grid = [{ "kernel": ["rbf"], "gamma": [1e-3,1e-4], "C": [1,10,100,1000] },{ "kernel": ["linear"], "C": [1,10,100,1000] },]
    param_grid = { 'n_neighbors':(k_range), 'weights':['uniform', 'distance'], 'metric':['euclidean','manhattan'] }          

Para luego ingresarla en la función: *GridSearchCV()*, encargada de hacer todo el procesamiento de las diferentes variables. 

    gridSearchCV_SVC = GridSearchCV( estimator = SVC(), param_grid = param_grid, cv = 3, scoring = "f1_micro", refit = True, return_train_score = False,)

## Resultados (puede poner unas imagenes de una tabla de resultados en el README.md)

Los resultados decidi dividirlos en varias partes dado a que me parece pertinente mencionar como cambian los resultados con muy pequeños cambios.

1. Modelos antes y despues de PCA (Los resultados basicos)

Uno de los resultados que me parecio mas interesantes fue como cambia toda la creación del sistema con y sin PCA, donde a pesar de que quitar un sólo componente principal, hiciese que disminuyera sólo en un ~0.04%, volvio al sistema muchisimo mas ineficiente.

   * Antes de PCA

        ![imagen](https://user-images.githubusercontent.com/86111238/171601774-acf18da1-3684-41c2-9e38-d390111cf271.png)
        ![imagen](https://user-images.githubusercontent.com/86111238/171605252-98b2aa92-2cbd-4b56-8d3f-9693fb667883.png)
        ![imagen](https://user-images.githubusercontent.com/86111238/171601707-7f66d482-3391-4b4c-ab8f-c621e333e3ef.png)


   * Despues de PCA

      * Para 6 componenetes (Todos los componentes):

        ![imagen](https://user-images.githubusercontent.com/86111238/171602383-b6e472c8-5266-4d79-b1ce-23d9c9fa3255.png)
        ![imagen](https://user-images.githubusercontent.com/86111238/171605092-3a036dbb-04f7-4aca-97db-452ee9d73d30.png)
        ![imagen](https://user-images.githubusercontent.com/86111238/171602745-5333795a-0414-46ad-b972-86537491158a.png)

      * Para 5 componenetes:

        ![imagen](https://user-images.githubusercontent.com/86111238/171601132-5092b940-f90c-4ba9-a132-f169ec8da7f5.png)
        ![imagen](https://user-images.githubusercontent.com/86111238/171605022-57c6c524-80b7-4170-8183-aff90c1706ac.png)
        ![imagen](https://user-images.githubusercontent.com/86111238/171601225-d5a4b72d-e255-4725-8681-556e1365d282.png)

A pesar de quees notable como el uso de PCA afecta fuertemente la calidad del modelo, la comparación entre 5 y 6 componentes es bastante, dando respuesta a la pregunta: "Evaluar si este se viera beneficiado por reducir su dimensionalidad", donde la respuesta contundente es NO, NO SE VERA BENEFICIADO POR REDUCIR SU DIMENSIONALIDAD.

2. Comparación entre los métodos

Los resultados obtenidos fueron dados para PCA con 5 componentes

   * Resultados de comparación por evaluadores:

        ![imagen](https://user-images.githubusercontent.com/86111238/171601132-5092b940-f90c-4ba9-a132-f169ec8da7f5.png)
        ![imagen](https://user-images.githubusercontent.com/86111238/171605008-9bc480cb-f2c0-4919-be6a-a9791d3ed732.png)
        ![imagen](https://user-images.githubusercontent.com/86111238/171601225-d5a4b72d-e255-4725-8681-556e1365d282.png)

3. Implementación del Grid Search

Grid search nos indica:
   * cambiar los valores de la MSV por: {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
   * cambiar los valores de KNN por: {'metric': 'manhattan', 'n_neighbors': 25, 'weights': 'distance'}

        - Resultados:

        ![imagen](https://user-images.githubusercontent.com/86111238/171606271-90a3440f-ef28-4d07-98aa-742374df8ced.png)

## Conclusiones

1. *La que considero la conclusión mas importante, es que para este caso el uso de PCA afecta inmensamente el modelo, inclusive luego de hacer PCA el sólo reducir una dimensión de este mismo lo afecta en mas de un 10%, dandonos a entender que todas y cada una de los datos son escenciales para un modelo predictivo eficaz.*
2. *El mejor modelo con seguridad es el de regresión logistica, para cualquiera de los métodos de evaluación siempre obtuvo las mejores notas, pudiendo indicar que para datasets de pocas Colummnas es el modelo que mas se ajusta a una predicción eficaz.*
3. *EL MCC nos indico la gran inviabilidad de la PCA, donde sus valores oscilaban el 0 y llegando a negativo demostrando que practicamente el modelo ya estaba era adivinando los resultados dadas las entradas.*


## Link al video de youtube en el readme.md


