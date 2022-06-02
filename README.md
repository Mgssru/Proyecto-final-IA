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

## Desarrollo

De las caracteristicas anteriores, se desea predecir lo propensa que es una persona a ser fumadora, es decir, un problema de clasificación, donde 0 -> No Fumador y 1 -> Fumador, para ello se utilizo 3 tipos de métodos:

  * Maquina de soporte Vectorial
  * KNN
  * Regresión Logistica

### Maquina de soporte vectorial:

Para usar maquinas de soporte vectorial 

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
