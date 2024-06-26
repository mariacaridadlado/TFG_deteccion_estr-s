# TFG_detección_estres
Este repositorio contiene el código y los scripts utilizados en mi Trabajo de Fin de Grado (TFG) titulado "Implementación de Clasificadores para el Reconocimiento de Estrés Basado en Bioseñales". El objetivo principal de este proyecto es desarrollar y evaluar diferentes algoritmos de aprendizaje automático y profundo para la detección de estrés a partir de señales electrodermales (EDA) obtenidas de los datasets AMIGOS y WESAD.
# Contenido:
1. **AprendizajeAutomatico_AMIGOS.ipynb**: Notebook que contiene la implementación de diferentes clasificadores de aprendizaje automático, utilizando señales EDA del dataset AMIGOS.
2. **AprendizajeAutomatico_WESAD.ipynb**: Notebook que contiene la implementación de diferentes clasificadores de aprendizaje automático, utilizando señales EDA del dataset WESAD.
3. **PreprocesamientoCNN.ipynb**: Notebook que detalla el preprocesamiento de datos para la implementación de redes neuronales, aplicando técnicas de preprocesamiento tanto en el dataset AMIGOS como en el WESAD.
4. **REDCNN_TFG.ipynb**: Notebook que contiene la implementación de una red neuronal convolucional (CNN) y otra que combina la red neuronal convolucional con las LSTM para la clasificación de estrés basado en señales EDA, utilizando ambos datasets.
5. **lstm.py**: Módulo que contiene la implementación de una red neuronal que combina capas convolucionales y una LSTM para la detección de estrés a partir de señales EDA. Este módulo incluye funciones para preprocesamiento de datos, creación del modelo, entrenamiento y evaluación del modelo.

## Tecnologías y Herramientas:

- Python
- Jupyter Notebooks
- Bibliotecas: NumPy, Pandas, Scikit-learn, TensorFlow, Keras, Matplotlib, Seaborn

## Descripción del modulo 'lstm.py':
El módulo lstm.py incluye las siguientes clases y funciones:
### Clases:
1. **Net** : Define la estructura de la red neuronal que combina capas convolucionales (CNN) y una capa LSTM para la clasificación de estrés.

    **init**: Inicializa las capas de la red neuronal.
    **forward**: Define el paso hacia adelante de la red neuronal.
2. **MiDataset** : Clase personalizada para manejar el dataset.

    **init**: Inicializa el dataset con datos y etiquetas.
    **len**: Devuelve el número de muestras en el dataset.
    **getitem**: Devuelve una muestra y su etiqueta correspondiente.
### Funciones:
1. **cargar_datos**: Carga los datos de EDA desde archivos CSV.
2. **preprocesar_datos**: Preprocesa las señales EDA, aplica transformaciones y guarda los datos preprocesados en un archivo CSV.
3. **crear_modelo_lstm**: Crea un modelo LSTM con los parámetros especificados.
4. **entrenar_modelo**: Entrena el modelo LSTM utilizando datos preprocesados.
5. **cargar_modelo**: Carga un modelo LSTM previamente entrenado.
6. **evaluar_modelo**: Evalúa un modelo LSTM en un sujeto específico y calcula la exactitud.
### Uso:
1. **Preprocesar los datos**:

   `lstm.preprocesar_datos(4,'/content/drive/MyDrive/TFG/WESAD/EDA-','/content/drive/MyDrive/TFG/',columna_eda=0,sujetos=[3,4])`

2. **Crear y entrenar el modelo**:
   
   `modelo = crear_modelo_lstm(hidden_size=200, num_layers=1)`
   
   `modelo, media_resultados = entrenar_modelo(data, modelo, sujetos, n_epochs=10, batch_size=40, learning_rate=0.001)`

4. **Guardar y cargar el modelo**:

   `torch.save(modelo.state_dict(), 'modelo_lstm.pth')`
   
   `modelo = cargar_modelo('/ruta/modelo/modelo_lstm.pth')`

5. **Evaluar el modelo**:
  `predicted, test_accuracy = evaluar_modelo(modelo, data, sujeto, criterion, batch_size=40)`
   


   




