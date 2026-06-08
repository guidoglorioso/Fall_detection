import csv
import glob
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
## Levanto el dataset (unicamente los datos de acelerometro)

def get_columnas_csv(path, file):
    
    patron_archivos = os.path.join(path, file)

    # glob.glob devuelve una lista con las rutas de todos los archivos que coincidan
    archivo  = glob.glob(patron_archivos)[0]

    df_temporal = pd.read_csv(archivo, skiprows=5)

    return df_temporal.columns.tolist()


def extraer_datos_csv(path, file, sensors):
    patron_archivos = os.path.join(path, file)

    col = get_columnas_csv(path, file)
    
    for sen in sensors:
        if sen not in col:
            raise ValueError(f"El sensor {sen} no esta disponible en el archivo indicado.")
        
    # glob.glob devuelve una lista con las rutas de todos los archivos que coincidan
    lista_archivos = sorted(glob.glob(patron_archivos))

    data = []

    for archivo in lista_archivos:
        # skiprows=5 salta el bloque de metadatos
        df_temporal = pd.read_csv(
            archivo, 
            skiprows=5, 
            usecols=sensors
        )
        sensores = df_temporal[sensors].values

        data.append(sensores)

    return np.array(data)
    
def extraer_tags_csv(path, file):

    patron_archivos = os.path.join(path, file)

    # glob.glob devuelve una lista con las rutas de todos los archivos que coincidan
    lista_archivos = sorted(glob.glob(patron_archivos))

    tags = []

    for archivo in lista_archivos:
    
        with open(archivo, 'r') as file:
            encabezado = [next(file).strip() for _ in range(5)]  
            for e in encabezado:
                if "TYPE" not in e:
                    continue

                if "NOT FALL" in e :
                    tags.append(0)
                elif "FALL" in e:
                    tags.append(1)
                else:
                    print(f"No se dectecto el TAG del dato en el archivo: {archivo}. ")
    return np.array(tags)

class dataScaler:
    def __init__(self):
        self.num_sensores = 0
        self.scaler = StandardScaler()

            

    def escalar_datos(self,datos):
        """
        Escala datos 3D (ejemplos, muestras, sensores) aplicando un StandardScaler 
        por cada sensor considerando todos los ejemplos a la vez.
        """

        num_ejemplos, num_muestras, num_sensores = datos.shape
        
        
        datos_2d = datos.reshape(-1, num_sensores)
            
        if not hasattr(self.scaler, 'mean_') or num_sensores == self.num_sensores:
            self.set_escala(datos)

        datos_escalados_2d = self.scaler.transform(datos_2d)
        
        datos_escalados = datos_escalados_2d.reshape(num_ejemplos, num_muestras, num_sensores)
        
        return datos_escalados
    
    def set_escala(self, datos):
        
        num_ejemplos, num_muestras, self.num_sensores = datos.shape
        
        # Reformateamos a 2D: (ejemplos * muestras, sensores)
        # Cada columna ahora es TODA la data histórica de un solo sensor (X, Y o Z)
        datos_2d = datos.reshape(-1, self.num_sensores)
        
        #Inicializamos y entrenamos el escalador
        
        self.scaler.fit(datos_2d)


def plot_stats(history):
    loss_entrenamiento = history.history['loss']
    loss_validacion = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))


    ax1.plot(loss_entrenamiento, label='Entrenamiento (Loss)', color='blue')
    ax1.plot(loss_validacion, label='Validación (Val Loss)', color='red')
    ax1.set_title('Evolución del Error (Loss) durante el Entrenamiento')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Pérdida (Binary Crossentropy)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(acc, label='Entrenamiento (Acc)', color='blue')
    # Corregí el label de 'Acc Loss' a 'Val Acc'
    ax2.plot(val_acc, label='Validación (Val Acc)', color='red') 
    ax2.set_title('Evolución de la Precisión (Accuracy) durante el Entrenamiento')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Precisión [%]')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout() 
    
    return fig

def model_perfom(model,dataset,tags, threshold = 0.5):
    
    y_pred_prob = model.predict(dataset)


    y_pred = (y_pred_prob > threshold).astype(int)


    print("=== REPORTE DE CLASIFICACIÓN ===")
    print(classification_report(tags, y_pred, target_names=["NOT FALL (0)", "FALL (1)"]))

    # 4. Calculamos la matriz de confusión
    matriz = confusion_matrix(tags, y_pred)
    print("\n=== MATRIZ DE CONFUSIÓN ===")
    print("                  Predice NO CAÍDA   Predice CAÍDA")
    print(f"Real NO CAÍDA:    {matriz[0][0]}                 {matriz[0][1]}")
    print(f"Real CAÍDA:       {matriz[1][0]}                 {matriz[1][1]}")



######### Extraccion de features para modelo de sensor barometrico


def median_filter(window , data):

    output = []

    for indx in range(window-1,len(data)):

        sorted_data = sorted(data[indx - window + 1: indx])
        output.append(sorted_data[window//2])

    return output

def calcular_altura(data ,window_filter = 10):
    data_output = []

    kernel = np.ones(window_filter) / window_filter

    for _data in data:
        sensor_temp =_data[: , 1]   
        sensor_temp = sensor_temp[~np.isnan(sensor_temp)]
        sensor_presion = _data[: , 0]    
        sensor_presion = sensor_presion[~np.isnan(sensor_presion)] 
        altura = ((sensor_temp + 273.15) / 0.0065) * (1 - (sensor_presion / 1013.25)**0.19)
        altura = median_filter(window_filter,altura)
        #altura = np.convolve(altura, kernel, mode='valid') ## similar a 
        #altura = pd.Series(altura)
        #altura = altura.rolling(window=window_filter).median()
        
        # data_output.append(altura[window_filter:])

        data_output.append(altura)
    return np.array(data_output)


## Features

# min_indice = 0 
# max_indice = 1
# dif_altura_indice = 2
# min_ubicacion_indice = 3
# max_ubicacion_indice = 4
# desvio_indice = 5
# asimetria_indice = 6
# curtosis_indice = 7
# rms_indice = 8
# percentil90_indice = 9
# percentil10_indice = 10

# Calculo las features de la señal y las agrupo en una matriz.
import numpy as np

def calcular_asimetria(señales, axis=1):
    """
    Calcula la asimetría (skewness) de un conjunto de señales.
    Se suma un epsilon (1e-10) a la desviación estándar para evitar divisiones por cero 
    en caso de que alguna señal sea una línea recta constante.
    """
    mu = np.mean(señales, axis=axis, keepdims=True)
    sigma = np.std(señales, axis=axis, keepdims=True) + 1e-10 
    
    # E[(X - mu / sigma)^3]
    asimetria = np.mean(((señales - mu) / sigma)**3, axis=axis)
    return asimetria

def calcular_curtosis(señales, axis=1):
    """
    Calcula el exceso de curtosis de un conjunto de señales.
    Una distribución normal perfecta devolverá 0.
    """
    mu = np.mean(señales, axis=axis, keepdims=True)
    sigma = np.std(señales, axis=axis, keepdims=True) + 1e-10
    
    # E[(X - mu / sigma)^4] - 3
    curtosis = np.mean(((señales - mu) / sigma)**4, axis=axis) - 3.0
    return curtosis

def extraer_features(señales):
    """
    Extrae 11 features de una matriz de señales [numero_señales, muestras].
    Retorna una matriz de dimensiones [numero_señales, 11].
    """
    # 0. min
    f_min = np.min(señales, axis=1)
    
    # 1. max
    f_max = np.max(señales, axis=1)
    
    # 2. dif_altura
    f_dif_altura = f_max - f_min
    
    # 3. min_ubicacion
    f_min_ubicacion = np.argmin(señales, axis=1)
    
    # 4. max_ubicacion
    f_max_ubicacion = np.argmax(señales, axis=1)
    
    # 5. desvio
    f_desvio = np.std(señales, axis=1)
    
    # 6. asimetria
    f_asimetria = calcular_asimetria(señales, axis=1)
    
    # 7. curtosis
    f_curtosis = calcular_curtosis(señales, axis=1)
    
    # 8. rms
    f_rms = np.sqrt(np.mean(señales**2, axis=1))
    
    # 9. percentil90
    f_percentil90 = np.percentile(señales, 90, axis=1)
    
    # 10. percentil10
    f_percentil10 = np.percentile(señales, 10, axis=1)
    
    # Agrupación en la matriz final de salida [numero_señales, 11]
    # np.column_stack toma arreglos 1D y los apila como columnas de una matriz 2D
    matriz_features = np.column_stack((
        f_min,
        f_max,
        f_dif_altura,
        f_min_ubicacion,
        f_max_ubicacion,
        f_desvio,
        f_asimetria,
        f_curtosis,
        f_rms,
        f_percentil90,
        f_percentil10
    ))
    
    return matriz_features


    