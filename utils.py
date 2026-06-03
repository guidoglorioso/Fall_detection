import csv
import glob
import os
import pandas as pd
import numpy as np


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
    lista_archivos = glob.glob(patron_archivos)

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
    
from sklearn.preprocessing import StandardScaler

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