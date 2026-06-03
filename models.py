## 
# 

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

import csv
import glob
import os

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = tf.keras.Sequential([

        layers.Input(shape=(520, 3, 1)),

        layers.Conv2D(filters= 16, 
                      kernel_size=(3, 3), 
                      activation='relu', 
                      padding='valid', 
                      strides=1,
                      name="CONV1_16"),
        
        layers.MaxPooling2D(pool_size=(2,1),
                            strides= None,
                            padding= 'same'),

        layers.Conv2D(filters=32, 
                      kernel_size = (3, 1), 
                      activation='relu', 
                      padding='valid', 
                      strides=1,
                      name="CONV2_32"),
        
        layers.MaxPooling2D(pool_size=(2,1),
                        strides= None,
                        padding= 'same'),

        layers.Conv2D(filters= 64,
                      kernel_size= (3, 1), 
                      activation='relu', 
                      padding='valid', 
                      strides=1,
                      name="CONV3_64"),
        
        layers.MaxPooling2D(pool_size=(2,1),
                        strides= None,
                        padding= 'same'),
        
        layers.Conv2D(filters=128,
                      kernel_size= (3, 1), 
                      activation='relu', 
                      padding='valid',
                      strides=1,
                      name="CONV4_128"),

        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(62, 1, 128)),
            
            layers.Conv2DTranspose(
                filters = 64, 
                kernel_size = (3, 1), 
                activation='relu', 
                padding='valid', 
                strides=1,
                name="DECONV1"),    
            
            layers.UpSampling2D(size=(2, 1)),
            layers.Cropping2D(cropping=((1, 0), (0, 0))), # Ajuste por dimensiones impares

            layers.Conv2DTranspose(
                filters = 32, 
                kernel_size = (3, 1), 
                activation='relu', 
                padding='valid', 
                strides=1,
                name="DECONV2"),

            layers.UpSampling2D(size=(2, 1)),
            layers.Cropping2D(cropping=((1, 0), (0, 0))),

            layers.Conv2DTranspose(
                filters = 16,
                kernel_size = (3, 1), 
                activation='relu', 
                padding='valid', 
                strides=1,
                name="DECONV3"),

            layers.UpSampling2D(size=(2, 1)), 
            
            layers.Conv2DTranspose(
                filters = 1,
                kernel_size = (3, 3), 
                activation='linear', 
                padding='valid', 
                strides=1,
                name="DECONV4"),
    ])


    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def call_encoder(self,x):
        encoded = self.encoder(x)
        return encoded

    def set_coef(self, ruta_csv, enc_dec):
        """
        Carga los coeficientes (filtros y sesgos) desde un archivo CSV y los inyecta
        en las capas correspondientes del modelo de Keras proporcionado.

        Parámetros:
        - ruta_csv: String con la ruta al archivo CSV que contiene los coeficientes.
        - enc_dec: String "encoder" / "decoder", indica que coeficientes se quieren cargar.
        """
        pesos_reconstruidos = {}

        print(f"Leyendo coeficientes desde: {ruta_csv}...")

        # 1. Leer el CSV y reconstruir los tensores
        try:
            with open(ruta_csv, 'r') as f:
                reader = csv.reader(f)
                for fila in reader:
                    if not fila:
                        continue  # Saltar filas vacías si las hay
                        
                    nombre_capa = fila[0]
                    tipo = fila[1]
                    
                    # Inicializar la lista [filtros, sesgos] para la capa si no existe
                    if nombre_capa not in pesos_reconstruidos:
                        pesos_reconstruidos[nombre_capa] = [None, None] 
                        
                    if tipo == 'filtros':
                        # Los índices 2 al 5 contienen la forma (alto, ancho, prof, num_filtros)
                        forma = tuple(map(int, fila[2:6]))
                        # Los valores empiezan en el índice 6
                        valores = np.array(fila[6:], dtype=float).reshape(forma)
                        pesos_reconstruidos[nombre_capa][0] = valores
                        
                    elif tipo == 'sesgos':
                        # El índice 2 contiene la forma (num_filtros,)
                        forma = tuple(map(int, fila[2:3]))
                        # Los valores empiezan en el índice 3
                        valores = np.array(fila[3:], dtype=float).reshape(forma)
                        pesos_reconstruidos[nombre_capa][1] = valores
                        
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo '{ruta_csv}'.")
            return

        # 2. Inyectar los pesos reconstruidos en las capas del modelo
        capas_actualizadas = 0
        if enc_dec == 'encoder':
            layers = self.encoder.layers
        elif enc_dec == 'decoder':
            layers = self.decoder.layers

        for capa in layers:
            if capa.name in pesos_reconstruidos:
                # Obtenemos la lista [filtros, sesgos]
                nuevos_pesos = pesos_reconstruidos[capa.name]
                
                # Verificación de seguridad: asegurarnos de que tenemos ambos
                if nuevos_pesos[0] is not None and nuevos_pesos[1] is not None:
                    capa.set_weights(nuevos_pesos)
                    print(f" [+] Coeficientes cargados con éxito en la capa: {capa.name}")
                    capas_actualizadas += 1
                else:
                    print(f" [!] Advertencia: Faltan filtros o sesgos para la capa {capa.name} en el CSV.")

        print(f"\nProceso finalizado. Se actualizaron {capas_actualizadas} capas.")


## Clase CNN

class NN(Model):
    def __init__(self):
        super(NN, self).__init__()

        self.modelNN = tf.keras.Sequential([
                layers.Flatten(input_shape=(62, 1, 128)),
                layers.Dense(10, activation = 'relu', kernel_regularizer = regularizers.L2(0.01)),
                layers.Dense(4, activation = 'relu', kernel_regularizer = regularizers.L2(0.01)),
                layers.Dense(units=1,activation="sigmoid", kernel_regularizer = regularizers.L2(0.01)),
        ])

    def call(self, x):
        return self.redCnn(x)



