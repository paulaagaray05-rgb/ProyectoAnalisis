# Importación de librerías necesarias

import os                  # Permite trabajar con archivos y carpetas
import pandas as pd        # Permite crear y manipular tablas (DataFrames)
import pydicom             # Permite leer imágenes médicas en formato DICOM
import numpy as np         # Librería para cálculos numéricos
from tqdm import tqdm      # Barra de progreso para los bucles


# Ruta donde se encuentran las imágenes DICOM del dataset RSNA
IMG_DIR = "/kaggle/input/competitions/rsna-pneumonia-detection-challenge/stage_2_train_images"


# Lista vacía donde se almacenarán los resultados
# Cada elemento contendrá:
# [patientId, intensidad_media]
datos = []


# Recorre todos los archivos de la carpeta
for file in tqdm(os.listdir(IMG_DIR)):

    # Procesar únicamente los archivos DICOM
    if file.endswith(".dcm"):

        # Elimina la extensión .dcm para obtener el identificador del paciente
        patient_id = file[:-4]

        # Lee el archivo DICOM y extrae la matriz de píxeles
        img = pydicom.dcmread(
            os.path.join(IMG_DIR, file)
        ).pixel_array

        # Calcula la intensidad media de todos los píxeles de la imagen
        intensidad_media = img.mean()

        # Guarda el identificador del paciente y su intensidad media
        datos.append([patient_id, intensidad_media])


# Convierte la lista en un DataFrame de pandas
df_int = pd.DataFrame(
    datos,
    columns=["patientId", "intensidad_media"]
)


# Muestra el DataFrame completo
print(df_int)
