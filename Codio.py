# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')






import kagglehub

# Download latest version
path = kagglehub.competition_download('rsna-pneumonia-detection-challenge')

print("Path to competition files:", path)





import os

path = "/kaggle/input/competitions/rsna-pneumonia-detection-challenge"

for item in sorted(os.listdir(path)):
    full = os.path.join(path, item)
    if os.path.isdir(full):
        print(f"[carpeta] {item}  ({len(os.listdir(full))} elementos)")
    else:
        print(f"[archivo] {item}  ({os.path.getsize(full) / 1e6:.1f} MB)")






import pandas as pd

labels = pd.read_csv(f"{path}/stage_2_train_labels.csv")
classes = pd.read_csv(f"{path}/stage_2_detailed_class_info.csv")

print(labels.head())
labels.info()
print()
print("Filas:", len(labels))
print("Pacientes únicos:", labels["patientId"].nunique())
print()
print(classes["class"].value_counts())








!pip install -q pydicom

import pydicom
import matplotlib.pyplot as plt

pid = labels["patientId"].iloc[0]
ds = pydicom.dcmread(f"{path}/stage_2_train_images/{pid}.dcm")

print("Interpretación fotométrica:", ds.PhotometricInterpretation)
print("Tamaño:", ds.Rows, "x", ds.Columns)
print("Bits almacenados:", ds.BitsStored)
print()
print(ds)

plt.imshow(ds.pixel_array, cmap="gray")
plt.axis("off")
plt.show()






import pandas as pd
from sklearn.model_selection import train_test_split

path = "/kaggle/input/competitions/rsna-pneumonia-detection-challenge"

labels = pd.read_csv(f"{path}/stage_2_train_labels.csv")
classes = pd.read_csv(f"{path}/stage_2_detailed_class_info.csv")

# Una fila por paciente: Target = 1 si tiene alguna opacidad
patients = labels.groupby("patientId")["Target"].max().reset_index()

# Clase original (Normal, No Lung Opacity / Not Normal, Lung Opacity)
classes_pat = classes.drop_duplicates("patientId")[["patientId", "class"]]
patients = patients.merge(classes_pat, on="patientId")

print("Pacientes:", len(patients))
print(patients["Target"].value_counts())
print(patients["Target"].value_counts(normalize=True).round(4))
print()
print(patients["class"].value_counts())

# Partición estratificada 80/10/10 con semilla fija
train, temp = train_test_split(
    patients, test_size=0.20, stratify=patients["Target"], random_state=42
)
val, test = train_test_split(
    temp, test_size=0.50, stratify=temp["Target"], random_state=42
)

for nombre, df in [("train", train), ("val", val), ("test", test)]:
    print(nombre, len(df), "positivos:", round(df["Target"].mean(), 4))
    df.to_csv(f"/kaggle/working/{nombre}.csv", index=False)











