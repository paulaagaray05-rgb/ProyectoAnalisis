import os
import hashlib
import pandas as pd
import pydicom
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Rutas del dataset en Kaggle
DATA_DIR = "/kaggle/input/competitions/rsna-pneumonia-detection-challenge"
LABELS_PATH = os.path.join(DATA_DIR, "stage_2_train_labels.csv")
IMG_DIR = os.path.join(DATA_DIR, "stage_2_train_images")
HASH_PATH = "/kaggle/working/hashes.csv"


def hash_imagen(patient_id):
    """Devuelve el hash del contenido de la imagen (píxeles), o None si no se puede leer."""
    try:
        ds = pydicom.dcmread(os.path.join(IMG_DIR, f"{patient_id}.dcm"))
        arr = ds.pixel_array
        return patient_id, hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()
    except Exception:
        return patient_id, None


def main():
    # 1. Cargar etiquetas y agrupar por paciente
    df = pd.read_csv(LABELS_PATH)
    df_patients = df.groupby("patientId")["Target"].max().reset_index()
    print(f"Filas totales en el CSV: {len(df)}")
    print(f"Pacientes únicos: {len(df_patients)}")
    print(f"patientId repetidos tras agrupar (debe ser 0): {df_patients['patientId'].duplicated().sum()}\n")

    # 2. Comprobar que coinciden etiquetas y archivos DICOM
    archivos = {f[:-4] for f in os.listdir(IMG_DIR) if f.endswith(".dcm")}
    ids = set(df_patients["patientId"])
    print("--- CONSISTENCIA ETIQUETAS / ARCHIVOS ---")
    print(f"Archivos DICOM sin etiqueta: {len(archivos - ids)}")
    print(f"Etiquetas sin archivo DICOM: {len(ids - archivos)}\n")

    # 3. Calcular el hash de cada imagen (se guarda para no repetirlo)
    if os.path.exists(HASH_PATH):
        hashes = pd.read_csv(HASH_PATH)
    else:
        with ThreadPoolExecutor(max_workers=8) as ex:
            resultados = list(tqdm(ex.map(hash_imagen, sorted(archivos)), total=len(archivos)))
        hashes = pd.DataFrame(resultados, columns=["patientId", "hash"])
        hashes.to_csv(HASH_PATH, index=False)

    print("--- IMÁGENES DUPLICADAS (mismo contenido, distinto patientId) ---")
    print(f"Archivos ilegibles: {hashes['hash'].isna().sum()}")

    dup = hashes.dropna().loc[lambda d: d.duplicated("hash", keep=False)]
    dup = dup.sort_values("hash").merge(df_patients, on="patientId")
    print(f"Imágenes que comparten contenido con otra: {len(dup)}")
    print(f"Grupos de duplicados: {dup['hash'].nunique()}")

    if len(dup) > 0:
        contradictorios = dup.groupby("hash")["Target"].nunique()
        print(f"Grupos con etiquetas contradictorias: {(contradictorios > 1).sum()}")
        print(dup.head(20))


if __name__ == "__main__":
    main()