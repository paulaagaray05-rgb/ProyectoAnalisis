import os
import pandas as pd

# Ruta del dataset en Kaggle
DATA_DIR = "/kaggle/input/competitions/rsna-pneumonia-detection-challenge"
LABELS_PATH = os.path.join(DATA_DIR, "stage_2_train_labels.csv")

def main():
    # 1. Cargar el CSV de etiquetas
    df = pd.read_csv(LABELS_PATH)
    print(f"Filas totales en el CSV (incluyendo múltiples bounding boxes): {len(df)}")

    # 2. Agrupar por patientId (un paciente con varias cajas solo cuenta una vez)
    df_patients = df.groupby("patientId")["Target"].max().reset_index()
    print(f"Imágenes/pacientes únicos: {len(df_patients)}\n")

    # 3. Calcular y mostrar el balanceo de clases
    counts = df_patients["Target"].value_counts().sort_index()
    percentages = df_patients["Target"].value_counts(normalize=True).sort_index() * 100

    print("--- BALANCEO DE CLASES (Clasificación Binaria) ---")
    print(f"Clase 0 (No Neumonía): {counts[0]} imágenes ({percentages[0]:.2f}%)")
    print(f"Clase 1 (Neumonía):    {counts[1]} imágenes ({percentages[1]:.2f}%)")
    print(f"Ratio negativo/positivo: {counts[0] / counts[1]:.2f}:1")

if __name__ == "__main__":
    main()