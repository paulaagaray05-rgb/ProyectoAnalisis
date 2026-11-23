import os
import hashlib
import pandas as pd
import pydicom
import matplotlib
matplotlib.use("Agg")  # Necesario para guardar gráficas al ejecutar como script .py
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Rutas del dataset en Kaggle
DATA_DIR = "/kaggle/input/competitions/rsna-pneumonia-detection-challenge"
LABELS_PATH = os.path.join(DATA_DIR, "stage_2_train_labels.csv")
IMG_DIR = os.path.join(DATA_DIR, "stage_2_train_images")
HASH_PATH = "/kaggle/working/hashes.csv"
FIG_DUP_PATH = "/kaggle/working/duplicados_visual.png"


def hash_imagen(patient_id):
    """Devuelve el hash del contenido de la imagen (píxeles), o None si no se puede leer."""
    try:
        ds = pydicom.dcmread(os.path.join(IMG_DIR, f"{patient_id}.dcm"))
        arr = ds.pixel_array
        return patient_id, hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()
    except Exception:
        return patient_id, None


def visualizar_duplicados(dup, max_grupos=4):
    """Muestra y guarda en PNG pares/grupos de imágenes con el mismo hash."""
    grupos = list(dup.groupby("hash"))[:max_grupos]
    n_filas = len(grupos)
    n_cols = max(len(g) for _, g in grupos)

    fig, axes = plt.subplots(n_filas, n_cols, figsize=(4.5 * n_cols, 4.5 * n_filas))
    if n_filas == 1:
        axes = [axes]

    for fila_idx, (h, grupo) in enumerate(grupos):
        for col_idx in range(n_cols):
            ax = axes[fila_idx][col_idx] if n_cols > 1 else axes[fila_idx]
            if col_idx < len(grupo):
                row = grupo.iloc[col_idx]
                pid = row["patientId"]
                target = row["Target"]
                estado = "CONSERVADA" if col_idx == 0 else "EXCLUIDA"
                color_titulo = "green" if col_idx == 0 else "red"

                img = pydicom.dcmread(os.path.join(IMG_DIR, f"{pid}.dcm")).pixel_array
                ax.imshow(img, cmap="gray")
                ax.set_title(
                    f"[{estado}]\nID: {pid[:10]}...\nTarget: {target} | Hash: {h[:8]}",
                    color=color_titulo,
                    fontsize=9,
                    fontweight="bold"
                )
            ax.axis("off")

    plt.suptitle("Demostración de imágenes duplicadas (mismo hash MD5, distinto patientId)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DUP_PATH, dpi=150)
    plt.close()
    print(f"Gráfica de duplicados guardada en: {FIG_DUP_PATH}")


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
        # Generar imagen visual de demostración
        visualizar_duplicados(dup, max_grupos=4)

    # 4. Excluir copias duplicadas (se conserva la primera de cada grupo)
    excluir = dup[dup.duplicated("hash", keep="first")]["patientId"].tolist()
    df_limpio = df_patients[~df_patients["patientId"].isin(excluir)]
    print(f"\nExcluidas: {len(excluir)} imagen(es), quedan {len(df_limpio)} pacientes")

    # 5. Verificación explícita tras la limpieza
    hashes_limpios = df_limpio.merge(hashes, on="patientId")
    dup_restantes = hashes_limpios["hash"].duplicated().sum()
    print(f"VERIFICACIÓN: Hashes duplicados restantes en dataset limpio (debe ser 0): {dup_restantes}")

    # Guardar la tabla limpia para usarla en la partición
    df_limpio.to_csv("/kaggle/working/patients_clean.csv", index=False)


if __name__ == "__main__":
    main()