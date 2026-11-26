import os
import math
import numpy as np
import pandas as pd
import pydicom
import matplotlib
matplotlib.use("Agg")  # Quita esta línea si lo ejecutas directamente en celdas de notebook2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Rutas en Kaggle
DATA_DIR = "/kaggle/input/competitions/rsna-pneumonia-detection-challenge"
IMG_DIR = os.path.join(DATA_DIR, "stage_2_train_images")
DETAILED_PATH = os.path.join(DATA_DIR, "stage_2_detailed_class_info.csv")
CLEAN_PATH = "/kaggle/working/patients_clean.csv"  # Generado en el paso anterior

# 1. Cargar pacientes limpios (sin duplicados) y unir la clase detallada
df = pd.read_csv(CLEAN_PATH)
df_class = pd.read_csv(DETAILED_PATH).drop_duplicates("patientId")
df = df.merge(df_class, on="patientId", how="left")


def extraer_metricas(patient_id):
    """Extrae brillo medio, contraste, fracción de negros, nitidez y metadatos DICOM."""
    path = os.path.join(IMG_DIR, f"{patient_id}.dcm")
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    # Normalizar a [0, 255] según profundidad de bits (12/16 bits -> 255) para métricas comparables
    max_val = arr.max() if arr.max() > 0 else 1.0
    arr_255 = (arr / max_val) * 255.0

    # Si está invertida (MONOCHROME1: aire=blanco), lo anotamos y corregimos para medir bien black_frac
    photo = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photo == "MONOCHROME1":
        arr_255 = 255.0 - arr_255

    mean_val = float(np.mean(arr_255))
    std_val = float(np.std(arr_255))
    black_frac = float(np.mean(arr_255 < 10.0))  # Proporción de píxeles casi negros

    # Nitidez (varianza del Laplaciano aproximado por diferencias finitas 2D)
    lap = (
        arr_255[:-2, 1:-1] + arr_255[2:, 1:-1] +
        arr_255[1:-1, :-2] + arr_255[1:-1, 2:] -
        4.0 * arr_255[1:-1, 1:-1]
    )
    sharpness = float(np.var(lap))

    view_pos = getattr(ds, "ViewPosition", "UNK")
    return patient_id, view_pos, photo, mean_val, std_val, black_frac, sharpness


# 2. Calcular características en paralelo (tarda ~25-40 s en Kaggle)
print("Extrayendo características de intensidad y nitidez de los DICOM...")
with ThreadPoolExecutor(max_workers=8) as ex:
    rows = list(tqdm(ex.map(extraer_metricas, df["patientId"]), total=len(df)))

df_metrics = pd.DataFrame(
    rows,
    columns=["patientId", "view_position", "photometric", "mean", "std", "black_frac", "sharpness"]
)
df = df.merge(df_metrics, on="patientId")
df.to_csv("/kaggle/working/dataset_metrics.csv", index=False)

# 3. Detección de outliers mediante Z-score agrupado por proyección (AP vs PA)
features = ["mean", "std", "black_frac", "sharpness"]
grouped = df.groupby("view_position", observed=True)[features]

z = (df[features] - grouped.transform("mean")) / grouped.transform("std")
df["outlier_score"] = z.abs().max(axis=1)
df["is_outlier"] = df["outlier_score"] > 4

# Guardamos también qué característica disparó el outlier
df["motivo_outlier"] = z.abs().idxmax(axis=1)

print(f"\nProyecciones encontradas:\n{df['view_position'].value_counts()}")
print(f"Fotometría (MONOCHROME1 = invertida):\n{df['photometric'].value_counts()}")
print(f"\nImágenes marcadas como atípicas (Z > 4): {df['is_outlier'].sum()} ({df['is_outlier'].mean():.2%})")

# 4. Comparativa visual: Imagen más típica vs más atípica en proyección AP y PA
for view in ["PA", "AP"]:
    subset = df[df["view_position"] == view]
    if len(subset) == 0:
        continue
    typical = subset.loc[subset["outlier_score"].idxmin()]
    atypical = subset.loc[subset["outlier_score"].idxmax()]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, row, title in zip(axes, [typical, atypical], ["Imagen típica", "Imagen atípica"]):
        img = pydicom.dcmread(os.path.join(IMG_DIR, f"{row['patientId']}.dcm")).pixel_array
        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"{title} (score: {row['outlier_score']:.1f})\n"
            f"Clase: {row['class']} | Motivo: {row['motivo_outlier']}",
            fontsize=9
        )
        ax.axis("off")
    fig.suptitle(f"Proyección {view}: Típica vs Atípica")
    plt.tight_layout()
    plt.savefig(f"/kaggle/working/tipica_vs_atipica_{view}.png", dpi=150)
    plt.close()

# 5. Boxplots comparativos (adaptados a proyección y clase clínica)
view_types = [v for v in ["PA", "AP"] if v in df["view_position"].unique()]
class_order = df.groupby("class", observed=True)["mean"].median().sort_values().index

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].boxplot(
    [df.loc[df["view_position"] == v, "mean"] for v in view_types],
    tick_labels=view_types,
    showfliers=False
)
axes[0].set_title("Brillo promedio por proyección (PA vs AP)")
axes[0].set_ylabel("Intensidad normalizada (0-255)")

axes[1].boxplot(
    [df.loc[df["class"] == c, "mean"] for c in class_order],
    tick_labels=class_order,
    showfliers=False
)
axes[1].set_title("Brillo promedio por clase diagnóstica")
axes[1].tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig("/kaggle/working/boxplots_brillo.png", dpi=150)
plt.close()

# 6. MOSTRAR TODAS LAS IMÁGENES ATÍPICAS PARA DECIDIR SI DESCARTARLAS O NO
outliers_df = df[df["is_outlier"]].sort_values("outlier_score", ascending=False).reset_index(drop=True)
outliers_df.to_csv("/kaggle/working/outliers_list.csv", index=False)

n_outliers = len(outliers_df)
if n_outliers == 0:
    print("No se encontraron imágenes con outlier_score > 4.")
else:
    cols = 5
    filas_por_pagina = 6  # 30 imágenes por página PNG para que se vean con claridad
    por_pagina = cols * filas_por_pagina
    n_paginas = math.ceil(n_outliers / por_pagina)

    for pag in range(n_paginas):
        lote = outliers_df.iloc[pag * por_pagina : (pag + 1) * por_pagina]
        n_rows = math.ceil(len(lote) / cols)
        fig, axes = plt.subplots(n_rows, cols, figsize=(3.2 * cols, 3.6 * n_rows))
        axes = np.array(axes).reshape(-1)

        for ax, (_, row) in zip(axes, lote.iterrows()):
            img = pydicom.dcmread(os.path.join(IMG_DIR, f"{row['patientId']}.dcm")).pixel_array
            ax.imshow(img, cmap="gray")
            ax.set_title(
                f"ID: {row['patientId'][:8]}...\n"
                f"{row['view_position']} | T={row['Target']} | z={row['outlier_score']:.1f}\n"
                f"Motivo: {row['motivo_outlier']}",
                fontsize=8
            )
            ax.axis("off")

        for ax in axes[len(lote):]:
            ax.axis("off")

        plt.suptitle(f"Revisión de Outliers (Página {pag + 1}/{n_paginas} — Total: {n_outliers})", fontsize=12)
        plt.tight_layout()
        out_img = f"/kaggle/working/outliers_revision_pag_{pag + 1}.png"
        plt.savefig(out_img, dpi=140)
        plt.close()
        print(f"Guardada cuadrícula de revisión: {out_img}")

