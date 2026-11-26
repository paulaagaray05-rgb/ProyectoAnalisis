import os
import pydicom
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SEED = 42
DATA_DIR = "/kaggle/input/competitions/rsna-pneumonia-detection-challenge"
LABELS_PATH = os.path.join(DATA_DIR, "stage_2_train_labels.csv")
IMG_DIR = os.path.join(DATA_DIR, "stage_2_train_images")

# Cargar CSV original (para tener también las coordenadas x, y, width, height de las cajas)
df_raw = pd.read_csv(LABELS_PATH)
df_patients = df_raw.groupby("patientId")["Target"].max().reset_index()

# 4 muestras aleatorias SIN neumonía (arriba) y 4 CON neumonía (abajo)
no_neumonia = df_patients[df_patients["Target"] == 0].sample(4, random_state=SEED)
si_neumonia = df_patients[df_patients["Target"] == 1].sample(4, random_state=SEED)
samples = pd.concat([no_neumonia, si_neumonia])

fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))

for ax, (_, row) in zip(axes.flat, samples.iterrows()):
    pid = row["patientId"]
    target = row["Target"]

    # Leer imagen DICOM
    dcm_path = os.path.join(IMG_DIR, f"{pid}.dcm")
    img = pydicom.dcmread(dcm_path).pixel_array
    ax.imshow(img, cmap="gray")

    # Si tiene neumonía, dibujar todas sus bounding boxes en rojo
    if target == 1:
        cajas = df_raw[df_raw["patientId"] == pid]
        for _, caja in cajas.iterrows():
            if pd.notna(caja["x"]):
                rect = patches.Rectangle(
                    (caja["x"], caja["y"]), caja["width"], caja["height"],
                    linewidth=2, edgecolor="red", facecolor="none"
                )
                ax.add_patch(rect)
        ax.set_title(f"Neumonía (1)\nID: {pid[:8]}...", color="crimson", fontsize=9, fontweight="bold")
    else:
        ax.set_title(f"No Neumonía (0)\nID: {pid[:8]}...", color="darkgreen", fontsize=9, fontweight="bold")

    ax.axis("off")

plt.suptitle("Muestras aleatorias del dataset RSNA (Fila 1: Clase 0 | Fila 2: Clase 1 con opacidades)", fontsize=11)
plt.tight_layout()
plt.savefig("/kaggle/working/muestras_clases.png", dpi=150)
plt.show()
