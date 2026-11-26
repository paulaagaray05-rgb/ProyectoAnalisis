import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2

SEED = 42
IMG_SIZE = 224

DATA_DIR = "/kaggle/input/competitions/rsna-pneumonia-detection-challenge"
LABELS_PATH = os.path.join(DATA_DIR, "stage_2_train_labels.csv")
IMG_DIR = os.path.join(DATA_DIR, "stage_2_train_images")
OUTPUT_DIR = "/kaggle/working"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def leer_y_redimensionar_224(dcm_path, size=IMG_SIZE, use_clahe=False):
    """
    Lee un DICOM, corrige MONOCHROME1 si hiciera falta, escala a uint8 [0, 255]
    y redimensiona a (size, size). Devuelve un array 2D (H, W) uint8.
    """
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)

    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = np.max(img) - img

    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    img_u8 = (img * 255).astype(np.uint8)

    img_224 = cv2.resize(img_u8, (size, size), interpolation=cv2.INTER_AREA)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_224 = clahe.apply(img_224)

    return img_224


def normalizar_tensor_224(img_224, mode="scratch"):
    """
    Convierte una imagen 2D (224, 224) uint8 a tensor de 3 canales (3, 224, 224)
    y normaliza según el modo:
      - 'scratch': rango [0.0, 1.0]
      - 'transfer': media y std de ImageNet
    """
    img_3ch = np.stack([img_224, img_224, img_224], axis=0)
    tensor = torch.from_numpy(img_3ch)
    tensor = v2.ToDtype(torch.float32, scale=True)(tensor)

    if mode == "scratch":
        return tensor
    if mode == "transfer":
        return v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(tensor)

    raise ValueError("El modo debe ser 'scratch' o 'transfer'.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(LABELS_PATH).drop_duplicates(subset=["patientId"]).reset_index(drop=True)
    muestras = df.sample(4, random_state=SEED).reset_index(drop=True)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))

    for col, (_, row) in enumerate(muestras.iterrows()):
        dcm_path = os.path.join(IMG_DIR, f"{row['patientId']}.dcm")
        img_224 = leer_y_redimensionar_224(dcm_path, size=IMG_SIZE, use_clahe=False)
        img_224_clahe = leer_y_redimensionar_224(dcm_path, size=IMG_SIZE, use_clahe=True)

        t_scratch = normalizar_tensor_224(img_224, mode="scratch")
        t_transfer = normalizar_tensor_224(img_224_clahe, mode="transfer")

        if col == 0:
            print(f"Forma tensor: {tuple(t_scratch.shape)}")
            print(f"Normalización [0, 1] (scratch)   -> min={t_scratch.min():.3f}, max={t_scratch.max():.3f}")
            print(f"Normalización ImageNet (transfer) -> min={t_transfer.min():.3f}, max={t_transfer.max():.3f}")

        axes[0, col].imshow(img_224, cmap="gray")
        axes[0, col].set_title(f"224x224 sin CLAHE (Target={int(row['Target'])})", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(img_224_clahe, cmap="gray")
        axes[1, col].set_title(f"224x224 con CLAHE (Target={int(row['Target'])})", fontsize=9)
        axes[1, col].axis("off")

    plt.suptitle("Comprobación de redimensionado a 224x224 y CLAHE", fontsize=11)
    plt.tight_layout()
    out_img = os.path.join(OUTPUT_DIR, "normalizacion_224.png")
    plt.savefig(out_img, dpi=150)
    plt.close()
    print(f"Figura guardada en: {out_img}")


if __name__ == "__main__":
    main()