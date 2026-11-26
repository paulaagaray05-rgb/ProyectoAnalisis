import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2

SEED = 42
IMG_SIZE = 224

DATA_DIR = "/kaggle/input/competitions/rsna-pneumonia-detection-challenge"
LABELS_PATH = os.path.join(DATA_DIR, "stage_2_train_labels.csv")
IMG_DIR = os.path.join(DATA_DIR, "stage_2_train_images")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def leer_y_redimensionar_224(dcm_path, size=IMG_SIZE):
    """Lee un DICOM, corrige MONOCHROME1, pasa a uint8 [0, 255] y redimensiona a (size, size)."""
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)

    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = np.max(img) - img

    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    img_u8 = (img * 255).astype(np.uint8)
    return cv2.resize(img_u8, (size, size), interpolation=cv2.INTER_AREA)


def normalizar_tensor_224(img_224, mode="scratch"):
    """Pasa una imagen (224, 224) a tensor (3, 224, 224) y normaliza ('scratch' o 'transfer')."""
    img_3ch = np.stack([img_224, img_224, img_224], axis=0)
    tensor = v2.ToDtype(torch.float32, scale=True)(torch.from_numpy(img_3ch))

    if mode == "scratch":
        return tensor
    if mode == "transfer":
        return v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(tensor)
    raise ValueError("Usa mode='scratch' o mode='transfer'.")


def main():
    df = pd.read_csv(LABELS_PATH).drop_duplicates(subset=["patientId"]).reset_index(drop=True)
    fila = df.iloc[0]
    dcm_path = os.path.join(IMG_DIR, f"{fila['patientId']}.dcm")

    orig_shape = pydicom.dcmread(dcm_path).pixel_array.shape
    img_224 = leer_y_redimensionar_224(dcm_path)
    t_scratch = normalizar_tensor_224(img_224, mode="scratch")
    t_transfer = normalizar_tensor_224(img_224, mode="transfer")

    print("Original DICOM:", orig_shape, "-> Redimensionada:", img_224.shape)
    print("Tensor final:", tuple(t_scratch.shape))
    print(f"Normalizado [0, 1]:   min={t_scratch.min():.3f}, max={t_scratch.max():.3f}")
    print(f"Normalizado ImageNet: min={t_transfer.min():.3f}, max={t_transfer.max():.3f}")


if __name__ == "__main__":
    main()