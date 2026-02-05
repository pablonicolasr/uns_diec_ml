import numpy as np
import cv2


from pathlib import Path
from skimage.feature import hog, local_binary_pattern
from typing import Any


# Preprocesamiento de image para RX
def preprocess_xray(img_bgr, out_size):
    """
    - Grayscale
    - Resize
    - CLAHE (mejora contraste en RX)
    - Denoise suave
    Devuelve uint8 [0..255] 2D
    """
    if img_bgr is None:
        
        raise ValueError("Imagen None (posible path corrupto).")

    # Grayscale
    if img_bgr.ndim == 3:
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
    else:
        
        gray = img_bgr.copy()

    # Resize (W,H)
    gray = cv2.resize(gray, out_size, interpolation=cv2.INTER_AREA)

    # CLAHE --> Contrast Limited Adaptive Histogram Equalization --> Algoritmo robusto para aplicar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Denoise suave (no matar bordes)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray


# Features híbridas (HOG + LBP + Hist + Stats) - Ingeniería de Datos
def features_hog(gray, cfg):
    feat = hog(
        gray,
        orientations=cfg.hog_orientations,
        pixels_per_cell=cfg.hog_pixels_per_cell,
        cells_per_block=cfg.hog_cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )
    return feat.astype(np.float32)
    

def features_lbp(gray, cfg):
    lbp = local_binary_pattern(gray, P=cfg.lbp_points, R=cfg.lbp_radius, method="uniform")
    # histograma LBP (uniform -> bins = P+2)
    n_bins = cfg.lbp_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins), density=True)
    return hist.astype(np.float32)
    

def features_intensity(gray, cfg):
    # Histograma de intensidades
    hist = cv2.calcHist([gray], [0], None, [cfg.hist_bins], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-9)

    # Stats
    g = gray.astype(np.float32) / 255.0
    stats = np.array([
        g.mean(),
        g.std(),
        np.median(g),
        np.percentile(g, 25),
        np.percentile(g, 75),
        (g > 0.5).mean(),       # proporción de pixeles “claros”
        (g < 0.2).mean(),       # proporción de pixeles “oscuros”
    ], dtype=np.float32)

    return np.concatenate([hist.astype(np.float32), stats], axis=0)
    

def extract_features_from_path(path, cfg):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    gray = preprocess_xray(img, cfg.image_size)

    f_hog = features_hog(gray, cfg)
    f_lbp = features_lbp(gray, cfg)
    f_int = features_intensity(gray, cfg)

    return np.concatenate([f_hog, f_lbp, f_int], axis=0).astype(np.float32)