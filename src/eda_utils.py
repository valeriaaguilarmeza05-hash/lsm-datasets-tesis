"""
eda_utils.py
============
Utilidades comunes para el EDA de los cinco datasets de Lengua de Señas
Mexicana (LSM): inventario de archivos, extracción de metadatos de imágenes
y videos, embeddings ligeros para t-SNE/UMAP.

Autor: Valeria Aguilar Meza · FIAD-UABC · 2026-1
"""

import os
import json
import pickle
import zipfile
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# 1.  Inventario de archivos en una carpeta o ZIP
# ---------------------------------------------------------------------------

def inventario_carpeta(ruta, extensiones=None):
    """
    Recorre recursivamente 'ruta' y devuelve un DataFrame con: ruta, nombre,
    extensión, tamaño_bytes, carpeta_padre (= clase si la estructura es
    'ruta/CLASE/archivo.ext').
    """
    extensiones = extensiones or [".jpg", ".jpeg", ".png", ".bmp", ".tif",
                                  ".mp4", ".mov", ".avi", ".mkv",
                                  ".csv", ".tsv", ".json", ".xml",
                                  ".pickle", ".pkl", ".h5", ".mat", ".npy"]
    registros = []
    for raiz, _, archivos in os.walk(ruta):
        for nombre in archivos:
            ext = os.path.splitext(nombre)[1].lower()
            if ext in extensiones:
                ruta_archivo = os.path.join(raiz, nombre)
                try:
                    tamano = os.path.getsize(ruta_archivo)
                except OSError:
                    tamano = np.nan
                registros.append({
                    "ruta": ruta_archivo,
                    "nombre": nombre,
                    "extension": ext,
                    "tamano_bytes": tamano,
                    "tamano_mb": round(tamano / (1024 ** 2), 4) if tamano else np.nan,
                    "carpeta_padre": os.path.basename(raiz),
                    "clase": os.path.basename(raiz),
                })
    return pd.DataFrame(registros)


def inventario_zip(ruta_zip):
    """
    Lee un ZIP sin extraerlo y devuelve un DataFrame con nombre, extensión,
    tamaño comprimido y descomprimido, y carpeta padre (clase).
    """
    registros = []
    with zipfile.ZipFile(ruta_zip, "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            ext = os.path.splitext(info.filename)[1].lower()
            partes = info.filename.split("/")
            carpeta = partes[-2] if len(partes) >= 2 else ""
            registros.append({
                "nombre": os.path.basename(info.filename),
                "ruta_zip": info.filename,
                "extension": ext,
                "tamano_compr": info.compress_size,
                "tamano_descompr": info.file_size,
                "tamano_mb": round(info.file_size / (1024 ** 2), 4),
                "clase": carpeta,
            })
    return pd.DataFrame(registros)


# ---------------------------------------------------------------------------
# 2.  Extracción de metadatos de imagen
# ---------------------------------------------------------------------------

def metadatos_imagen(ruta_img, calcular_brillo=True):
    """
    Devuelve dict con ancho, alto, modo, n_canales y brillo medio (0-255).
    Robusto a errores de archivo corrupto.
    """
    try:
        with Image.open(ruta_img) as im:
            ancho, alto = im.size
            modo = im.mode
            n_canales = len(im.getbands())
            brillo = np.nan
            if calcular_brillo:
                arr = np.asarray(im.convert("L"), dtype=np.float32)
                brillo = float(arr.mean())
            return {
                "ancho": ancho, "alto": alto, "modo": modo,
                "n_canales": n_canales, "aspect_ratio": round(ancho / alto, 4),
                "brillo_medio": round(brillo, 3) if not np.isnan(brillo) else np.nan,
            }
    except Exception as e:
        return {"ancho": np.nan, "alto": np.nan, "modo": None,
                "n_canales": np.nan, "aspect_ratio": np.nan,
                "brillo_medio": np.nan, "error": str(e)}


def extraer_imagen_de_zip(z, ruta_interna):
    """Abre una imagen contenida en un ZipFile abierto y devuelve un PIL Image."""
    with z.open(ruta_interna) as f:
        return Image.open(f).copy()


def metadatos_imagenes_en_zip(ruta_zip, lista_rutas_internas, n_max=None,
                              calcular_brillo=True, sample_seed=42):
    """
    Procesa una lista de imágenes dentro de un ZIP y devuelve un DataFrame
    con sus metadatos. Si n_max se proporciona, hace muestreo estratificado.
    """
    if n_max is not None and n_max < len(lista_rutas_internas):
        rng = np.random.default_rng(sample_seed)
        idx = rng.choice(len(lista_rutas_internas), size=n_max, replace=False)
        lista_rutas_internas = [lista_rutas_internas[i] for i in idx]

    registros = []
    with zipfile.ZipFile(ruta_zip, "r") as z:
        for ruta in lista_rutas_internas:
            try:
                with z.open(ruta) as f:
                    with Image.open(f) as im:
                        ancho, alto = im.size
                        modo = im.mode
                        n_canales = len(im.getbands())
                        if calcular_brillo:
                            arr = np.asarray(im.convert("L"), dtype=np.float32)
                            brillo = float(arr.mean())
                        else:
                            brillo = np.nan
                registros.append({
                    "ruta_interna": ruta,
                    "ancho": ancho, "alto": alto, "modo": modo,
                    "n_canales": n_canales,
                    "aspect_ratio": round(ancho / alto, 4),
                    "brillo_medio": round(brillo, 3) if not np.isnan(brillo) else np.nan,
                    "clase": ruta.split("/")[-2] if "/" in ruta else "",
                })
            except Exception as e:
                registros.append({"ruta_interna": ruta, "error": str(e)})
    return pd.DataFrame(registros)


# ---------------------------------------------------------------------------
# 3.  Extracción de metadatos de video (sin reproducir todo el clip)
# ---------------------------------------------------------------------------

def metadatos_video(ruta_video):
    """Devuelve resolución, fps, número de frames, duración, codec."""
    import cv2
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        return {"error": "no abre"}
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion = n_frames / fps if fps > 0 else np.nan
    cap.release()
    return {
        "ancho": ancho, "alto": alto, "fps": round(fps, 2),
        "n_frames": n_frames, "duracion_s": round(duracion, 3),
        "aspect_ratio": round(ancho / alto, 4) if alto else np.nan,
    }


# ---------------------------------------------------------------------------
# 4.  Embedding ligero para t-SNE/UMAP en imágenes
# ---------------------------------------------------------------------------

def embedding_simple(img, tamano=(32, 32)):
    """
    Vector de características de baja dimensionalidad para t-SNE rápido:
    redimensiona a 32x32 grayscale y aplana (1024 dims).
    """
    if isinstance(img, str):
        img = Image.open(img)
    g = img.convert("L").resize(tamano)
    return np.asarray(g, dtype=np.float32).flatten() / 255.0


def embedding_hog(img, tamano=(64, 64)):
    """Histogram of Oriented Gradients vía scikit-image (rápido y discriminativo)."""
    from skimage.feature import hog
    if isinstance(img, str):
        img = Image.open(img)
    g = np.asarray(img.convert("L").resize(tamano), dtype=np.float32)
    return hog(g, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), feature_vector=True)


# ---------------------------------------------------------------------------
# 5.  Hash y trazabilidad
# ---------------------------------------------------------------------------

def md5_archivo(ruta, chunk_size=8192):
    """Hash MD5 de un archivo (para reproducibilidad y citación)."""
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
