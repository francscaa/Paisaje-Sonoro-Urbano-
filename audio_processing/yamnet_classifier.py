from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Tuple

import certifi
import numpy as np
import scipy.signal
import tensorflow as tf
import tensorflow_hub as hub


def _patch_numpy_printoptions() -> None:
    """Normaliza np.set_printoptions para evitar fallos con legacy=1.25."""
    real_set = np.set_printoptions

    def safe_set_printoptions(*args, **kwargs):
        if kwargs.get("legacy") == "1.25":
            kwargs["legacy"] = False
        return real_set(*args, **kwargs)

    np.set_printoptions = safe_set_printoptions


_patch_numpy_printoptions()


def class_names_from_csv(class_map_csv_text: str) -> List[str]:
    names: List[str] = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            names.append(row["display_name"])
    return names


def ensure_sample_rate(sr: int, wav: np.ndarray, desired: int = 16000) -> Tuple[int, np.ndarray]:
    if sr == desired:
        return sr, wav.astype(np.float32)
    desired_len = int(round(float(len(wav)) / sr * desired))
    wav = wav.astype(np.float32)
    wav = scipy.signal.resample(wav, desired_len)
    return desired, wav


def normalize_waveform(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    maxabs = np.max(np.abs(x)) if x.size else 0.0
    if maxabs > 0:
        x = x / maxabs
    x *= 0.5
    return x


def load_yamnet_model(handle: str, hub_cache: Path):
    """Carga YAMNet desde TF Hub o desde un saved_model local."""
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("TFHUB_CACHE_DIR", str(hub_cache))
    hub_cache.mkdir(parents=True, exist_ok=True)
    print(f"Usando SSL_CERT_FILE={os.environ['SSL_CERT_FILE']}")

    # Fuerza CPU para evitar problemas con GPU/Metal.
    try:
        tf.config.set_visible_devices([], "GPU")
        tf.config.set_visible_devices([], "TPU")
    except Exception:
        pass

    def _loader(h: str):
        print(f"Cargando YAMNet desde: {h}")
        model = hub.load(h)
        class_names = class_names_from_csv(model.class_map_path().numpy())
        print(f"✔ Modelo cargado. Clases: {len(class_names)}")
        return model, class_names

    return _loader(handle)


def predict_top_class(
    segment: np.ndarray, sr: int, model, class_names: List[str], timestamp: float | None = None
) -> dict:
    """Ejecuta YAMNet sobre un segmento y retorna clase, índice y probabilidad."""
    segment = normalize_waveform(segment)
    top_class = "Unknown"
    confidence = 0.0
    top_idx = -1
    try:
        sr_y, seg_y = ensure_sample_rate(sr, segment)
        seg_y = normalize_waveform(seg_y)
        scores, _, _ = model(seg_y)
        mean_scores = np.mean(scores.numpy(), axis=0)
        top_idx = int(np.argmax(mean_scores))
        top_class = class_names[top_idx]
        confidence = float(mean_scores[top_idx])
    except Exception as exc:
        if timestamp is not None:
            print(f"[aviso] Error YAMNet en t={timestamp:.2f}s: {exc}")
    return {"Clase_YAMNet": top_class, "Probabilidad": confidence, "TopIndex": top_idx}
