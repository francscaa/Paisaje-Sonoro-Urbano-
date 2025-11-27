"""Clasificador de segmentos con YAMNet + descriptores basicos.

Requisitos previos (ya instalados en tu entorno):
- tensorflow, tensorflow_hub, librosa, numpy, matplotlib, scipy, pandas

Uso:
    python scripts/YAMNet.py --file /ruta/audio.wav --window 3.0
"""

from __future__ import annotations

import argparse
import certifi
import csv
import os
from pathlib import Path
from typing import List, Tuple

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import tensorflow as tf
import tensorflow_hub as hub
from tkinter import Tk, filedialog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmenta un audio, calcula descriptores y clasifica con YAMNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Ruta a un WAV/MP3. Si no se indica, se abre un selector.",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=3.0,
        help="Tamaño de ventana en segundos para segmentar el audio.",
    )
    parser.add_argument(
        "--hop",
        type=float,
        default=None,
        help="Paso entre ventanas. Si no se indica, usa el mismo valor de window.",
    )
    parser.add_argument(
        "--model-handle",
        default="https://tfhub.dev/google/yamnet/1",
        help="Handle de TF Hub o ruta local a un modelo YAMNet descomprimido.",
    )
    parser.add_argument(
        "--hub-cache",
        type=Path,
        default=Path("models/tfhub"),
        help="Carpeta de cache de TF Hub para evitar descargas repetidas.",
    )
    return parser.parse_args()


def select_file(path: Path | None) -> Path:
    if path:
        if not path.exists():
            raise SystemExit(f"No existe el archivo: {path}")
        return path
    Tk().withdraw()
    picked = filedialog.askopenfilename(
        title="Selecciona un archivo de audio",
        filetypes=[("Audio", "*.wav *.mp3"), ("All files", "*.*")],
    )
    if not picked:
        raise SystemExit("No se selecciono ningun archivo.")
    return Path(picked)


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


def load_model():
    # Forzar certificados a usar certifi (evita errores SSL en macOS/Python)
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    print(f"Usando SSL_CERT_FILE={os.environ['SSL_CERT_FILE']}")

    def _loader(handle: str):
        print(f"Cargando YAMNet desde: {handle}")
        model = hub.load(handle)
        class_names = class_names_from_csv(model.class_map_path().numpy())
        print(f"✔ Modelo cargado. Clases: {len(class_names)}")
        return model, class_names

    return _loader


def normalize_waveform(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    maxabs = np.max(np.abs(x)) if x.size else 0.0
    if maxabs > 0:
        x = x / maxabs
    return x


def process_segment(segment: np.ndarray, sr: int, timestamp: float, model, class_names: List[str]) -> dict:
    # Descriptores
    try:
        mean_centroid = float(np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)))
    except Exception:
        mean_centroid = np.nan
    try:
        mean_zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=segment)))
    except Exception:
        mean_zcr = np.nan
    try:
        mean_rms = float(np.mean(librosa.feature.rms(y=segment)))
    except Exception:
        mean_rms = np.nan

    # YAMNet
    top_class = "Unknown"
    confidence = 0.0
    try:
        sr_y, seg_y = ensure_sample_rate(sr, segment)
        seg_y = normalize_waveform(seg_y)
        scores, _, _ = model(seg_y)
        mean_scores = np.mean(scores.numpy(), axis=0)
        top_idx = int(np.argmax(mean_scores))
        top_class = class_names[top_idx]
        confidence = float(mean_scores[top_idx])
    except Exception as exc:
        print(f"[aviso] Error YAMNet en t={timestamp:.2f}s: {exc}")

    lat = 33.45 + (timestamp / 200) * 0.01
    lon = -70.66 + np.random.rand() * 0.001

    return {
        "Timestamp": timestamp,
        "Latitud": lat,
        "Longitud": lon,
        "Clase_YAMNet": top_class,
        "Probabilidad": confidence,
        "Centr_Espectral": mean_centroid,
        "ZCR": mean_zcr,
        "RMS": mean_rms,
    }


def plot_descriptors(df: pd.DataFrame) -> None:
    out_path = Path("results") / "yamnet_descriptores.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(df["Timestamp"], df["Centr_Espectral"], label="Centroide Espectral", color="blue")
    plt.ylabel("Frecuencia (Hz)")
    plt.title("Descriptores Acusticos en el Tiempo")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(df["Timestamp"], df["ZCR"], label="ZCR", color="orange")
    plt.ylabel("Tasa")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(df["Timestamp"], df["RMS"], label="RMS", color="green")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Grafico de descriptores: {out_path}")


def plot_classes(df: pd.DataFrame) -> None:
    out_path = Path("results") / "yamnet_clases.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    scatter = plt.scatter(
        df["Timestamp"],
        df["Probabilidad"],
        c=pd.factorize(df["Clase_YAMNet"])[0],
        cmap="tab20",
    )
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Probabilidad")
    plt.title("Clases YAMNet en el Tiempo")
    plt.grid(True)

    handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
    legend_labels = pd.factorize(df["Clase_YAMNet"])[1]
    plt.legend(handles, legend_labels, loc="best", title="Clase YAMNet")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Grafico de clases: {out_path}")


def main() -> None:
    args = parse_args()
    hop = args.hop or args.window
    audio_path = select_file(args.file)

    # Evita posibles fallos de Metal en Macs; fuerza CPU.
    try:
        tf.config.set_visible_devices([], "GPU")
        tf.config.set_visible_devices([], "TPU")
    except Exception:
        pass

    os.environ.setdefault("TFHUB_CACHE_DIR", str(args.hub_cache))
    args.hub_cache.mkdir(parents=True, exist_ok=True)
    loader = load_model()

    # Si no se indica handle y existe un modelo local, usarlo.
    default_local = args.hub_cache
    handle = args.model_handle
    if handle.startswith("http") and default_local.exists() and (default_local / "saved_model.pb").exists():
        print(f"Detectado modelo local en {default_local}, usandolo en lugar de TF Hub.")
        handle = str(default_local)

    try:
        model, class_names = loader(handle)
    except Exception as exc:
        raise SystemExit(
            "No se pudo cargar YAMNet.\n"
            f"Handle: {handle}\n"
            "Si es error SSL, intenta:\n"
            "  export SSL_CERT_FILE=$(python -m certifi)\n"
            "  export TFHUB_CACHE_DIR=./models/tfhub\n"
            "  (y descarga el modelo manualmente y descomprime en esa carpeta)\n"
            f"Detalle: {exc}"
        ) from exc

    print(f"Cargando audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Duracion: {duration:.2f}s")

    results = []
    for start_time in np.arange(0, duration, hop):
        end_time = min(start_time + args.window, duration)
        start_sample = librosa.time_to_samples(start_time, sr=sr)
        end_sample = librosa.time_to_samples(end_time, sr=sr)
        segment = y[start_sample:end_sample]
        if segment.size == 0:
            continue
        results.append(process_segment(segment, sr, start_time, model, class_names))
        print(f"Procesando segmento: {start_time:.1f}s - {end_time:.1f}s")

    if not results:
        raise SystemExit("No se pudieron procesar segmentos.")

    df = pd.DataFrame(results)
    out_csv = Path("results") / f"{audio_path.stem}_yamnet.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nCSV guardado en: {out_csv}")
    print(df.head())

    plot_descriptors(df)
    plot_classes(df)


if __name__ == "__main__":
    main()
