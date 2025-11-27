"""Segmenta audio, clasifica con YAMNet y calcula descriptores psicoacusticos.

Metrica por segmento (window/hop): clase YAMNet + Loudness, Sharpness, Roughness, Tonality.
Sin centroides/ZCR/RMS para acelerar.

Uso:
    python scripts/YAMNet_Soundscapy.py --file /ruta/audio.wav --window 3.0 --hop 3.0
"""

from __future__ import annotations

import argparse
import certifi
import csv
import os
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
import scipy.signal
import tensorflow as tf
import tensorflow_hub as hub
from tkinter import Tk, filedialog
import mosqito as mq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmenta audio, clasifica con YAMNet y calcula descriptores psicoacusticos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=Path,
        nargs="+",
        help="Rutas a uno o varios WAV/MP3. Si no se indica, se abre un selector multiple.",
    )
    parser.add_argument("--window", type=float, default=3.0, help="Tamaño de ventana en segundos.")
    parser.add_argument("--hop", type=float, default=None, help="Paso entre ventanas. Si no se indica, usa window.")
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


def select_files(paths: list[Path] | None) -> list[Path]:
    if paths:
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"No existen: {missing}")
        return paths
    Tk().withdraw()
    picked = filedialog.askopenfilenames(
        title="Selecciona uno o mas archivos de audio",
        filetypes=[("Audio", "*.wav *.mp3"), ("All files", "*.*")],
    )
    if not picked:
        raise SystemExit("No se seleccionaron archivos.")
    return [Path(p) for p in picked]


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
    x *= 0.5  # margen para evitar saturacion en metricas
    return x


def load_model():
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    print(f"Usando SSL_CERT_FILE={os.environ['SSL_CERT_FILE']}")

    def _loader(handle: str):
        print(f"Cargando YAMNet desde: {handle}")
        model = hub.load(handle)
        class_names = class_names_from_csv(model.class_map_path().numpy())
        print(f"✔ Modelo cargado. Clases: {len(class_names)}")
        return model, class_names

    return _loader


def compute_psycho(segment: np.ndarray, sr: int) -> dict:
    metrics = {}
    try:
        N, *_ = mq.loudness_zwst(segment, sr)
        metrics["loudness_sones"] = float(np.mean(np.atleast_1d(N)))
    except Exception as exc:
        print(f"[aviso] Loudness fallo: {exc}")
        metrics["loudness_sones"] = np.nan

    try:
        S = mq.sharpness_din_st(segment, sr)
        metrics["sharpness_acum"] = float(np.mean(np.atleast_1d(S)))
    except Exception as exc:
        print(f"[aviso] Sharpness fallo: {exc}")
        metrics["sharpness_acum"] = np.nan

    try:
        R, *_ = mq.roughness_dw(segment, sr)
        metrics["roughness_asper"] = float(np.mean(np.atleast_1d(R)))
    except Exception as exc:
        print(f"[aviso] Roughness fallo: {exc}")
        metrics["roughness_asper"] = np.nan

    try:
        tnr_total, *_ = mq.tnr_ecma_st(segment, sr)
        pr_total, *_ = mq.pr_ecma_st(segment, sr)
        metrics["tonality_tnr_db"] = float(np.mean(np.atleast_1d(tnr_total)))
        metrics["tonality_pr_db"] = float(np.mean(np.atleast_1d(pr_total)))
    except Exception as exc:
        print(f"[aviso] Tonality fallo: {exc}")
        metrics["tonality_tnr_db"] = np.nan
        metrics["tonality_pr_db"] = np.nan
    return metrics


def process_segment(segment: np.ndarray, sr: int, timestamp: float, model, class_names: List[str]) -> dict:
    segment = normalize_waveform(segment)

    # YAMNet
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
        print(f"[aviso] Error YAMNet en t={timestamp:.2f}s: {exc}")

    psycho = compute_psycho(segment, sr)

    return {
        "Timestamp": timestamp,
        "Recording": "",
        "Clase_YAMNet": top_class,
        "Probabilidad": confidence,
        "TopIndex": top_idx,
        **psycho,
    }


def plot_sources(df: pd.DataFrame, class_names: List[str]) -> None:
    out_path = Path("results/plots") / "yamnet_fuentes_scatter.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Scatter temporal con clases como colores
    plt.figure(figsize=(12, 6))
    factor, labels = pd.factorize(df["Clase_YAMNet"])
    scatter = plt.scatter(df["Timestamp"], df["Probabilidad"], c=factor, cmap="tab20", alpha=0.8)
    handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
    plt.legend(handles, labels, loc="best", title="Clase YAMNet", fontsize="small", ncol=2)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Probabilidad")
    plt.title("Clases YAMNet en el tiempo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Grafico fuentes (scatter): {out_path}")


def plot_compare_descriptors(df: pd.DataFrame) -> None:
    """Grafico comparativo de descriptores psicoacusticos promedio por audio (si hay >1)."""
    if "Recording" not in df.columns:
        return
    out_path = Path("results/plots") / "yamnet_psico_comparacion.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    descriptors = ["loudness_sones", "sharpness_acum", "roughness_asper", "tonality_tnr_db"]
    grouped = df.groupby("Recording")[descriptors].mean()
    if grouped.shape[0] < 2:
        print("[aviso] Solo hay un recording; se omite comparacion de audios.")
        return

    plt.figure(figsize=(10, 6))
    x = np.arange(len(grouped.index))
    width = 0.2
    for i, desc in enumerate(descriptors):
        if desc in grouped.columns:
            plt.bar(x + i * width, grouped[desc], width=width, label=desc)
    plt.xticks(x + width * (len(descriptors) - 1) / 2, grouped.index, rotation=45, ha="right")
    plt.ylabel("Valor promedio")
    plt.title("Comparacion de descriptores psicoacusticos por audio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Grafico comparacion psico: {out_path}")


def main() -> None:
    args = parse_args()
    hop = args.hop or args.window
    audio_paths = select_files(args.file)

    # Evita posibles fallos de Metal en Macs; fuerza CPU.
    try:
        tf.config.set_visible_devices([], "GPU")
        tf.config.set_visible_devices([], "TPU")
    except Exception:
        pass

    os.environ.setdefault("TFHUB_CACHE_DIR", str(args.hub_cache))
    args.hub_cache.mkdir(parents=True, exist_ok=True)
    loader = load_model()

    # Si hay modelo local en hub-cache, usarlo por defecto.
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
            "  (y coloca el modelo descomprimido en esa carpeta)\n"
            f"Detalle: {exc}"
        ) from exc

    all_rows = []
    for audio_path in audio_paths:
        print(f"Cargando audio: {audio_path}")
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"Duracion: {duration:.2f}s")

        for start_time in np.arange(0, duration, hop):
            end_time = min(start_time + args.window, duration)
            start_sample = librosa.time_to_samples(start_time, sr=sr)
            end_sample = librosa.time_to_samples(end_time, sr=sr)
            segment = y[start_sample:end_sample]
            if segment.size == 0:
                continue
            row = process_segment(segment, sr, start_time, model, class_names)
            row["Recording"] = audio_path.stem
            all_rows.append(row)
            print(f"Procesando segmento: {start_time:.1f}s - {end_time:.1f}s ({audio_path.stem})")

    if not all_rows:
        raise SystemExit("No se pudieron procesar segmentos.")

    df = pd.DataFrame(all_rows)
    out_csv = Path("results") / "yamnet_psico_segmentado.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nCSV guardado en: {out_csv}")
    print(df.head())

    # Graficos
    plot_sources(df, class_names)
    plot_compare_descriptors(df)


if __name__ == "__main__":
    main()
