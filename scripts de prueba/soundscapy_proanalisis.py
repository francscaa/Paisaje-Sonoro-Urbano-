"""Pipeline completo: segmenta audios, clasifica con YAMNet, calcula psico (MoSQITo) y analiza.

Salidas:
- CSV segmentado con clase YAMNet + loudness/sharpness/roughness/tonality: results/yamnet_psico_segmentado.csv
- CSV agregado por Recording: results/yamnet_psico_recordings.csv
- Graficos en results/plots: scatter de fuentes, barras, comparacion por audio, correlaciones, clusters, mapas perceptuales.

Uso tipico:
  python scripts/soundscapy_proanalisis.py --files audio1.wav audio2.wav --window 3.0 --hop 3.0
  # si ya tienes el CSV segmentado:
  python scripts/soundscapy_proanalisis.py --csv results/yamnet_psico_segmentado.csv
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
import mosqito as mq
import numpy as np
import pandas as pd
import scipy.signal
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Soundscapy usa np.set_printoptions(legacy="1.25") en algunas rutas; lo normalizamos para evitar TypeError.
def _patch_numpy_printoptions() -> None:
    real_set = np.set_printoptions

    def safe_set_printoptions(*args, **kwargs):
        if kwargs.get("legacy") == "1.25":
            kwargs["legacy"] = False
        return real_set(*args, **kwargs)

    np.set_printoptions = safe_set_printoptions

_patch_numpy_printoptions()
try:
    from soundscapy.plotting import Backend, density_plot, scatter_plot
    HAS_SOUNDSCAPY_PLOTS = True
except Exception:
    HAS_SOUNDSCAPY_PLOTS = False

# Fallback local para graficos tipo SoundscapePlot / LocationComparisons
# usando P/E calculados (proxy) y Recording como LocationID.
def plot_soundscape_fallback(df_rec: pd.DataFrame) -> Path | None:
    if df_rec.empty:
        return None
    out_path = PLOT_DIR / "soundscapy_lugares_fallback.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(df_rec["P_iso"], df_rec["E_iso"], s=120, alpha=0.9)
    for _, row in df_rec.iterrows():
        plt.text(row["P_iso"], row["E_iso"], row["Recording"], fontsize=8, ha="left", va="bottom")
    plt.xlabel("P (proxy)")
    plt.ylabel("E (proxy)")
    plt.title("Comparacion perceptual (fallback)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_location_comparisons_fallback(df_rec: pd.DataFrame) -> Path | None:
    if df_rec.shape[0] < 2:
        return None
    out_path = PLOT_DIR / "soundscapy_lugares_multiple_fallback.png"
    plt.figure(figsize=(10, 6))
    plt.scatter(df_rec["P_iso"], df_rec["E_iso"], s=120, alpha=0.9)
    for _, row in df_rec.iterrows():
        plt.text(row["P_iso"], row["E_iso"], row["Recording"], fontsize=8, ha="left", va="bottom")
    plt.xlabel("P (proxy)")
    plt.ylabel("E (proxy)")
    plt.title("Comparacion perceptual por audio (fallback)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

PLOT_DIR = Path("results/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DESCRIPTORS = ["loudness_sones", "sharpness_acum", "roughness_asper", "tonality_tnr_db"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmenta audios, calcula psico y analiza (correlaciones, clusters, graficos).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--files",
        type=Path,
        nargs="+",
        help="Rutas a uno o varios WAV/MP3. Si no se indican, se abre selector.",
    )
    parser.add_argument("--window", type=float, default=3.0, help="Ventana de analisis en segundos.")
    parser.add_argument("--hop", type=float, default=None, help="Paso entre ventanas. Por defecto = window.")
    parser.add_argument(
        "--model-handle",
        default="https://tfhub.dev/google/yamnet/1",
        help="Handle de TF Hub o ruta local a modelo YAMNet (saved_model.pb).",
    )
    parser.add_argument(
        "--hub-cache",
        type=Path,
        default=Path("models/tfhub"),
        help="Cache de TF Hub / carpeta del modelo local.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/yamnet_psico_segmentado.csv"),
        help="CSV segmentado (si ya lo tienes y no quieres re-procesar audios).",
    )
    return parser.parse_args()


def select_files(paths: list[Path] | None) -> list[Path]:
    if paths:
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"No existen: {missing}")
        return paths
    try:
        from tkinter import Tk, filedialog
    except Exception:
        raise SystemExit(
            "Tkinter no esta disponible en este Python. Pasa los archivos con --files "
            "para omitir el selector grafico."
        )
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
    x *= 0.5
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


def process_audios(audio_paths: list[Path], window: float, hop: float, model_handle: str, hub_cache: Path) -> pd.DataFrame:
    # fuerza CPU para evitar problemas con Metal
    try:
        tf.config.set_visible_devices([], "GPU")
        tf.config.set_visible_devices([], "TPU")
    except Exception:
        pass

    os.environ.setdefault("TFHUB_CACHE_DIR", str(hub_cache))
    hub_cache.mkdir(parents=True, exist_ok=True)
    loader = load_model()
    handle = model_handle
    default_local = hub_cache
    if handle.startswith("http") and default_local.exists() and (default_local / "saved_model.pb").exists():
        print(f"Detectado modelo local en {default_local}, usandolo en lugar de TF Hub.")
        handle = str(default_local)
    try:
        model, class_names = loader(handle)
    except Exception as exc:
        raise SystemExit(
            "No se pudo cargar YAMNet.\n"
            f"Handle: {handle}\n"
            "Si es error SSL, descarga el modelo y descomprímelo en models/tfhub.\n"
            f"Detalle: {exc}"
        ) from exc

    results = []
    for audio_path in audio_paths:
        print(f"Cargando audio: {audio_path}")
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"Duracion: {duration:.2f}s")
        step = hop or window
        for start_time in np.arange(0, duration, step):
            end_time = min(start_time + window, duration)
            start_sample = librosa.time_to_samples(start_time, sr=sr)
            end_sample = librosa.time_to_samples(end_time, sr=sr)
            segment = y[start_sample:end_sample]
            if segment.size == 0:
                continue
            row = process_segment(segment, sr, start_time, model, class_names)
            row["Recording"] = audio_path.stem
            results.append(row)
            print(f"Procesando segmento: {start_time:.1f}-{end_time:.1f}s ({audio_path.stem})")
    if not results:
        raise SystemExit("No se pudieron procesar segmentos.")
    df = pd.DataFrame(results)
    out_csv = Path("results") / "yamnet_psico_segmentado.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nCSV segmentado guardado en: {out_csv}")
    return df


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(
            f"No existe el CSV: {csv_path}\n"
            "Primero ejecuta con --files para generar el CSV segmentado."
        )
    df = pd.read_csv(csv_path)
    if "Recording" not in df.columns:
        df["Recording"] = "audio"
    return df


def compute_iso_proxies(df: pd.DataFrame) -> pd.DataFrame:
    zdf = df.copy()
    for d in DESCRIPTORS:
        if d in zdf.columns:
            zdf[d + "_z"] = (zdf[d] - zdf[d].mean()) / (zdf[d].std() or 1)
        else:
            zdf[d + "_z"] = 0.0
    zdf["P_iso"] = (
        -zdf["loudness_sones_z"]
        - zdf["roughness_asper_z"]
        - zdf["sharpness_acum_z"]
        - zdf["tonality_tnr_db_z"]
    )
    zdf["E_iso"] = (
        zdf["loudness_sones_z"]
        + zdf["roughness_asper_z"]
        + zdf["sharpness_acum_z"]
        + zdf["tonality_tnr_db_z"]
    )
    return zdf


def aggregate_by_recording(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("Recording").mean(numeric_only=True).reset_index()
    return agg


def plot_sources(df: pd.DataFrame) -> Path:
    out_path = PLOT_DIR / "yamnet_fuentes_scatter.png"
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
    return out_path


def plot_descriptor_bars(df: pd.DataFrame) -> Path:
    out_path = PLOT_DIR / "fuentes_por_descriptor.png"
    top = (
        df.groupby("Clase_YAMNet")[DESCRIPTORS]
        .mean()
        .sort_values(by="loudness_sones", ascending=False)
        .head(10)
    )
    top.plot(kind="bar", figsize=(12, 6))
    plt.ylabel("Valor promedio")
    plt.title("Fuentes (YAMNet) ordenadas por loudness (top 10) con otros descriptores")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_compare_recordings(df_rec: pd.DataFrame) -> Path | None:
    if df_rec.shape[0] < 2:
        return None
    out_path = PLOT_DIR / "comparacion_recordings_psico.png"
    x = np.arange(len(df_rec["Recording"]))
    width = 0.2
    plt.figure(figsize=(10, 6))
    for i, d in enumerate(DESCRIPTORS):
        if d in df_rec.columns:
            plt.bar(x + i * width, df_rec[d], width=width, label=d)
    plt.xticks(x + width * (len(DESCRIPTORS) - 1) / 2, df_rec["Recording"], rotation=45, ha="right")
    plt.ylabel("Valor promedio")
    plt.title("Comparacion de descriptores psicoacusticos por audio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_correlations(df: pd.DataFrame) -> Path:
    out_path = PLOT_DIR / "correlaciones.png"
    cols = [c for c in DESCRIPTORS + ["P_iso", "E_iso", "Probabilidad"] if c in df.columns]
    corr = df[cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlaciones")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_clusters(df: pd.DataFrame) -> Path:
    out_path = PLOT_DIR / "clusters_psico.png"
    data = df[DESCRIPTORS].fillna(df[DESCRIPTORS].median())
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    k = 3 if len(df) >= 3 else len(df)
    if k < 1:
        return out_path
    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    labels = km.fit_predict(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(df["P_iso"], df["E_iso"], c=labels, cmap="tab10", alpha=0.8)
    plt.xlabel("P_iso (proxy)")
    plt.ylabel("E_iso (proxy)")
    plt.title("Clusters en el espacio P/E (proxy)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_perceptual(df: pd.DataFrame, df_rec: pd.DataFrame) -> list[Path]:
    out_paths: list[Path] = []
    out_seg = PLOT_DIR / "perceptual_segmentos.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(df["P_iso"], df["E_iso"], c=df["Probabilidad"], cmap="viridis", alpha=0.6)
    plt.xlabel("P_iso (proxy)")
    plt.ylabel("E_iso (proxy)")
    plt.title("Mapa perceptual por segmento")
    plt.colorbar(label="Probabilidad YAMNet")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_seg, dpi=150)
    plt.close()
    out_paths.append(out_seg)

    out_rec = PLOT_DIR / "perceptual_recordings.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(df_rec["P_iso"], df_rec["E_iso"], s=120, alpha=0.9)
    for _, row in df_rec.iterrows():
        plt.text(row["P_iso"], row["E_iso"], row["Recording"], fontsize=8, ha="left", va="bottom")
    plt.xlabel("P_iso (proxy)")
    plt.ylabel("E_iso (proxy)")
    plt.title("Mapa perceptual por Recording")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_rec, dpi=150)
    plt.close()
    out_paths.append(out_rec)
    return out_paths


def plot_soundscape(df_rec: pd.DataFrame) -> Path | None:
    if df_rec.empty or not HAS_SOUNDSCAPY_PLOTS:
        return None
    df_s = df_rec.rename(columns={"P_iso": "ISOPleasant", "E_iso": "ISOEventful"}).copy()
    df_s["LocationID"] = df_s["Recording"]
    out_path = PLOT_DIR / "soundscapy_lugares.png"
    ax = scatter_plot(
        df_s,
        x="ISOPleasant",
        y="ISOEventful",
        hue="LocationID",
        title="Comparacion perceptual de grabaciones (proxy)",
        backend=Backend.SEABORN,
    )
    fig = ax.get_figure() if hasattr(ax, "get_figure") else ax
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_location_comparisons(df_rec: pd.DataFrame) -> Path | None:
    if df_rec.shape[0] < 2 or not HAS_SOUNDSCAPY_PLOTS:
        return None
    df_s = df_rec.rename(columns={"P_iso": "ISOPleasant", "E_iso": "ISOEventful"}).copy()
    df_s["LocationID"] = df_s["Recording"]
    out_path = PLOT_DIR / "soundscapy_lugares_multiple.png"
    ax = density_plot(
        df_s,
        x="ISOPleasant",
        y="ISOEventful",
        hue="LocationID",
        title="Comparacion perceptual por audio (Soundscapy)",
        backend=Backend.SEABORN,
    )
    fig = ax.get_figure() if hasattr(ax, "get_figure") else ax
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_fuentes_loudness(df_seg: pd.DataFrame) -> Path:
    out_path = PLOT_DIR / "yamnet_fuentes_loudness.png"
    df_fuentes = (
        df_seg.groupby("Clase_YAMNet")[["loudness_sones", "sharpness_acum", "roughness_asper", "tonality_tnr_db"]]
        .mean()
        .sort_values(by="loudness_sones", ascending=False)
    )
    df_fuentes.to_csv(Path("results") / "yamnet_fuentes_psico.csv")
    df_fuentes.head(10).plot(kind="bar", figsize=(14, 6))
    plt.title("Fuentes YAMNet ordenadas por loudness (top 10)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main() -> None:
    args = parse_args()
    hop = args.hop or args.window

    if args.files:
        audio_paths = select_files(args.files)
        df_seg = process_audios(audio_paths, args.window, hop, args.model_handle, args.hub_cache)
    else:
        df_seg = load_csv(args.csv)

    df = compute_iso_proxies(df_seg)
    df_rec = aggregate_by_recording(df)

    out_csv_rec = Path("results") / "yamnet_psico_recordings.csv"
    out_csv_rec.parent.mkdir(parents=True, exist_ok=True)
    df_rec.to_csv(out_csv_rec, index=False)

    plots = []
    plots.append(plot_sources(df))
    plots.append(plot_fuentes_loudness(df_seg))
    plots.append(plot_descriptor_bars(df))
    pc = plot_compare_recordings(df_rec)
    if pc:
        plots.append(pc)
    plots.append(plot_correlations(df))
    plots.append(plot_clusters(df))
    plots.extend(plot_perceptual(df, df_rec))
    sc = plot_soundscape(df_rec) if HAS_SOUNDSCAPY_PLOTS else plot_soundscape_fallback(df_rec)
    if sc:
        plots.append(sc)
    lc = plot_location_comparisons(df_rec) if HAS_SOUNDSCAPY_PLOTS else plot_location_comparisons_fallback(df_rec)
    if lc:
        plots.append(lc)

    print("\nAnalisis completado.")
    print(f"CSV por recording: {out_csv_rec}")
    print("Graficos generados:")
    for p in plots:
        print(f"  - {p}")
    if not HAS_SOUNDSCAPY_PLOTS:
        print("[aviso] SoundscapePlot/LocationComparisons no disponibles; se usaron graficos fallback.")


if __name__ == "__main__":
    main()
