from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.iso_proxies import aggregate_by_recording, compute_iso_proxies
from audio_processing.load_audio import select_files
from audio_processing.pipeline import process_audios
from audio_processing.yamnet_classifier import load_yamnet_model
from config import DEFAULT_MODEL_HANDLE, RECORDINGS_CSV, SEGMENT_CSV, TFHUB_CACHE_DIR
from gps_utils.load_gps import load_gps, pick_gps_file
from gps_utils.sync_gps import sync_audio_gps
from visualization import plots_iso, plots_psico, plots_yamnet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmenta audios, calcula psico y analiza (correlaciones, clusters, graficos).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--files",
        type=Path,
        nargs="*",
        help="Rutas a uno o varios WAV/MP3. Si no se indican, se abre selector.",
    )
    parser.add_argument("--window", type=float, default=3.0, help="Ventana de analisis en segundos.")
    parser.add_argument("--hop", type=float, default=None, help="Paso entre ventanas. Por defecto = window.")
    parser.add_argument(
        "--model-handle",
        default=DEFAULT_MODEL_HANDLE,
        help="Handle de TF Hub o ruta local a modelo YAMNet (saved_model.pb).",
    )
    parser.add_argument(
        "--hub-cache",
        type=Path,
        default=TFHUB_CACHE_DIR,
        help="Cache de TF Hub / carpeta del modelo local.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=SEGMENT_CSV,
        help="CSV segmentado (si ya lo tienes y no quieres re-procesar audios).",
    )
    parser.add_argument(
        "--gps",
        type=Path,
        help="Ruta a archivo GeoJSON/GPX/KML para sincronizar GPS con segmentos (opcional).",
    )
    return parser.parse_args()


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(
            f"No existe el CSV: {csv_path}\n"
            "Primero ejecuta con --files para generar el CSV segmentado."
        )
    df = pd.read_csv(csv_path)
    if "Recording" not in df.columns:
        df["Recording"] = "audio"
    if "AbsTime" not in df.columns and "Timestamp" in df.columns:
        df["AbsTime"] = df["Timestamp"]
    return df


def resolve_model_handle(model_handle: str, hub_cache: Path) -> str:
    """Prefiere el modelo local descomprimido si existe."""
    default_local = hub_cache
    if model_handle.startswith("http") and default_local.exists() and (default_local / "saved_model.pb").exists():
        print(f"Detectado modelo local en {default_local}, usandolo en lugar de TF Hub.")
        return str(default_local)
    return model_handle


def main() -> None:
    args = parse_args()
    hop = args.hop or args.window

    df_seg = None
    audio_paths: list[Path] | None = None
    if args.files is not None:
        audio_paths = select_files(args.files if len(args.files) > 0 else None)
    if audio_paths:
        handle = resolve_model_handle(args.model_handle, args.hub_cache)
        try:
            model, class_names = load_yamnet_model(handle, args.hub_cache)
        except Exception as exc:
            raise SystemExit(
                "No se pudo cargar YAMNet.\n"
                f"Handle: {handle}\n"
                "Si es error SSL, descarga el modelo y descomprímelo en models/tfhub.\n"
                f"Detalle: {exc}"
            ) from exc
        df_seg = process_audios(audio_paths, args.window, hop, model, class_names)
    if df_seg is None:
        try:
            df_seg = load_csv(args.csv)
        except SystemExit:
            audio_paths = select_files(None)
            handle = resolve_model_handle(args.model_handle, args.hub_cache)
            model, class_names = load_yamnet_model(handle, args.hub_cache)
            df_seg = process_audios(audio_paths, args.window, hop, model, class_names)

    gps_path = pick_gps_file(args.gps)
    if gps_path:
        df_gps = load_gps(gps_path)
        if not df_gps.empty:
            df_seg = sync_audio_gps(df_seg, df_gps)
        df_seg.to_csv(SEGMENT_CSV, index=False)

    df = compute_iso_proxies(df_seg)
    df_rec = aggregate_by_recording(df)

    RECORDINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_rec.to_csv(RECORDINGS_CSV, index=False)

    plots: list[Path] = []
    plots.append(plots_yamnet.plot_sources(df))
    plots.append(plots_yamnet.plot_fuentes_loudness(df_seg))
    plots.append(plots_psico.plot_descriptor_bars(df))
    pc = plots_psico.plot_compare_recordings(df_rec)
    if pc:
        plots.append(pc)
    plots.append(plots_iso.plot_correlations(df))
    plots.append(plots_iso.plot_clusters(df))
    plots.extend(plots_iso.plot_perceptual(df, df_rec))
    sc = plots_iso.plot_soundscape(df_rec) if plots_iso.HAS_SOUNDSCAPY_PLOTS else plots_iso.plot_soundscape_fallback(df_rec)
    if sc:
        plots.append(sc)
    lc = (
        plots_iso.plot_location_comparisons(df_rec)
        if plots_iso.HAS_SOUNDSCAPY_PLOTS
        else plots_iso.plot_location_comparisons_fallback(df_rec)
    )
    if lc:
        plots.append(lc)

    print("\nAnalisis completado.")
    print(f"CSV por recording: {RECORDINGS_CSV}")
    print("Graficos generados:")
    for p in plots:
        print(f"  - {p}")
    if not plots_iso.HAS_SOUNDSCAPY_PLOTS:
        print("[aviso] SoundscapePlot/LocationComparisons no disponibles; se usaron graficos fallback.")


if __name__ == "__main__":
    main()
