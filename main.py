from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np

import config
from analysis.iso_proxies import aggregate_by_recording, compute_iso_proxies
from analysis.spatial_perceptual import compute_perceptual_space, compute_spatial_clusters, join_space_perceptual
from audio_processing.load_audio import select_files
from audio_processing.pipeline import process_audios
from audio_processing.yamnet_classifier import load_yamnet_model
from gps_utils.load_gps import load_gps, pick_gps_file, pick_gps_files
from gps_utils.sync_gps import sync_audio_gps
from gps_utils.uts import aggregate_by_uts, assign_uts_id
from visualization import plots_iso, plots_psico, plots_yamnet
from visualization.export_geo_advanced import export_geojson_clusters, export_geojson_heatmap
from visualization.export_gis import export_csv_gis, export_geojson_linestring, export_geojson_points
from visualization.plot_advanced import (
    plot_perceptual_map,
    plot_spatial_clusters,
    plot_spatial_heatmap,
    plot_spatial_perceptual,
)
from visualization.plot_spatial import plot_route_colored_by_class, plot_route_colored_by_descriptor
from visualization.plots_correlacion import parse_time_group, plot_correlation_matrices
from visualization.plots_comparisons import plot_distributions_by_time
from visualization.plots_longitudinal import plot_longitudinal_by_time


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
        default=config.DEFAULT_MODEL_HANDLE,
        help="Handle de TF Hub o ruta local a modelo YAMNet (saved_model.pb).",
    )
    parser.add_argument(
        "--hub-cache",
        type=Path,
        default=config.TFHUB_CACHE_DIR,
        help="Cache de TF Hub / carpeta del modelo local.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV segmentado (si ya lo tienes y no quieres re-procesar audios).",
    )
    parser.add_argument(
        "--gps",
        type=Path,
        nargs="*",
        help="Ruta a archivo GeoJSON/GPX/KML para sincronizar GPS con segmentos (opcional).",
    )
    parser.add_argument(
        "--uts-meters",
        type=float,
        default=30.0,
        help="Longitud del tramo UTS en metros (ej: 30, 50, 100).",
    )
    parser.add_argument(
        "--compare-uts-a",
        type=Path,
        default=None,
        help="Ruta a yamnet_psico_uts.csv del recorrido A (ej: mañana).",
    )
    parser.add_argument(
        "--compare-uts-b",
        type=Path,
        default=None,
        help="Ruta a yamnet_psico_uts.csv del recorrido B (ej: noche).",
    )
    parser.add_argument(
        "--debug-plots",
        action="store_true",
        help="Genera plots de depuracion/legacy (segmentos). Si no se indica, se omiten.",
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


def pick_run_name(audio_paths: Iterable[Path] | None, csv_path: Path | None) -> str:
    """Usa el nombre del audio (o CSV) para crear la carpeta de resultados."""
    if audio_paths:
        audio_list = list(audio_paths)
        if len(audio_list) > 1:
            base = "comparacion"
            existing = []
            if config.RESULTS_ROOT.exists():
                existing = [p.name for p in config.RESULTS_ROOT.glob(f"{base}_*") if p.is_dir()]
            nums = []
            for name in existing:
                m = re.match(rf"{base}_(\d+)$", name)
                if m:
                    nums.append(int(m.group(1)))
            next_num = max(nums) + 1 if nums else 1
            return f"{base}_{next_num}"
        first = audio_list[0] if audio_list else None
        if first:
            return first.stem
    if csv_path:
        parent = csv_path.parent
        if parent.name == "results" and parent.parent.name not in (".", "", "results"):
            return parent.parent.name
        if parent.name not in (".", "", "results"):
            return parent.name
        return csv_path.stem
    return "run"


def prepare_uts_for_spatial(df_uts: pd.DataFrame) -> pd.DataFrame:
    """Mapea columnas de agregados UTS a las esperadas por funciones espaciales."""
    out = df_uts.copy()
    # Usa el centroide como lat/lon
    if "lat_uts" in out.columns:
        out["lat"] = out["lat_uts"]
    elif "lat_mean" in out.columns:
        out["lat"] = out["lat_mean"]
    if "lon_uts" in out.columns:
        out["lon"] = out["lon_uts"]
    elif "lon_mean" in out.columns:
        out["lon"] = out["lon_mean"]
    # Usa promedios MOSQITO como columnas base para proxies y clusters
    for d in config.DESCRIPTORS:
        mean_col = f"{d}_mean"
        if mean_col in out.columns:
            out[d] = out[mean_col]
    if "distancia_m_mean" in out.columns:
        out["distancia_m"] = out["distancia_m_mean"]
    # Orden para ploteo si no hay tiempo real
    if "Timestamp" not in out.columns:
        out["Timestamp"] = out.get("uts_id")
    return out


def compare_uts(
    uts_a: pd.DataFrame, uts_b: pd.DataFrame, label_a: str, label_b: str, plots_dir: Path, out_csv: Path
) -> pd.DataFrame:
    """Alinea por uts_id y calcula deltas entre dos recorridos."""
    if uts_a.empty or uts_b.empty:
        print("[aviso] UTS vacíos para comparar.")
        return pd.DataFrame()
    df_a = uts_a.copy()
    df_b = uts_b.copy()
    suffixes = (f"_{label_a}", f"_{label_b}")
    merged = pd.merge(df_a, df_b, on="uts_id", how="outer", suffixes=suffixes)

    # Distancia de referencia: promedio de ambas columnas si existen, si no usa uts_id
    dist_cols = []
    for suf in suffixes:
        col = f"distancia_m_mean{suf}"
        if col in merged.columns:
            dist_cols.append(col)
    if dist_cols:
        merged["distancia_m_ref"] = merged[dist_cols].mean(axis=1, skipna=True)
    else:
        merged["distancia_m_ref"] = merged["uts_id"]

    # Deltas para descriptores
    for d in config.DESCRIPTORS:
        col_a = f"{d}_mean{suffixes[0]}"
        col_b = f"{d}_mean{suffixes[1]}"
        if col_a not in merged or col_b not in merged:
            merged[f"delta_{d}"] = pd.NA
            continue
        merged[f"delta_{d}"] = pd.to_numeric(merged[col_a], errors="coerce") - pd.to_numeric(
            merged[col_b], errors="coerce"
        )

    # Coordenadas promedio para mapa
    lat_cols = [f"lat_mean{s}" for s in suffixes if f"lat_mean{s}" in merged]
    lon_cols = [f"lon_mean{s}" for s in suffixes if f"lon_mean{s}" in merged]
    if lat_cols:
        merged["lat_mean_avg"] = merged[lat_cols].mean(axis=1, skipna=True)
    if lon_cols:
        merged["lon_mean_avg"] = merged[lon_cols].mean(axis=1, skipna=True)

    # Export CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    # Plots de delta
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[aviso] matplotlib no disponible para plot de comparación.")
        return merged

    # (1) Línea delta vs distancia (todas las métricas)
    sorted_df = merged.sort_values("distancia_m_ref")
    plt.figure(figsize=(8, 5))
    for d in config.DESCRIPTORS:
        delta_col = f"delta_{d}"
        if delta_col not in sorted_df:
            continue
        plt.plot(sorted_df["distancia_m_ref"], sorted_df[delta_col], marker="o", label=delta_col)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Distancia (m, ref)")
    plt.ylabel("Delta (A - B)")
    plt.title(f"Deltas por UTS ({label_a} - {label_b})")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(plots_dir / "delta_vs_distancia.png", dpi=150)
    plt.close()
    # (1b) Curvas originales por recorrido (ej. loudness)
    loud_a = f"loudness_sones_mean{suffixes[0]}"
    loud_b = f"loudness_sones_mean{suffixes[1]}"
    if loud_a in sorted_df and loud_b in sorted_df:
        plt.figure(figsize=(8, 5))
        plt.plot(
            sorted_df["distancia_m_ref"],
            sorted_df[loud_a],
            marker="o",
            label=f"loudness_sones ({label_a})",
        )
        plt.plot(
            sorted_df["distancia_m_ref"],
            sorted_df[loud_b],
            marker="o",
            linestyle="--",
            label=f"loudness_sones ({label_b})",
        )
        plt.xlabel("Distancia (m, ref)")
        plt.ylabel("Loudness (sones)")
        plt.title(f"Loudness por UTS: {label_a} vs {label_b}")
        plt.legend(fontsize="small")
        plt.tight_layout()
        plt.savefig(plots_dir / "loudness_vs_distancia.png", dpi=150)
        plt.close()

    # (2) Mapa delta_loudness
    if "lat_mean_avg" in sorted_df and "lon_mean_avg" in sorted_df:
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            sorted_df["lon_mean_avg"], sorted_df["lat_mean_avg"], c=sorted_df["delta_loudness_sones"], cmap="coolwarm"
        )
        plt.colorbar(sc, label="delta_loudness_sones (A - B)")
        plt.xlabel("Lon")
        plt.ylabel("Lat")
        plt.title("Mapa delta loudness por UTS")
        plt.tight_layout()
        plt.savefig(plots_dir / "map_delta_loudness.png", dpi=150)
        plt.close()

    return merged


def main() -> None:
    args = parse_args()
    hop = args.hop or args.window

    df_seg = None
    compare_only = args.compare_uts_a and args.compare_uts_b and not args.files and args.csv is None
    audio_paths: list[Path] | None = None
    if args.files is not None:
        audio_paths = select_files(args.files if len(args.files) > 0 else None)
    run_name = pick_run_name(audio_paths, args.csv)
    config.set_run_results_dir(run_name)
    segment_csv = args.csv or config.SEGMENT_CSV
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
    if df_seg is None and not args.compare_uts_a and not args.compare_uts_b:
        try:
            df_seg = load_csv(segment_csv)
        except SystemExit:
            audio_paths = select_files(None)
            run_name = pick_run_name(audio_paths, args.csv)
            config.set_run_results_dir(run_name)
            segment_csv = config.SEGMENT_CSV
            handle = resolve_model_handle(args.model_handle, args.hub_cache)
            model, class_names = load_yamnet_model(handle, args.hub_cache)
            df_seg = process_audios(audio_paths, args.window, hop, model, class_names)

    run_slug = config.RUN_DIR.name
    gps_paths: list[Path] = []
    if not compare_only:
        if args.gps is None:
            gps_paths = pick_gps_files(None)
        else:
            gps_paths = list(args.gps)

    if df_seg is not None:
        if gps_paths:
            if "Recording" not in df_seg.columns:
                print("[aviso] No hay columna Recording; se aplicará el primer GPS a todos los segmentos.")
                gps_paths = gps_paths[:1]
            recordings = list(dict.fromkeys(df_seg["Recording"])) if "Recording" in df_seg else [None]
            if len(gps_paths) < len(recordings):
                print(f"[aviso] Hay {len(recordings)} recordings pero solo {len(gps_paths)} GPS; se reutiliza el último GPS.")
            if len(gps_paths) > len(recordings):
                print(f"[aviso] Hay más GPS ({len(gps_paths)}) que recordings ({len(recordings)}); se ignorarán los extra.")
            synced_parts = []
            for idx, rec in enumerate(recordings):
                gp_idx = min(idx, len(gps_paths) - 1)
                gps_path = gps_paths[gp_idx]
                part = df_seg[df_seg["Recording"] == rec] if rec is not None else df_seg
                df_gps = load_gps(gps_path)
                if not df_gps.empty:
                    part = sync_audio_gps(part, df_gps)
                else:
                    print(f"[aviso] GPS vacío para {gps_path}, se mantienen coordenadas vacías en Recording={rec}.")
                part = assign_uts_id(part, args.uts_meters)
                synced_parts.append(part)
            df_seg = pd.concat(synced_parts).sort_index()
        else:
            df_seg = assign_uts_id(df_seg, args.uts_meters)
        df_seg.to_csv(segment_csv, index=False)

    # Agregado por tramo UTS
    uts_csv = config.RESULTS_DIR / "yamnet_psico_uts.csv"
    df_uts = pd.DataFrame()
    df_uts_geo = pd.DataFrame()
    df_uts_gis = pd.DataFrame()
    if df_seg is not None:
        df_uts = aggregate_by_uts(df_seg)
        if not df_uts.empty:
            uts_csv.parent.mkdir(parents=True, exist_ok=True)
            df_uts.to_csv(uts_csv, index=False)
            df_uts_geo = prepare_uts_for_spatial(df_uts)
            df_uts_gis = df_uts_geo.copy()
            export_csv_gis(df_uts_gis, config.RESULTS_DIR / "gis_ready_uts.csv")
            export_geojson_points(df_uts_gis, config.RESULTS_DIR / "gis_points_uts.geojson")

    # Salidas GIS viven junto al resto de resultados de la corrida: results/<nombre_audio>/
    gis_ready = config.RESULTS_DIR / f"gis_ready_{run_slug}.csv"
    geo_points = config.RESULTS_DIR / "gis_points.geojson"
    geo_route = config.RESULTS_DIR / "gis_route.geojson"
    if df_seg is not None:
        export_csv_gis(df_seg, gis_ready)
        export_geojson_points(df_seg, geo_points)
        export_geojson_linestring(df_seg, geo_route)
        # Plots de segmentos (legacy) para comparación
        if args.debug_plots:
            plots_segmentos = config.PLOT_DIR / "plots_segmentos"
            plots_segmentos.mkdir(parents=True, exist_ok=True)
            plot_route_colored_by_descriptor(df_seg, "loudness_sones", plots_segmentos / "route_loudness.png")
            plot_route_colored_by_class(df_seg, plots_segmentos / "route_classes.png")

    # Preparar datasets segmento y UTS para psico/espacial
    df_seg_iso = pd.DataFrame()
    df_rec = pd.DataFrame()
    if df_seg is not None:
        df_seg_iso = compute_iso_proxies(df_seg)
        df_seg_iso = join_space_perceptual(df_seg_iso)
        df_seg_iso = compute_spatial_clusters(df_seg_iso, method="kmeans")
        df_rec = aggregate_by_recording(df_seg_iso)

    df_uts_iso = pd.DataFrame()
    if not df_uts.empty:
        if df_uts_geo.empty:
            df_uts_geo = prepare_uts_for_spatial(df_uts)
        df_uts_iso = compute_iso_proxies(df_uts_geo)
        df_uts_iso = join_space_perceptual(df_uts_iso)
        df_uts_iso = compute_spatial_clusters(df_uts_iso, method="kmeans")

    # Dataset principal de análisis: UTS si existe, si no segmentos
    df_main_iso = df_uts_iso if not df_uts_iso.empty else df_seg_iso

    # Visualizaciones avanzadas (segmentos) y nuevas (UTS)
    if not df_uts_iso.empty:
        plot_route_colored_by_descriptor(df_uts_iso, "loudness_sones", config.PLOT_DIR / "route_loudness.png")
    elif not df_seg_iso.empty:
        plot_route_colored_by_descriptor(df_seg_iso, "loudness_sones", config.PLOT_DIR / "route_loudness.png")

    if not df_main_iso.empty:
        plot_perceptual_map(df_main_iso, config.PLOT_DIR / "perceptual_map_clusters.png")
        plot_spatial_heatmap(df_main_iso, "loudness_sones", config.PLOT_DIR / "spatial_heatmap_loudness.png")
        plot_spatial_clusters(df_main_iso, config.PLOT_DIR / "spatial_clusters.png")
        plot_spatial_perceptual(df_main_iso, config.PLOT_DIR / "spatial_perceptual.png")

    if not df_seg_iso.empty and args.debug_plots:
        plot_spatial_clusters(df_seg_iso, plots_segmentos / "spatial_clusters.png")

    # Correlaciones por momento del día (prioriza UTS)
    # Excluimos tonality_tnr_db en gráficos por falta de variabilidad en este caso de estudio.
    corr_cols = ["loudness_sones", "sharpness_acum", "roughness_asper", "P_iso", "E_iso"]
    corr_df = None
    if not df_uts_iso.empty:
        corr_df = df_uts_iso
    elif not df_main_iso.empty:
        corr_df = df_main_iso
    elif not df_uts.empty:
        corr_df = df_uts
    elif df_seg is not None:
        corr_df = df_seg
        print("[aviso] Correlaciones con segmentos temporales (no UTS); puede haber sesgo por velocidad.")
    if corr_df is not None:
        plot_correlation_matrices(corr_df, run_slug, corr_cols, config.PLOT_DIR / "correlations")

    # Distribuciones por momento del día (UTS si existe)
    dist_df = None
    if not df_uts.empty:
        dist_df = df_uts
    elif df_uts_iso is not None and not df_uts_iso.empty:
        dist_df = df_uts_iso
    if dist_df is not None:
        plot_distributions_by_time(
            dist_df,
            ["loudness_sones", "sharpness_acum", "roughness_asper"],
            run_slug,
            config.PLOT_DIR / "comparisons",
            include_unknown=False,
        )
        plot_longitudinal_by_time(
            dist_df,
            "loudness_sones",
            run_slug,
            config.PLOT_DIR / "comparisons",
            include_unknown=False,
        )

    # Comparación entre dos recorridos por UTS, si se proporcionan rutas
    if args.compare_uts_a and args.compare_uts_b:
        try:
            uts_a = pd.read_csv(args.compare_uts_a)
            uts_b = pd.read_csv(args.compare_uts_b)
        except Exception as exc:
            raise SystemExit(f"No se pudieron leer CSV de comparación: {exc}") from exc
        label_a = args.compare_uts_a.parent.name or args.compare_uts_a.stem
        label_b = args.compare_uts_b.parent.name or args.compare_uts_b.stem
        if label_a == label_b:
            label_a = f"{label_a}_a"
            label_b = f"{label_b}_b"
        comp_dir = config.RESULTS_DIR / "plots_comparacion"
        comp_csv = config.RESULTS_DIR / "comparacion_uts.csv"
        compare_uts(uts_a, uts_b, label_a, label_b, comp_dir, comp_csv)

    # Exportaciones GeoJSON avanzadas
    if not df_main_iso.empty:
        export_geojson_clusters(df_main_iso, config.RESULTS_DIR / "gis_clusters.geojson")
        export_geojson_heatmap(df_main_iso, "loudness_sones", config.RESULTS_DIR / "gis_heatmap_loudness.geojson")

    if not df_seg_iso.empty:
        config.RECORDINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_rec.to_csv(config.RECORDINGS_CSV, index=False)

        plots: list[Path] = []
        plots.append(plots_yamnet.plot_sources(df_seg_iso))
        pc = plots_psico.plot_compare_recordings(df_rec)
        if pc:
            plots.append(pc)
        plots.append(plots_iso.plot_correlations(df_seg_iso))
        plots.append(plots_iso.plot_clusters(df_seg_iso))
        plots.extend(plots_iso.plot_perceptual(df_seg_iso, df_rec))
        sc = (
            plots_iso.plot_soundscape(df_rec)
            if plots_iso.HAS_SOUNDSCAPY_PLOTS
            else plots_iso.plot_soundscape_fallback(df_rec)
        )
        if sc:
            plots.append(sc)
        lc = (
            plots_iso.plot_location_comparisons(df_rec)
            if plots_iso.HAS_SOUNDSCAPY_PLOTS
            else plots_iso.plot_location_comparisons_fallback(df_rec)
        )
        if lc:
            plots.append(lc)

        annex_dir = config.PLOT_DIR / "annex"
        annex_dir.mkdir(parents=True, exist_ok=True)
        moved: list[Path] = []
        for p in plots:
            dest = annex_dir / p.name
            try:
                p.rename(dest)
                moved.append(dest)
            except Exception:
                moved.append(p)

        print("\nAnalisis completado.")
        print(f"CSV por recording: {config.RECORDINGS_CSV}")
        print("Graficos generados:")
        for p in moved:
            print(f"  - {p}")
        if not plots_iso.HAS_SOUNDSCAPY_PLOTS:
            print("[aviso] SoundscapePlot/LocationComparisons no disponibles; se usaron graficos fallback.")


if __name__ == "__main__":
    main()
