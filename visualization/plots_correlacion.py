from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def normalize_recording_name(recording: str) -> str:
    """Normaliza Recording: recorta, reemplaza espacios/guiones por _, colapsa múltiples _."""
    s = str(recording).strip()
    s = re.sub(r"[\s-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def parse_time_group(recording: str) -> str:
    """
    Mapea nombres Bici_* a HH:MM.
    Soporta sufijos _seg/_segment, formatos am (Bici_h_am o Bici_h_mm_am) y 24h (Bici_hh_mm).
    Retorna UNKNOWN si no matchea.
    """
    if not recording:
        return "UNKNOWN"
    norm = normalize_recording_name(recording)
    s = norm.lower()
    s = re.sub(r"_(seg|segment)$", "", s)  # tolera sufijos de segmentación
    # Formato am con o sin minutos: Bici_9_am, Bici_9_00_am
    match_am = re.fullmatch(r"bici_(\d{1,2})(?:_(\d{2}))?_am", s)
    if match_am:
        hour = int(match_am.group(1))
        minute = int(match_am.group(2) or 0)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"
    # Formato 24h: Bici_20_20, Bici_21_30
    match_hm = re.fullmatch(r"bici_(\d{1,2})_(\d{2})", s)
    if match_hm:
        hour = int(match_hm.group(1))
        minute = int(match_hm.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"
    return "UNKNOWN"


def _safe_corr(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    cols_present = [c for c in cols if c in df.columns]
    numeric_df = df[cols_present].select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr()


def plot_correlacion(df: pd.DataFrame, cols: list[str], out_png: Path, out_csv: Path, title: str) -> bool:
    """
    Genera heatmap de correlaciones y CSV. Retorna True si se guardó, False si se omitió.
    No grafica si hay menos de 5 filas o menos de 2 columnas numéricas.
    """
    if df.shape[0] < 5:
        print(f"[aviso] No se grafica correlación ({title}): menos de 5 filas.")
        return False
    corr = _safe_corr(df, cols)
    if corr.empty:
        print(f"[aviso] No se grafica correlación ({title}): columnas numéricas insuficientes.")
        return False
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_csv)
    try:
        plt.figure(figsize=(7, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        return True
    except Exception as exc:
        print(f"[aviso] Error al graficar correlación ({title}): {exc}")
        return False


def plot_correlation_matrices(df: pd.DataFrame, run_slug: str, cols: list[str], out_dir: Path) -> list[Path]:
    """
    Genera correlaciones globales y por grupo de tiempo derivado de Recording.
    Retorna lista de rutas existentes.
    """
    paths: list[Path] = []
    if df is None or df.empty:
        print("[aviso] No hay datos para correlaciones.")
        return paths

    out_dir.mkdir(parents=True, exist_ok=True)
    # Global
    global_png = out_dir / "corr_GLOBAL.png"
    global_csv = out_dir / "corr_GLOBAL.csv"
    if plot_correlacion(df, cols, global_png, global_csv, f"{run_slug} - GLOBAL (N={len(df)})"):
        paths.extend([global_png, global_csv])

    if "Recording" not in df.columns:
        print("[aviso] Sin columna Recording; no se generan correlaciones por grupo.")
        return paths

    # Por grupo horario
    df = df.copy()
    df["Recording_norm"] = df["Recording"].apply(normalize_recording_name)
    df["time_group"] = df["Recording"].apply(parse_time_group)
    # Diagnóstico: ejemplos y conteo
    sample_map = df[["Recording", "Recording_norm", "time_group"]].head(10).to_records(index=False)
    print(f"[info] Ejemplos Recording->norm->time_group: {list(sample_map)}")
    print(f"[info] Conteo por time_group: {df['time_group'].value_counts().to_dict()}")
    for grp, gdf in df.groupby("time_group"):
        label = grp.replace(":", "-")
        png = out_dir / f"corr_{label}.png"
        csv = out_dir / f"corr_{label}.csv"
        if plot_correlacion(gdf, cols, png, csv, f"{run_slug} - {grp} (N={len(gdf)})"):
            paths.extend([png, csv])
    return paths
