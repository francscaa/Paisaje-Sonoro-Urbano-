from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from visualization.plots_correlacion import parse_time_group, normalize_recording_name


def _resolve_column(df: pd.DataFrame, base: str, extras: list[str]) -> str | None:
    candidates = [base] + extras
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_distance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dist_col = _resolve_column(out, "distancia_m_mean", ["distancia_m"])
    if dist_col:
        dist = pd.to_numeric(out[dist_col], errors="coerce")
        if dist.notna().any():
            total = dist.max() if dist.max() > 0 else 1.0
            out["dist_pct"] = (dist / total).clip(0, 1) * 100
        else:
            out["dist_pct"] = pd.NA
    elif "uts_id" in out:
        out["dist_pct"] = (pd.to_numeric(out["uts_id"], errors="coerce") / out["uts_id"].max()) * 100
        print("[aviso] Sin columna distancia_m; se usa uts_id como proxy de distancia.")
    else:
        out["dist_pct"] = pd.NA
    return out


def plot_longitudinal_by_time(
    df: pd.DataFrame,
    descriptor: str,
    run_slug: str,
    out_dir: Path,
    n_points: int = 100,
    include_unknown: bool = False,
) -> Path | None:
    """
    Traza el perfil espacial (0-100%) de un descriptor por time_group.
    Usa interpolación lineal sobre dist_pct para manejar diferentes longitudes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{descriptor}_longitudinal_by_time.png"
    if df is None or df.empty:
        print(f"[aviso] No hay datos para grafico longitudinal de {descriptor}.")
        return None

    print(f"[info] Grafico longitudinal: columnas disponibles = {list(df.columns)}")
    descriptor_candidates = [descriptor, f"{descriptor}_mean", f"{descriptor}_median", f"{descriptor}_avg"]
    descriptor_col = _resolve_column(df, descriptor, descriptor_candidates[1:])
    print(f"[info] Se buscaron columnas para {descriptor}: {descriptor_candidates}; encontrada: {descriptor_col}")
    if descriptor_col != descriptor:
        if descriptor_col:
            print(f"[info] Usando columna {descriptor_col} para descriptor solicitado {descriptor}.")
    if descriptor_col is None:
        print(f"[aviso] No se encontró columna para descriptor {descriptor}.")
        return None
    # Bandas y mediana opcionales
    p10_col = _resolve_column(df, f"{descriptor}_p10", [])
    p90_col = _resolve_column(df, f"{descriptor}_p90", [])
    median_col = _resolve_column(df, f"{descriptor}_median", [])
    print(f"[info] Columnas de banda: p10={p10_col}, p90={p90_col}; mediana={median_col}")

    base = _normalize_distance(df)
    if base["dist_pct"].isna().all():
        print(f"[aviso] No hay distancia para grafico longitudinal de {descriptor}.")
        return None
    if "Recording" not in base.columns:
        print("[aviso] No hay columna Recording; no se genera grafico longitudinal.")
        return None

    base["Recording_norm"] = base["Recording"].apply(normalize_recording_name)
    base["time_group"] = base["Recording"].apply(parse_time_group)
    print(f"[info] (longitudinal) Ejemplos Recording->norm->time_group: {list(base[['Recording','Recording_norm','time_group']].head(10).to_records(index=False))}")
    print(f"[info] (longitudinal) Conteo por time_group antes de filtro: {base['time_group'].value_counts().to_dict()}")
    if not include_unknown:
        unknown_n = (base["time_group"] == "UNKNOWN").sum()
        if unknown_n:
            print(f"[aviso] Excluyendo UNKNOWN en longitudinal (n={unknown_n}). Usa include_unknown=True para incluirlos.")
        base = base[base["time_group"] != "UNKNOWN"]
    if base.empty:
        print("[aviso] Sin datos tras filtrar UNKNOWN para longitudinal.")
        return None
    x_ref = np.linspace(0, 100, n_points)
    plt.figure(figsize=(9, 5))
    groups = list(base.groupby("time_group"))
    groups.append(("GLOBAL", base))
    for grp, gdf in groups:
        cols = ["dist_pct", descriptor_col]
        if p10_col:
            cols.append(p10_col)
        if p90_col:
            cols.append(p90_col)
        if median_col:
            cols.append(median_col)
        gdf = gdf[cols].dropna(subset=["dist_pct", descriptor_col]).sort_values("dist_pct")
        if gdf.empty:
            continue
        # eliminar duplicados en dist_pct para interp1d
        gdf = gdf.groupby("dist_pct", as_index=False).mean(numeric_only=True)
        f = interp1d(
            gdf["dist_pct"],
            gdf[descriptor_col],
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )
        y_interp = f(x_ref)
        plt.plot(x_ref, y_interp, label=f"{grp} (N={len(gdf)})", linewidth=2)

        # Banda p10-p90
        if p10_col and p90_col and p10_col in gdf and p90_col in gdf:
            f_p10 = interp1d(
                gdf["dist_pct"],
                gdf[p10_col],
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
            )
            f_p90 = interp1d(
                gdf["dist_pct"],
                gdf[p90_col],
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
            )
            plt.fill_between(x_ref, f_p10(x_ref), f_p90(x_ref), alpha=0.12, label=None)

        # Línea mediana
        if median_col and median_col in gdf:
            f_med = interp1d(
                gdf["dist_pct"],
                gdf[median_col],
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
            )
            plt.plot(x_ref, f_med(x_ref), linestyle="--", linewidth=1, alpha=0.8, label=None)

    plt.xlabel("Distancia normalizada del recorrido (%)")
    plt.ylabel(descriptor)
    plt.title(f"Perfil espacial de {descriptor} por momento del día")
    plt.suptitle(f"{run_slug} | Unidad: UTS | Interpolado a {n_points} puntos", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Momento del día", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png
