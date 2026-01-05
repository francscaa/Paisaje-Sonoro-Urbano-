from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import config


def compute_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia Haversine en metros entre dos puntos."""
    r = 6371000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return r * c


def _cumulative_distance_from_coords(df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> pd.Series:
    """Calcula distancia acumulada en metros usando lat/lon ordenados."""
    if df.empty or lat_col not in df or lon_col not in df:
        return pd.Series(dtype=float)
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    dist = [0.0]
    for i in range(1, len(df)):
        if pd.isna(lat.iat[i]) or pd.isna(lon.iat[i]) or pd.isna(lat.iat[i - 1]) or pd.isna(lon.iat[i - 1]):
            dist.append(dist[-1])
            continue
        dist.append(dist[-1] + compute_distance_m(lat.iat[i - 1], lon.iat[i - 1], lat.iat[i], lon.iat[i]))
    return pd.Series(dist, index=df.index, dtype=float)


def assign_uts_id(df: pd.DataFrame, uts_size_m: float) -> pd.DataFrame:
    """Asigna uts_id = floor(distancia_m / uts_size_m)."""
    out = df.copy()
    if uts_size_m is None or uts_size_m <= 0:
        out["uts_id"] = pd.NA
        return out

    dist = pd.to_numeric(out["distancia_m"], errors="coerce") if "distancia_m" in out else pd.Series(dtype=float)
    if dist.empty or dist.isna().all():
        # Si no hay distancia pero hay GPS, la calculamos
        ordered = out
        if "AbsTime" in out.columns:
            ordered = out.sort_values("AbsTime")
        elif "Timestamp" in out.columns:
            ordered = out.sort_values("Timestamp")
        dist = _cumulative_distance_from_coords(ordered)
        ordered["distancia_m"] = dist
        out = ordered

    dist = pd.to_numeric(out.get("distancia_m", pd.Series(dtype=float)), errors="coerce")
    out["uts_id"] = pd.Series(np.floor(dist / uts_size_m), index=out.index).astype("Int64")
    return out


def aggregate_by_uts(
    df: pd.DataFrame,
    uts_col: str = "uts_id",
    route_col: str = "Recording",
    class_col: str = "Clase_YAMNet",
    prob_col: str = "Probabilidad",
) -> pd.DataFrame:
    """Agrega métricas por UTS (y ruta si está presente)."""
    if uts_col not in df.columns:
        print("[aviso] No se encontró uts_id para agregar.")
        return pd.DataFrame()

    group_keys: list[str] = []
    if route_col in df.columns:
        group_keys.append(route_col)
    group_keys.append(uts_col)

    gdf = df.dropna(subset=[uts_col])
    if gdf.empty:
        print("[aviso] uts_id está vacío; no se genera agregado UTS.")
        return pd.DataFrame()

    grouped = gdf.groupby(group_keys, dropna=True)
    rows: list[dict] = []

    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict = {}
        for i, key in enumerate(group_keys):
            row[key] = keys[i]
        row["n_samples"] = len(g)

        # Descriptores: mean, median, p10, p90, count válidos
        for d in config.DESCRIPTORS:
            s = pd.to_numeric(g[d], errors="coerce") if d in g else pd.Series(dtype=float)
            row[f"{d}_mean"] = s.mean()
            row[f"{d}_median"] = s.median()
            row[f"{d}_p10"] = s.quantile(0.10)
            row[f"{d}_p90"] = s.quantile(0.90)
            row[f"{d}_count"] = s.count()

        # Distancia representativa
        if "distancia_m" in g:
            s = pd.to_numeric(g["distancia_m"], errors="coerce")
            row["distancia_m_mean"] = s.mean()
            row["distancia_m_median"] = s.median()

        # Punto representativo GPS
        lat = pd.to_numeric(g["lat"], errors="coerce") if "lat" in g else pd.Series(dtype=float)
        lon = pd.to_numeric(g["lon"], errors="coerce") if "lon" in g else pd.Series(dtype=float)
        row["lat_uts"] = lat.median()
        row["lon_uts"] = lon.median()
        row["lat_mean"] = lat.mean()
        row["lon_mean"] = lon.mean()

        # YAMNet: top1 dominante y top3 con porcentajes
        if class_col in g and not g[class_col].isna().all():
            vc = g[class_col].value_counts(dropna=True)
            total = vc.sum()
            probs = g.groupby(class_col)[prob_col].mean() if prob_col in g else None
            top_classes = vc.index.tolist()
            top_counts = vc.values.tolist()
            row["yamnet_top1"] = top_classes[0] if top_classes else pd.NA
            row["yamnet_top1_pct"] = float(top_counts[0] / total) if total and top_counts else pd.NA
            for i in range(3):
                cls = top_classes[i] if i < len(top_classes) else pd.NA
                row[f"top{i+1}_class"] = cls
                if total and i < len(top_counts):
                    row[f"top{i+1}_pct"] = float(top_counts[i] / total)
                else:
                    row[f"top{i+1}_pct"] = pd.NA
                if probs is not None and cls in probs:
                    row[f"top{i+1}_prob_mean"] = float(probs[cls])
                else:
                    row[f"top{i+1}_prob_mean"] = pd.NA
        else:
            row["yamnet_top1"] = pd.NA
            row["yamnet_top1_pct"] = pd.NA
            for i in range(3):
                row[f"top{i+1}_class"] = pd.NA
                row[f"top{i+1}_pct"] = pd.NA
                row[f"top{i+1}_prob_mean"] = pd.NA

        rows.append(row)

    return pd.DataFrame(rows)
