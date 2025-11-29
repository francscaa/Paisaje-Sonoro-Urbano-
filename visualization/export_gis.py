from __future__ import annotations

from pathlib import Path
from typing import Any

import json
from copy import deepcopy

import pandas as pd


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = pd.NA
    return out


def export_csv_gis(df: pd.DataFrame, path: Path) -> Path:
    """Exporta CSV listo para GIS con columnas clave de tiempo, ubicación y acústica."""
    cols = [
        "Timestamp",
        "AbsTime",
        "lat",
        "lon",
        "alt",
        "Recording",
        "loudness_sones",
        "sharpness_acum",
        "roughness_asper",
        "tonality_tnr_db",
        "Clase_YAMNet",
        "Probabilidad",
    ]
    out_df = _ensure_columns(df, cols)[cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(path, index=False)
    return path


def _nan_to_none(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def export_geojson_points(df: pd.DataFrame, path: Path) -> Path:
    """Crea un GeoJSON de puntos con todas las columnas en properties."""
    path.parent.mkdir(parents=True, exist_ok=True)
    features = []
    for _, row in df.iterrows():
        if "lat" not in row or "lon" not in row or pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue
        props = {k: _nan_to_none(v) for k, v in row.to_dict().items()}
        geom = {"type": "Point", "coordinates": [_nan_to_none(row["lon"]), _nan_to_none(row["lat"])]}
        features.append({"type": "Feature", "geometry": geom, "properties": props})
    collection = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(collection, ensure_ascii=False, indent=2))
    return path


def export_geojson_linestring(df: pd.DataFrame, path: Path, color_by: str | None = None) -> Path:
    """Crea un GeoJSON con una ruta LineString ordenada por AbsTime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if "AbsTime" in df.columns:
        ordered = df.sort_values("AbsTime")
    elif "Timestamp" in df.columns:
        ordered = df.sort_values("Timestamp")
    else:
        ordered = df
    coords = []
    values = []
    for _, row in ordered.iterrows():
        if "lat" not in row or "lon" not in row or pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue
        coords.append([_nan_to_none(row["lon"]), _nan_to_none(row["lat"]), _nan_to_none(row.get("alt"))])
        if color_by and color_by in row:
            values.append(_nan_to_none(row[color_by]))
    feature: dict[str, Any] = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {},
    }
    if color_by:
        feature["properties"]["value"] = values
        feature["properties"]["color_by"] = color_by
    collection = {"type": "FeatureCollection", "features": [feature]}
    path.write_text(json.dumps(collection, ensure_ascii=False, indent=2))
    return path
