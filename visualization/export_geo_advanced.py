from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import pandas as pd


def _nan_to_none(val: Any) -> Any:
    if pd.isna(val):
        return None
    return val


def export_geojson_clusters(df: pd.DataFrame, path: Path) -> Path:
    """GeoJSON de puntos con cluster_id y descriptores."""
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


def export_geojson_heatmap(df: pd.DataFrame, value: str, path: Path) -> Path:
    """GeoJSON de puntos con un valor para usar en heatmaps GIS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    features = []
    for _, row in df.iterrows():
        if "lat" not in row or "lon" not in row or pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue
        props = {"value": _nan_to_none(row.get(value)), "field": value}
        geom = {"type": "Point", "coordinates": [_nan_to_none(row["lon"]), _nan_to_none(row["lat"])]}
        features.append({"type": "Feature", "geometry": geom, "properties": props})
    collection = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(collection, ensure_ascii=False, indent=2))
    return path
