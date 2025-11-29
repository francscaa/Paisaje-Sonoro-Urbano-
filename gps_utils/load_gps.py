from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

try:
    from tkinter import Tk, filedialog
except Exception:  # pragma: no cover - entorno sin GUI
    Tk = None
    filedialog = None

def _parse_geojson(path: Path) -> List[dict]:
    import geojson  # type: ignore

    with path.open() as f:
        data = geojson.load(f)
    features = data["features"] if isinstance(data, dict) else getattr(data, "features", [])
    rows = []
    for feat in features:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates", [None, None, None])
        props = feat.get("properties", {}) or {}
        rows.append(
            {
                "lon": coords[0],
                "lat": coords[1],
                "alt": coords[2] if len(coords) > 2 else None,
                "timestamp_real": props.get("time") or props.get("timestamp") or props.get("datetime"),
            }
        )
    return rows


def _parse_gpx(path: Path) -> List[dict]:
    import gpxpy  # type: ignore

    with path.open() as f:
        gpx = gpxpy.parse(f)
    rows = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                rows.append(
                    {
                        "lat": point.latitude,
                        "lon": point.longitude,
                        "alt": point.elevation,
                        "timestamp_real": point.time,
                    }
                )
    return rows


def _parse_kml(path: Path) -> List[dict]:
    from fastkml import kml  # type: ignore

    with path.open("rt", encoding="utf-8") as f:
        doc = f.read()
    k = kml.KML()
    k.from_string(doc.encode("utf-8"))
    rows = []

    def _iter_features(feat):
        if hasattr(feat, "features"):
            for f in feat.features():
                yield from _iter_features(f)
        else:
            yield feat

    for feat in _iter_features(k):
        geom = getattr(feat, "geometry", None)
        if geom is None:
            continue
        coords = list(getattr(geom, "coords", []))
        if not coords:
            continue
        lon, lat, *rest = coords[0]
        alt = rest[0] if rest else None
        ts = getattr(feat, "timestamp", None)
        ts_when = getattr(ts, "when", None) if ts else None
        rows.append({"lat": lat, "lon": lon, "alt": alt, "timestamp_real": ts_when})
    return rows


def load_gps(path: Path) -> pd.DataFrame:
    """Carga datos GPS desde GeoJSON/GPX/KML y agrega tiempo relativo."""
    suffix = path.suffix.lower()
    if suffix == ".geojson" or suffix == ".json":
        rows = _parse_geojson(path)
    elif suffix == ".gpx":
        rows = _parse_gpx(path)
    elif suffix == ".kml":
        rows = _parse_kml(path)
    else:
        raise SystemExit(f"Formato GPS no soportado: {suffix}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("[aviso] No se encontraron puntos GPS.")
        return df

    df["timestamp_real"] = pd.to_datetime(df["timestamp_real"], errors="coerce")
    if df["timestamp_real"].isna().all():
        print("[aviso] GPS sin timestamps; no se puede sincronizar por tiempo.")
        df["t_seconds"] = pd.NA
        return df

    first_ts = df["timestamp_real"].dropna().iloc[0]
    df["t_seconds"] = (df["timestamp_real"] - first_ts).dt.total_seconds()
    return df[["lat", "lon", "alt", "timestamp_real", "t_seconds"]]


def pick_gps_file(path: Path | None) -> Path | None:
    """Devuelve la ruta GPS proporcionada o abre un selector si es posible."""
    if path:
        return path
    if Tk is None or filedialog is None:
        return None
    Tk().withdraw()
    picked = filedialog.askopenfilename(
        title="Selecciona archivo GPS (GeoJSON/GPX/KML)",
        filetypes=[("GeoJSON", "*.geojson *.json"), ("GPX", "*.gpx"), ("KML", "*.kml"), ("Todos", "*.*")],
    )
    return Path(picked) if picked else None
