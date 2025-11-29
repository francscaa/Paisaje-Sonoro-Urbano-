from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_geo(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.copy()
    for col in ["lat", "lon"]:
        if col not in ordered.columns:
            ordered[col] = pd.NA
    if "AbsTime" in ordered.columns:
        ordered = ordered.sort_values("AbsTime")
    elif "Timestamp" in ordered.columns:
        ordered = ordered.sort_values("Timestamp")
    # Si no hay columnas o valores lat/lon, devolver vacío para evitar errores.
    return ordered.dropna(subset=["lat", "lon"], how="all")


def plot_route_colored_by_descriptor(df: pd.DataFrame, descriptor: str, path: Path) -> Path:
    """Plotea la ruta GPS coloreada por un descriptor acústico."""
    data = _prepare_geo(df)
    if descriptor not in data.columns or data.empty:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    vals = data[descriptor]
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(data["lon"], data["lat"], c=vals, cmap="viridis", s=30, alpha=0.9)
    plt.plot(data["lon"], data["lat"], color="gray", alpha=0.4, linewidth=1)
    plt.colorbar(sc, label=descriptor)
    plt.xlabel("Lon")
    plt.ylabel("Lat")
    plt.title(f"Ruta coloreada por {descriptor}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_route_colored_by_class(df: pd.DataFrame, path: Path) -> Path:
    """Plotea la ruta GPS coloreada por la clase YAMNet dominante en cada segmento."""
    data = _prepare_geo(df)
    if "Clase_YAMNet" not in data.columns or data.empty:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    labels, uniques = pd.factorize(data["Clase_YAMNet"])
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(data["lon"], data["lat"], c=labels, cmap="tab20", s=30, alpha=0.9)
    plt.plot(data["lon"], data["lat"], color="gray", alpha=0.3, linewidth=1)
    handles, _ = sc.legend_elements(prop="colors", alpha=0.7)
    plt.legend(handles, uniques, title="Clase YAMNet", fontsize="small", ncol=2)
    plt.xlabel("Lon")
    plt.ylabel("Lat")
    plt.title("Ruta coloreada por clase YAMNet")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
