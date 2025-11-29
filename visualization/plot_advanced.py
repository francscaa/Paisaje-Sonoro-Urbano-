from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_perceptual_map(df: pd.DataFrame, path: Path) -> Path:
    """Scatter P_iso vs E_iso coloreado por cluster_id si existe; fallback: Probabilidad."""
    if df.empty:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    color = df["cluster_id"] if "cluster_id" in df.columns else df.get("Probabilidad")
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df["P_iso"], df["E_iso"], c=color, cmap="tab10", alpha=0.7)
    plt.xlabel("P_iso (proxy)")
    plt.ylabel("E_iso (proxy)")
    plt.title("Mapa perceptual P/E coloreado por cluster")
    if color is not None:
        plt.colorbar(sc, label="cluster_id" if "cluster_id" in df.columns else "Probabilidad")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_spatial_heatmap(df: pd.DataFrame, value: str, path: Path) -> Path:
    """Heatmap espacial simple usando kdeplot sobre lon/lat ponderado por un descriptor."""
    data = df.dropna(subset=["lat", "lon"])
    if data.empty or value not in df.columns:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=data, x="lon", y="lat", weights=data[value], fill=True, cmap="magma", thresh=0.05, alpha=0.8)
    plt.scatter(data["lon"], data["lat"], s=10, c="white", alpha=0.3)
    plt.xlabel("Lon")
    plt.ylabel("Lat")
    plt.title(f"Heatmap espacial de {value}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_spatial_clusters(df: pd.DataFrame, path: Path) -> Path:
    """Mapa espacial coloreado por cluster_id."""
    data = df.dropna(subset=["lat", "lon"])
    if data.empty or "cluster_id" not in data.columns:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = data["cluster_id"]
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(data["lon"], data["lat"], c=labels, cmap="tab20", s=30, alpha=0.9)
    plt.plot(data["lon"], data["lat"], color="gray", alpha=0.3, linewidth=1)
    plt.colorbar(sc, label="cluster_id")
    plt.xlabel("Lon")
    plt.ylabel("Lat")
    plt.title("Clusters espaciales")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_spatial_perceptual(df: pd.DataFrame, path: Path) -> Path:
    """Puntos en lat/lon coloreados por magnitud perceptual (distancia en P/E)."""
    data = df.dropna(subset=["lat", "lon"])
    if data.empty or "P_iso" not in data.columns or "E_iso" not in data.columns:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    pe_norm = np.sqrt(np.square(data["P_iso"]) + np.square(data["E_iso"]))
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(data["lon"], data["lat"], c=pe_norm, cmap="viridis", s=30, alpha=0.9)
    plt.plot(data["lon"], data["lat"], color="gray", alpha=0.3, linewidth=1)
    plt.colorbar(sc, label="||P/E||")
    plt.xlabel("Lon")
    plt.ylabel("Lat")
    plt.title("Mapa espacial perceptual (magnitud P/E)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
