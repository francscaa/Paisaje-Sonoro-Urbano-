from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.clustering import assign_clusters
from analysis.correlations import correlation_matrix
from config import PLOT_DIR

try:
    from soundscapy.plotting import Backend, density_plot, scatter_plot

    HAS_SOUNDSCAPY_PLOTS = True
except Exception:
    HAS_SOUNDSCAPY_PLOTS = False


def plot_correlations(df: pd.DataFrame) -> Path:
    out_path = PLOT_DIR / "correlaciones.png"
    corr = correlation_matrix(df)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlaciones")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_clusters(df: pd.DataFrame) -> Path:
    out_path = PLOT_DIR / "clusters_psico.png"
    labels = assign_clusters(df)
    if labels.size == 0:
        return out_path
    plt.figure(figsize=(8, 6))
    plt.scatter(df["P_iso"], df["E_iso"], c=labels, cmap="tab10", alpha=0.8)
    plt.xlabel("P_iso (proxy)")
    plt.ylabel("E_iso (proxy)")
    plt.title("Clusters en el espacio P/E (proxy)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_perceptual(df: pd.DataFrame, df_rec: pd.DataFrame) -> list[Path]:
    out_paths: list[Path] = []
    out_seg = PLOT_DIR / "perceptual_segmentos.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(df["P_iso"], df["E_iso"], c=df["Probabilidad"], cmap="viridis", alpha=0.6)
    plt.xlabel("P_iso (proxy)")
    plt.ylabel("E_iso (proxy)")
    plt.title("Mapa perceptual por segmento")
    plt.colorbar(label="Probabilidad YAMNet")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_seg, dpi=150)
    plt.close()
    out_paths.append(out_seg)

    out_rec = PLOT_DIR / "perceptual_recordings.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(df_rec["P_iso"], df_rec["E_iso"], s=120, alpha=0.9)
    for _, row in df_rec.iterrows():
        plt.text(row["P_iso"], row["E_iso"], row["Recording"], fontsize=8, ha="left", va="bottom")
    plt.xlabel("P_iso (proxy)")
    plt.ylabel("E_iso (proxy)")
    plt.title("Mapa perceptual por Recording")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_rec, dpi=150)
    plt.close()
    out_paths.append(out_rec)
    return out_paths


def plot_soundscape(df_rec: pd.DataFrame) -> Path | None:
    if df_rec.empty or not HAS_SOUNDSCAPY_PLOTS:
        return None
    df_s = df_rec.rename(columns={"P_iso": "ISOPleasant", "E_iso": "ISOEventful"}).copy()
    df_s["LocationID"] = df_s["Recording"]
    out_path = PLOT_DIR / "soundscapy_lugares.png"
    ax = scatter_plot(
        df_s,
        x="ISOPleasant",
        y="ISOEventful",
        hue="LocationID",
        title="Comparacion perceptual de grabaciones (proxy)",
        backend=Backend.SEABORN,
    )
    fig = ax.get_figure() if hasattr(ax, "get_figure") else ax
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_location_comparisons(df_rec: pd.DataFrame) -> Path | None:
    if df_rec.shape[0] < 2 or not HAS_SOUNDSCAPY_PLOTS:
        return None
    df_s = df_rec.rename(columns={"P_iso": "ISOPleasant", "E_iso": "ISOEventful"}).copy()
    df_s["LocationID"] = df_s["Recording"]
    out_path = PLOT_DIR / "soundscapy_lugares_multiple.png"
    ax = density_plot(
        df_s,
        x="ISOPleasant",
        y="ISOEventful",
        hue="LocationID",
        title="Comparacion perceptual por audio (Soundscapy)",
        backend=Backend.SEABORN,
    )
    fig = ax.get_figure() if hasattr(ax, "get_figure") else ax
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_soundscape_fallback(df_rec: pd.DataFrame) -> Path | None:
    if df_rec.empty:
        return None
    out_path = PLOT_DIR / "soundscapy_lugares_fallback.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(df_rec["P_iso"], df_rec["E_iso"], s=120, alpha=0.9)
    for _, row in df_rec.iterrows():
        plt.text(row["P_iso"], row["E_iso"], row["Recording"], fontsize=8, ha="left", va="bottom")
    plt.xlabel("P (proxy)")
    plt.ylabel("E (proxy)")
    plt.title("Comparacion perceptual (fallback)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_location_comparisons_fallback(df_rec: pd.DataFrame) -> Path | None:
    if df_rec.shape[0] < 2:
        return None
    out_path = PLOT_DIR / "soundscapy_lugares_multiple_fallback.png"
    plt.figure(figsize=(10, 6))
    plt.scatter(df_rec["P_iso"], df_rec["E_iso"], s=120, alpha=0.9)
    for _, row in df_rec.iterrows():
        plt.text(row["P_iso"], row["E_iso"], row["Recording"], fontsize=8, ha="left", va="bottom")
    plt.xlabel("P (proxy)")
    plt.ylabel("E (proxy)")
    plt.title("Comparacion perceptual por audio (fallback)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
