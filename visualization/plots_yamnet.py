from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DESCRIPTORS, PLOT_DIR, RESULTS_DIR


def plot_sources(df: pd.DataFrame) -> Path:
    out_path = PLOT_DIR / "yamnet_fuentes_scatter.png"
    plt.figure(figsize=(12, 6))
    factor, labels = pd.factorize(df["Clase_YAMNet"])
    scatter = plt.scatter(df["Timestamp"], df["Probabilidad"], c=factor, cmap="tab20", alpha=0.8)
    handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
    plt.legend(handles, labels, loc="best", title="Clase YAMNet", fontsize="small", ncol=2)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Probabilidad")
    plt.title("Clases YAMNet en el tiempo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_fuentes_loudness(df_seg: pd.DataFrame) -> Path:
    out_path = PLOT_DIR / "yamnet_fuentes_loudness.png"
    cols = [d for d in DESCRIPTORS if d in df_seg.columns]
    df_fuentes = (
        df_seg.groupby("Clase_YAMNet")[cols]
        .mean()
        .sort_values(by="loudness_sones", ascending=False)
    )
    (RESULTS_DIR / "yamnet_fuentes_psico.csv").parent.mkdir(parents=True, exist_ok=True)
    df_fuentes.to_csv(RESULTS_DIR / "yamnet_fuentes_psico.csv")
    df_fuentes.head(10).plot(kind="bar", figsize=(14, 6))
    plt.title("Fuentes YAMNet ordenadas por loudness (top 10)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
