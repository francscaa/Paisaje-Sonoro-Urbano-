from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from config import DESCRIPTORS


def plot_descriptor_bars(df: pd.DataFrame) -> Path:
    out_path = config.PLOT_DIR / "fuentes_por_descriptor.png"
    plot_desc = [d for d in DESCRIPTORS if d != "tonality_tnr_db"]
    top = (
        df.groupby("Clase_YAMNet")[plot_desc]
        .mean()
        .sort_values(by="loudness_sones", ascending=False)
        .head(10)
    )
    top.plot(kind="bar", figsize=(12, 6))
    plt.ylabel("Valor promedio")
    plt.title("Top 10 clases YAMNet por loudness con descriptores psicoacústicos")
    plt.suptitle(f"Unidad: segmentos | N clases={len(top)} | Barras: promedio por descriptor", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_compare_recordings(df_rec: pd.DataFrame) -> Path | None:
    if df_rec.shape[0] < 2:
        return None
    out_path = config.PLOT_DIR / "comparacion_recordings_psico.png"
    x = np.arange(len(df_rec["Recording"]))
    width = 0.2
    unidad = "Recording (recorridos completos)"
    plot_desc = [d for d in DESCRIPTORS if d != "tonality_tnr_db"]
    plt.figure(figsize=(10, 6))
    for i, d in enumerate(plot_desc):
        if d in df_rec.columns:
            plt.bar(x + i * width, df_rec[d], width=width, label=d)
    plt.xticks(x + width * (len(plot_desc) - 1) / 2, df_rec["Recording"], rotation=45, ha="right")
    plt.ylabel("Valor promedio")
    plt.title("Comparación de descriptores psicoacústicos por recorrido")
    plt.suptitle(f"Unidad: {unidad} | N recorridos={df_rec.shape[0]} | Color: descriptor", fontsize=9)
    plt.legend(title="Descriptor", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
