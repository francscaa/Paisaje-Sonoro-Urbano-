from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DESCRIPTORS, PLOT_DIR


def plot_descriptor_bars(df: pd.DataFrame) -> Path:
    out_path = PLOT_DIR / "fuentes_por_descriptor.png"
    top = (
        df.groupby("Clase_YAMNet")[DESCRIPTORS]
        .mean()
        .sort_values(by="loudness_sones", ascending=False)
        .head(10)
    )
    top.plot(kind="bar", figsize=(12, 6))
    plt.ylabel("Valor promedio")
    plt.title("Fuentes (YAMNet) ordenadas por loudness (top 10) con otros descriptores")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_compare_recordings(df_rec: pd.DataFrame) -> Path | None:
    if df_rec.shape[0] < 2:
        return None
    out_path = PLOT_DIR / "comparacion_recordings_psico.png"
    x = np.arange(len(df_rec["Recording"]))
    width = 0.2
    plt.figure(figsize=(10, 6))
    for i, d in enumerate(DESCRIPTORS):
        if d in df_rec.columns:
            plt.bar(x + i * width, df_rec[d], width=width, label=d)
    plt.xticks(x + width * (len(DESCRIPTORS) - 1) / 2, df_rec["Recording"], rotation=45, ha="right")
    plt.ylabel("Valor promedio")
    plt.title("Comparacion de descriptores psicoacusticos por audio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
