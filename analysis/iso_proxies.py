from __future__ import annotations

import pandas as pd

from config import DESCRIPTORS


def compute_iso_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas P_iso y E_iso a partir de descriptores normalizados (z-score)."""
    zdf = df.copy()
    for d in DESCRIPTORS:
        if d in zdf.columns:
            zdf[d + "_z"] = (zdf[d] - zdf[d].mean()) / (zdf[d].std() or 1)
        else:
            zdf[d + "_z"] = 0.0
    zdf["P_iso"] = (
        -zdf["loudness_sones_z"]
        - zdf["roughness_asper_z"]
        - zdf["sharpness_acum_z"]
        - zdf["tonality_tnr_db_z"]
    )
    zdf["E_iso"] = (
        zdf["loudness_sones_z"]
        + zdf["roughness_asper_z"]
        + zdf["sharpness_acum_z"]
        + zdf["tonality_tnr_db_z"]
    )
    return zdf


def aggregate_by_recording(df: pd.DataFrame) -> pd.DataFrame:
    """Promedia métricas por Recording."""
    return df.groupby("Recording").mean(numeric_only=True).reset_index()
