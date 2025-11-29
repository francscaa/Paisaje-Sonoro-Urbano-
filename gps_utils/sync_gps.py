from __future__ import annotations

import numpy as np
import pandas as pd


def sync_audio_gps(audio_df: pd.DataFrame, gps_df: pd.DataFrame) -> pd.DataFrame:
    """Sincroniza segmentos de audio con GPS usando vecino más cercano e interpolación lineal.

    - Si hay dos puntos GPS antes y después del Timestamp del segmento, interpola lat/lon/alt.
    - Si solo hay vecino más cercano, lo asigna.
    - Agrega columnas: gps_error_s (|t_audio - t_gps_asignado|) y gps_source ("interpolated"/"nearest"/"none").
    - Si no hay timestamps GPS válidos, deja lat/lon/alt vacíos y gps_source="none".
    """
    df = audio_df.copy()
    df[["lat", "lon", "alt"]] = pd.NA
    df["gps_error_s"] = pd.NA
    df["gps_source"] = "none"

    if gps_df.empty or "t_seconds" not in gps_df.columns:
        return df

    gps_sorted = gps_df.dropna(subset=["t_seconds"]).sort_values("t_seconds")
    if gps_sorted.empty:
        return df

    gps_times = gps_sorted["t_seconds"].to_numpy(dtype=float)
    gps_coords = gps_sorted[["lat", "lon", "alt"]].to_numpy(dtype=float)

    for idx, row in df.iterrows():
        t = row.get("Timestamp")
        if pd.isna(t):
            continue
        pos = np.searchsorted(gps_times, t)

        # Caso: interpolación entre dos puntos
        if 0 < pos < len(gps_times):
            t0, t1 = gps_times[pos - 1], gps_times[pos]
            if t1 != t0:
                w = (t - t0) / (t1 - t0)
                interp = gps_coords[pos - 1] + w * (gps_coords[pos] - gps_coords[pos - 1])
                df.at[idx, "lat"] = interp[0]
                df.at[idx, "lon"] = interp[1]
                df.at[idx, "alt"] = interp[2]
                df.at[idx, "gps_error_s"] = 0.0
                df.at[idx, "gps_source"] = "interpolated"
                continue

        # Caso: vecino más cercano (bordes o tiempos idénticos)
        candidates = []
        if pos > 0:
            candidates.append((abs(t - gps_times[pos - 1]), pos - 1))
        if pos < len(gps_times):
            candidates.append((abs(t - gps_times[pos]), pos))
        if not candidates:
            continue
        err, best_idx = min(candidates, key=lambda x: x[0])
        df.at[idx, "lat"] = gps_coords[best_idx][0]
        df.at[idx, "lon"] = gps_coords[best_idx][1]
        df.at[idx, "alt"] = gps_coords[best_idx][2]
        df.at[idx, "gps_error_s"] = float(err)
        df.at[idx, "gps_source"] = "nearest"
    return df
