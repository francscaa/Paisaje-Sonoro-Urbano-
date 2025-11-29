from __future__ import annotations

from pathlib import Path

import pandas as pd


def sync_audio_gps(audio_df: pd.DataFrame, gps_df: pd.DataFrame) -> pd.DataFrame:
    """Sincroniza segmentos de audio con el punto GPS temporalmente más cercano."""
    if gps_df.empty or "t_seconds" not in gps_df.columns:
        return audio_df

    df = audio_df.copy()
    df[["lat", "lon", "alt"]] = pd.NA

    gps_sorted = gps_df.dropna(subset=["t_seconds"]).sort_values("t_seconds")
    if gps_sorted.empty:
        return df

    gps_times = gps_sorted["t_seconds"].to_numpy()
    gps_coords = gps_sorted[["lat", "lon", "alt"]].to_numpy()

    for idx, row in df.iterrows():
        t = row.get("Timestamp")
        if pd.isna(t):
            continue
        pos = gps_times.searchsorted(t)
        candidates = []
        if pos > 0:
            candidates.append((abs(t - gps_times[pos - 1]), pos - 1))
        if pos < len(gps_times):
            candidates.append((abs(t - gps_times[pos]), pos))
        if not candidates:
            continue
        _, best_idx = min(candidates, key=lambda x: x[0])
        df.at[idx, "lat"] = gps_coords[best_idx][0]
        df.at[idx, "lon"] = gps_coords[best_idx][1]
        df.at[idx, "alt"] = gps_coords[best_idx][2]
    return df
