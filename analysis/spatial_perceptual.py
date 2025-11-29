from __future__ import annotations

import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from config import DESCRIPTORS


def compute_perceptual_space(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve df ordenado por AbsTime (o Timestamp) con P/E ya presentes."""
    ordered = df
    if "AbsTime" in df.columns:
        ordered = df.sort_values("AbsTime")
    elif "Timestamp" in df.columns:
        ordered = df.sort_values("Timestamp")
    return ordered


def join_space_perceptual(df: pd.DataFrame) -> pd.DataFrame:
    """Combina coordenadas GPS con P_iso/E_iso para análisis conjunto."""
    out = df.copy()
    for col in ["lat", "lon", "P_iso", "E_iso"]:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def compute_spatial_clusters(
    df: pd.DataFrame, method: str = "kmeans", n_clusters: int = 4, eps: float = 0.001, min_samples: int = 3
) -> pd.DataFrame:
    """Clustering en espacio geo + psico (lat, lon + descriptores).

    method: "kmeans" o "dbscan"
    """
    out = df.copy()
    feats = []
    for col in ["lat", "lon"] + DESCRIPTORS:
        if col in out.columns:
            feats.append(col)
        else:
            out[col] = pd.NA
            feats.append(col)
    data = out[feats].astype(float)
    valid_mask = data[["lat", "lon"]].notna().all(axis=1)
    if valid_mask.sum() == 0:
        out["cluster_id"] = pd.NA
        return out
    X = data[valid_mask].fillna(data.mean(numeric_only=True))
    X_scaled = StandardScaler().fit_transform(X)

    if method == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
    else:
        k = min(n_clusters, len(X_scaled)) if len(X_scaled) > 0 else 0
        if k < 1:
            out["cluster_id"] = pd.NA
            return out
        model = KMeans(n_clusters=k, n_init="auto", random_state=0)
        labels = model.fit_predict(X_scaled)

    out["cluster_id"] = pd.NA
    out.loc[valid_mask, "cluster_id"] = labels
    return out
