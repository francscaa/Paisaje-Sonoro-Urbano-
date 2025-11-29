from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import DESCRIPTORS


def assign_clusters(df: pd.DataFrame) -> np.ndarray:
    """Calcula etiquetas de clústeres k-means en el espacio de descriptores."""
    data = df[DESCRIPTORS].fillna(df[DESCRIPTORS].median())
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    k = 3 if len(df) >= 3 else len(df)
    if k < 1:
        return np.array([])
    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    return km.fit_predict(X)
