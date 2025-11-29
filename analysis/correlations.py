from __future__ import annotations

import pandas as pd

from config import DESCRIPTORS


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula matriz de correlaciones sobre descriptores y proxies."""
    cols = [c for c in DESCRIPTORS + ["P_iso", "E_iso", "Probabilidad"] if c in df.columns]
    return df[cols].corr()
