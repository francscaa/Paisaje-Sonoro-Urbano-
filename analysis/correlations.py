from __future__ import annotations

import pandas as pd

from config import DESCRIPTORS

# Descriptor tonality_tnr_db se conserva en datos, pero se excluye de gráficos/lecturas
# por falta de variabilidad en este caso de estudio.
PLOT_DESCRIPTORS = [d for d in DESCRIPTORS if d != "tonality_tnr_db"]


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula matriz de correlaciones sobre descriptores y proxies."""
    cols = [c for c in PLOT_DESCRIPTORS + ["P_iso", "E_iso", "Probabilidad"] if c in df.columns]
    return df[cols].corr()
