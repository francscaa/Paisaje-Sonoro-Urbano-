from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def export_gis(df_rec: pd.DataFrame, output_path: Path) -> Path:
    """Punto de extensión para exportar resultados a GIS (GeoJSON/SHAPE)."""
    # Placeholder: se completará cuando se integre GPS.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_rec.to_csv(output_path, index=False)
    return output_path
