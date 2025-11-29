from __future__ import annotations

from pathlib import Path

import matplotlib

# Usar backend no interactivo para permitir generación de gráficos en batch.
matplotlib.use("Agg")

RESULTS_DIR: Path = Path("results")
PLOT_DIR: Path = RESULTS_DIR / "plots"
TFHUB_CACHE_DIR: Path = Path("models/tfhub")
DEFAULT_MODEL_HANDLE: str = "https://tfhub.dev/google/yamnet/1"

# Archivos de salida principales.
SEGMENT_CSV: Path = RESULTS_DIR / "yamnet_psico_segmentado.csv"
RECORDINGS_CSV: Path = RESULTS_DIR / "yamnet_psico_recordings.csv"

# Descriptores psicoacústicos utilizados a lo largo del pipeline.
DESCRIPTORS: list[str] = [
    "loudness_sones",
    "sharpness_acum",
    "roughness_asper",
    "tonality_tnr_db",
]

# Aseguramos que las carpetas de resultados existan al importar la configuración.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
