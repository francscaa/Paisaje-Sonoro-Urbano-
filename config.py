from __future__ import annotations

from pathlib import Path
import re

import matplotlib

# Usar backend no interactivo para permitir generación de gráficos en batch.
matplotlib.use("Agg")

# Carpeta raíz por corrida (nombre del WAV/CSV). Todo lo generado vive dentro de <run_dir>/results/.
RUN_DIR: Path = Path("run")
RESULTS_DIR: Path = RUN_DIR / "results"
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


def set_run_results_dir(run_name: str) -> None:
    """Actualiza rutas globales para agrupar salidas en <run_name>/results/."""
    global RUN_DIR, RESULTS_DIR, PLOT_DIR, SEGMENT_CSV, RECORDINGS_CSV
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", run_name).strip("_") or "run"
    RUN_DIR = Path(safe_name)
    RESULTS_DIR = RUN_DIR / "results"
    PLOT_DIR = RESULTS_DIR / "plots"
    SEGMENT_CSV = RESULTS_DIR / "yamnet_psico_segmentado.csv"
    RECORDINGS_CSV = RESULTS_DIR / "yamnet_psico_recordings.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


# Aseguramos que las carpetas de resultados existan al importar la configuración.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
