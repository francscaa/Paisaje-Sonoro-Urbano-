from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import librosa
import numpy as np


def select_files(paths: list[Path] | None) -> list[Path]:
    """Valida o selecciona archivos de audio desde un diálogo."""
    if paths:
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"No existen: {missing}")
        return paths
    try:
        from tkinter import Tk, filedialog
    except Exception as exc:  # pragma: no cover - fallback interactivo
        raise SystemExit(
            "Tkinter no esta disponible en este Python. Pasa los archivos con --files "
            "para omitir el selector grafico."
        ) from exc
    Tk().withdraw()
    picked = filedialog.askopenfilenames(
        title="Selecciona uno o mas archivos de audio",
        filetypes=[("Audio", "*.wav *.mp3"), ("All files", "*.*")],
    )
    if not picked:
        raise SystemExit("No se seleccionaron archivos.")
    return [Path(p) for p in picked]


def load_audio_file(path: Path) -> Tuple[np.ndarray, int, float]:
    """Carga un archivo de audio en mono preservando el sr original."""
    y, sr = librosa.load(path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    return y, sr, duration
