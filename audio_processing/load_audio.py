from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import librosa
import numpy as np


def _cli_select_files() -> list[Path]:
    """Selector simple por consola cuando no hay GUI disponible."""
    print("Selector por consola. Ingresa rutas completas separadas por coma o elige por numero.")
    candidates = sorted(Path("recordings").glob("*.wav")) + sorted(Path("recordings").glob("*.mp3"))
    if candidates:
        print("Archivos encontrados en recordings/:")
        for i, cand in enumerate(candidates, 1):
            print(f"  {i}) {cand}")
    raw = input("Rutas o numeros separados por coma: ").strip()
    if not raw:
        raise SystemExit("No se seleccionaron archivos.")
    choices = [c.strip() for c in raw.split(",") if c.strip()]
    picked: list[Path] = []
    for choice in choices:
        path = None
        if choice.isdigit() and candidates:
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                path = candidates[idx]
        if path is None:
            path = Path(choice).expanduser()
        if not path.exists():
            raise SystemExit(f"No existe el archivo: {path}")
        picked.append(path)
    return picked


def _tk_select_files() -> list[Path]:
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    try:
        picked = filedialog.askopenfilenames(
            title="Selecciona uno o mas archivos de audio",
            filetypes=[("Audio", "*.wav *.mp3"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    if not picked:
        raise SystemExit("No se seleccionaron archivos.")
    return [Path(p) for p in picked]


def select_files(paths: list[Path] | None) -> list[Path]:
    """Valida o selecciona archivos de audio desde GUI (tk) o consola."""
    if paths:
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"No existen: {missing}")
        return paths

    use_gui = not os.environ.get("CLI_FILE_PICKER") and not os.environ.get("NO_GUI")
    if use_gui:
        try:
            return _tk_select_files()
        except Exception:
            print("[aviso] No se pudo abrir el selector grafico, cambiando a modo consola.")
    return _cli_select_files()


def load_audio_file(path: Path) -> Tuple[np.ndarray, int, float]:
    """Carga un archivo de audio en mono preservando el sr original."""
    y, sr = librosa.load(path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    return y, sr, duration
