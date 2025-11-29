from __future__ import annotations

from typing import Generator, Tuple

import librosa
import numpy as np


def segment_audio(
    y: np.ndarray, sr: int, window: float, hop: float
) -> Generator[Tuple[float, float, float, np.ndarray], None, None]:
    """Genera segmentos de audio con marca temporal relativa y absoluta.

    - Timestamp: segundo relativo desde el inicio del audio.
    - AbsTime: igual a Timestamp por ahora; listo para reemplazarse por tiempo GPS absoluto.
      Futuro: se sincronizará con telemetría para mapear cada segmento en el eje espacio–temporal.
    """
    duration = librosa.get_duration(y=y, sr=sr)
    for start_time in np.arange(0, duration, hop):
        end_time = min(start_time + window, duration)
        start_sample = librosa.time_to_samples(start_time, sr=sr)
        end_sample = librosa.time_to_samples(end_time, sr=sr)
        segment = y[start_sample:end_sample]
        if segment.size == 0:
            continue
        abs_time = float(start_time)  # Placeholder; se sincronizará con GPS en el futuro.
        yield start_time, abs_time, end_time, segment
