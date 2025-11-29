from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from audio_processing.load_audio import load_audio_file
from audio_processing.psycho_features import compute_psycho
from audio_processing.segmentation import segment_audio
from audio_processing.yamnet_classifier import predict_top_class
from config import SEGMENT_CSV


def process_segment(
    segment: np.ndarray, sr: int, timestamp: float, abs_time: float, model, class_names: List[str]
) -> dict:
    yamnet = predict_top_class(segment, sr, model, class_names, timestamp=timestamp)
    psycho = compute_psycho(segment, sr)
    return {
        "Timestamp": timestamp,
        "AbsTime": abs_time,
        "Recording": "",
        **yamnet,
        **psycho,
    }


def process_audios(audio_paths: list[Path], window: float, hop: float, model, class_names: list[str]) -> pd.DataFrame:
    results = []
    for audio_path in audio_paths:
        print(f"Cargando audio: {audio_path}")
        y, sr, duration = load_audio_file(audio_path)
        print(f"Duracion: {duration:.2f}s")
        for start_time, abs_time, end_time, segment in segment_audio(y, sr, window, hop):
            row = process_segment(segment, sr, start_time, abs_time, model, class_names)
            row["Recording"] = audio_path.stem
            results.append(row)
            print(f"Procesando segmento: {start_time:.1f}-{end_time:.1f}s ({audio_path.stem})")
    if not results:
        raise SystemExit("No se pudieron procesar segmentos.")
    df = pd.DataFrame(results)
    out_csv = SEGMENT_CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nCSV segmentado guardado en: {out_csv}")
    return df
