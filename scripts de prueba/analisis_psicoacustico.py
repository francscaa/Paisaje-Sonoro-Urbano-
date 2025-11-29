#!/usr/bin/env python3
"""Ejemplo de uso de soundscapy para obtener descriptores psicoacusticos.

El script recorre uno o varios audios WAV y calcula, por canal:
- Loudness Zwicker dependiente del tiempo (`loudness_zwtv`)
- Roughness Daniel & Weber (`roughness_dw`)
- Sharpness DIN dependiente del tiempo (`sharpness_din_tv`)

Se usan las funciones de `soundscapy.analysis.metrics.mosqito_metric_1ch`,
que ya implementan las estadisticas basicas (percentiles, promedio, max, etc.).
Los resultados se devuelven en consola o se guardan en un JSON plano.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import warnings

# Asegura un directorio de cache para Matplotlib antes de importar librerias que la usen.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
from acoustics import Signal
from soundscapy.analysis.metrics import mosqito_metric_1ch

# Metric names soportados por mosqito_metric_1ch.
ALLOWED_METRICS = {
    "loudness_zwtv",
    "sharpness_din_tv",
    "sharpness_din_from_loudness",
    "sharpness_din_perseg",
    "roughness_dw",
}
DEFAULT_METRICS = ["loudness_zwtv", "roughness_dw", "sharpness_din_tv"]
DEFAULT_FUNC_ARGS: Dict[str, Dict[str, object]] = {
    # Evita el warning por el transitorio inicial de la sonoridad.
    "sharpness_din_tv": {"skip": 10},
}

# Oculta warnings de precision de scipy cuando la senal es muy estable.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="soundscapy.analysis.metrics")


def ensure_mplconfigdir(base: Path) -> None:
    """Evita problemas de permisos con el cache de Matplotlib."""
    mpl_dir = os.environ.setdefault("MPLCONFIGDIR", str(base / ".matplotlib"))
    Path(mpl_dir).mkdir(parents=True, exist_ok=True)


def discover_audio_files(inputs: Iterable[str]) -> List[Path]:
    """Devuelve la lista de archivos .wav a procesar."""
    files: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.wav")))
        elif path.is_file() and path.suffix.lower() == ".wav":
            files.append(path)
        else:
            print(f"[aviso] Se ignora la ruta no valida: {path}")
    return files


def split_channels(signal: Signal) -> Iterable[Tuple[str, Signal]]:
    """Separa un Signal en canales individuales conservando fs."""
    data = signal.view(np.ndarray)
    channels = getattr(signal, "channels", 1)
    if data.ndim == 1 or channels == 1:
        yield "ch1", Signal(data, signal.fs)
        return

    # Signal usa el eje 0 para canales (shape = (n_channels, n_samples))
    channels_first = data.shape[0] == channels
    for idx in range(channels):
        ch_data = data[idx] if channels_first else data[:, idx]
        yield f"ch{idx + 1}", Signal(ch_data, signal.fs)


def compute_psychoacoustics(signal: Signal, metrics: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """Calcula los descriptores psicoacusticos solicitados para un canal."""
    results: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        func_args = DEFAULT_FUNC_ARGS.get(metric, {})
        stats = mosqito_metric_1ch(
            signal,
            metric,
            as_df=False,
            return_time_series=False,
            func_args=func_args,
        )
        # Convierte numpy types a float para que el JSON sea serializable.
        results[metric] = {k: float(v) for k, v in stats.items()}
    return results


def flatten_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Flattea {'metric': {'stat': value}} -> {'metric_stat': value}."""
    flat: Dict[str, float] = {}
    for metric_name, stats in metrics.items():
        for stat_name, value in stats.items():
            flat[f"{metric_name}_{stat_name}"] = value
    return flat


def process_file(path: Path, metrics: Iterable[str]) -> List[Dict[str, object]]:
    """Procesa un audio y devuelve una fila por canal con las metricas."""
    signal = Signal.from_wav(path)
    rows: List[Dict[str, object]] = []
    for channel_name, channel_signal in split_channels(signal):
        metrics_dict = compute_psychoacoustics(channel_signal, metrics)
        row = {"file": str(path), "channel": channel_name}
        row.update(flatten_metrics(metrics_dict))
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula descriptores psicoacusticos con soundscapy/mosqito.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Archivos .wav o carpetas que contengan audios .wav",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        choices=sorted(ALLOWED_METRICS),
        help="Metricas psicoacusticas a calcular",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Ruta de salida (JSON). Si no se indica, se imprime por consola.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_mplconfigdir(Path.cwd())

    files = discover_audio_files(args.inputs)
    if not files:
        raise SystemExit("No se encontraron archivos .wav para procesar.")

    all_rows: List[Dict[str, object]] = []
    for wav_path in files:
        all_rows.extend(process_file(wav_path, args.metrics))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))
        print(f"Resultados guardados en: {args.output}")
    else:
        print(json.dumps(all_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
