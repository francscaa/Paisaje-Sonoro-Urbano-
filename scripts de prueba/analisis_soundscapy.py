"""Pipeline psicoacustico usando Soundscapy 0.7.x (modulo audio + MoSQITo).

- Selecciona metricas de MoSQITo (loudness, sharpness, roughness).
- Desactiva scikit-maad/acoustic si quieres acelerar.
- Exporta un CSV plano listo para Excel/PowerBI y genera un CSV de comparaciones.
"""

import argparse
import itertools
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import mosqito as mq
from scipy.io import wavfile
from tkinter import Tk, filedialog


def ensure_mplconfigdir(base: Path) -> None:
    """Evita problemas de permisos con el cache de Matplotlib."""
    os.environ.setdefault("MPLCONFIGDIR", str(base / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def patch_numpy_printoptions() -> None:
    """Soundscapy fija legacy='1.25', que es invalido en NumPy>=1.26. Lo normalizamos."""
    real_set_printoptions = np.set_printoptions

    def safe_set_printoptions(*args, **kwargs):
        if kwargs.get("legacy") == "1.25":
            kwargs["legacy"] = False
        return real_set_printoptions(*args, **kwargs)

    np.set_printoptions = safe_set_printoptions


# Preparar entorno antes de importar soundscapy (usa matplotlib y np.set_printoptions)
ensure_mplconfigdir(Path(__file__).parent)
patch_numpy_printoptions()

try:
    from soundscapy import AudioAnalysis
except ImportError as e:
    raise SystemExit(
        "No se pudo importar soundscapy audio. Instala dependencias opcionales con:\n"
        "  pip install \"soundscapy[audio]\""
    ) from e

import matplotlib.pyplot as plt


DEFAULT_AUDIO_FOLDER = "audios"
DEFAULT_RESULTS_FOLDER = "results"
DEFAULT_PLOTS_FOLDER = "results/plots"
MO_SQI_TO_METRICS = {
    "loudness": ["loudness_zwtv"],
    "sharpness": ["sharpness_din_from_loudness", "sharpness_din_perseg"],
    "roughness": ["roughness_dw"],
    "tonality": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline psicoacustico usando Soundscapy + MoSQITo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Ruta a un .wav especifico (si no se indica, se abre un selector).",
    )
    parser.add_argument(
        "--audio-folder",
        default=DEFAULT_AUDIO_FOLDER,
        help="Carpeta con los .wav a procesar (solo si no usas --file).",
    )
    parser.add_argument(
        "--results-folder",
        default=DEFAULT_RESULTS_FOLDER,
        help="Carpeta donde se guardan CSV y salidas.",
    )
    parser.add_argument(
        "--plots-folder",
        default=DEFAULT_PLOTS_FOLDER,
        help="Carpeta donde se guardan graficos.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["loudness", "sharpness", "roughness", "tonality"],
        choices=list(MO_SQI_TO_METRICS.keys()),
        help="Metricas MoSQITo a calcular.",
    )
    parser.add_argument(
        "--disable-maad",
        action="store_true",
        help="Desactiva los indices scikit-maad (alpha diversity) para acelerar.",
    )
    parser.add_argument(
        "--disable-acoustic",
        action="store_true",
        help="Desactiva metricas AcousticToolbox (LAeq, LZeq, LCeq, SEL).",
    )
    parser.add_argument(
        "--compare-label",
        default="N_avg",
        help="Columna (ej. N_avg, S_perseg_avg, R_avg) para comparar audios.",
    )
    parser.add_argument(
        "--resample",
        type=int,
        default=None,
        help="Frecuencia de remuestreo (Hz). Dejalo vacio para usar la nativa.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Workers en paralelo. Por defecto usa los cores disponibles.",
    )
    return parser.parse_args()


def ensure_dirs(results_folder: Path, plots_folder: Path) -> None:
    results_folder.mkdir(parents=True, exist_ok=True)
    plots_folder.mkdir(parents=True, exist_ok=True)


def select_metrics(settings, metrics: Iterable[str], disable_maad: bool, disable_acoustic: bool) -> None:
    """Activa/desactiva metricas dentro de AnalysisSettings."""
    wanted = set(metrics)
    if settings.MoSQITo:
        for metric_name, cfg in settings.MoSQITo.root.items():
            cfg.run = any(metric_name in MO_SQI_TO_METRICS[m] for m in wanted)
    if settings.scikit_maad:
        for cfg in settings.scikit_maad.root.values():
            cfg.run = not disable_maad
    if settings.AcousticToolbox:
        for cfg in settings.AcousticToolbox.root.values():
            cfg.run = not disable_acoustic


def flatten_results(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte el MultiIndex (Recording, Channel) a columnas planas."""
    return df.reset_index()


def plot_bar(metric_df: pd.DataFrame, label: str, plots_folder: Path) -> None:
    """Grafico de barras por archivo para una columna (promedio entre canales)."""
    if label not in metric_df.columns:
        print(f"[aviso] No se encontro la columna {label} para graficar.")
        return
    grouped = metric_df.groupby("Recording")[label].mean()
    plt.figure(figsize=(8, 4))
    grouped.plot(kind="bar", color="#1f77b4")
    plt.ylabel(label)
    plt.tight_layout()
    out_path = plots_folder / f"{label}_bar.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  ✔ Grafico {label} -> {out_path}")


def compare_pairs(metric_df: pd.DataFrame, label: str, results_folder: Path) -> None:
    """Genera CSV con diferencias de la columna seleccionada entre pares de audios."""
    if label not in metric_df.columns:
        print(f"[aviso] No se encontro la columna {label} para comparar.")
        return
    grouped = metric_df.groupby("Recording")[label].mean()
    rows: List[dict] = []
    for a, b in itertools.combinations(grouped.index, 2):
        rows.append(
            {
                "audio_1": a,
                "audio_2": b,
                f"diferencia_{label}": grouped[a] - grouped[b],
            }
        )
    if not rows:
        return
    comp_df = pd.DataFrame(rows)
    comp_path = results_folder / f"comparaciones_{label}.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"  ✔ Comparaciones guardadas en {comp_path}")


def ensure_stereo_file(path: Path, tmp_dir: Path) -> Path:
    """Convierte a stereo y normaliza a float32 [-1, 1] para evitar dB extremos."""
    try:
        sr, data = wavfile.read(path)
    except Exception as exc:
        raise SystemExit(f"No se pudo leer el wav: {exc}")

    # Normalizar a float32 [-1, 1] segun tipo
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)
    maxabs = np.max(np.abs(data))
    if maxabs > 1:
        data = data / maxabs
    # Deja algo de margen para evitar picos: baja 6 dB
    data *= 0.5

    channels = 1 if data.ndim == 1 else data.shape[1]
    if channels == 1:
        data = np.column_stack([data, data])
        print(f"[aviso] Archivo mono duplicado a stereo para procesar: {path}")
    elif channels != 2:
        raise SystemExit(f"Se requieren 1 o 2 canales; encontrado {channels}.")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{path.stem}_stereo_norm.wav"
    wavfile.write(tmp_path, sr, data)
    print(f"[aviso] Usando copia normalizada: {tmp_path}")
    return tmp_path


def compute_tonality_metrics(paths: List[Path]) -> pd.DataFrame:
    """Calcula tonality (TNR y PR promedio) por canal y devuelve MultiIndex."""
    rows: List[dict] = []
    for wav_path in paths:
        try:
            sr, data = wavfile.read(wav_path)
        except Exception as exc:
            print(f"[aviso] No se pudo leer {wav_path} para tonality: {exc}")
            continue

        # Normalizar a float32 [-1, 1]
        if np.issubdtype(data.dtype, np.integer):
            max_val = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / max_val
        else:
            data = data.astype(np.float32)
        maxabs = np.max(np.abs(data))
        if maxabs > 1:
            data = data / maxabs

        if data.ndim == 1:
            data = np.column_stack([data, data])

        recording = wav_path.stem
        for idx, ch_name in enumerate(["Left", "Right"]):
            sig = data[:, idx]
            tnr_total, *_ = mq.tnr_ecma_st(sig, sr)
            pr_total, *_ = mq.pr_ecma_st(sig, sr)
            rows.append(
                {
                    "Recording": recording,
                    "Channel": ch_name,
                    "tonality_tnr_db": float(np.mean(np.atleast_1d(tnr_total))),
                    "tonality_pr_db": float(np.mean(np.atleast_1d(pr_total))),
                }
            )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index(["Recording", "Channel"])
    return df


def main() -> None:
    args = parse_args()
    audio_folder = Path(args.audio_folder)
    results_folder = Path(args.results_folder)
    plots_folder = Path(args.plots_folder)

    ensure_dirs(results_folder, plots_folder)

    # Seleccionar archivo: CLI -> dialogo -> carpeta
    selected_file: Path | None = None
    if args.file:
        selected_file = args.file
    else:
        # Abrir dialogo de seleccion
        Tk().withdraw()
        picked = filedialog.askopenfilename(
            title="Selecciona un archivo WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if picked:
            selected_file = Path(picked)

    if selected_file:
        if not selected_file.exists() or selected_file.suffix.lower() != ".wav":
            raise SystemExit("La ruta indicada no es un .wav valido.")
        wav_files = [selected_file]
        print("\n📥 Procesando archivo unico:", selected_file)
    else:
        print("\n📥 Cargando audios desde:", audio_folder)
        wav_files = sorted(audio_folder.glob("*.wav"))
        print("  ✔ Audios cargados:", len(wav_files))
        if not wav_files:
            raise SystemExit("No se encontraron .wav en la carpeta indicada y no se selecciono ninguno.")

    print("\n🧠 Configurando metricas...")
    analysis = AudioAnalysis()
    select_metrics(analysis.settings, args.metrics, args.disable_maad, args.disable_acoustic)

    tmp_stereo_dir = results_folder / "tmp_stereo"
    tonality_paths: List[Path] = []

    print("\n⚙️ Procesando archivos (puede tardar segun la cantidad de metricas)...")
    if selected_file:
        stereo_path = ensure_stereo_file(selected_file, tmp_stereo_dir)
        tonality_paths.append(stereo_path)
        results = analysis.analyze_file(
            stereo_path,
            resample=args.resample,
        )
    else:
        results = analysis.analyze_folder(
            audio_folder,
            max_workers=args.max_workers,
            resample=args.resample,
        )

    # Tonality adicional (TNR/PR) si se pidio
    if "tonality" in args.metrics:
        if not tonality_paths:
            tonality_paths = wav_files
        ton_df = compute_tonality_metrics(tonality_paths)
        if not ton_df.empty:
            # results puede ser MultiIndex (Recording, Channel)
            try:
                results = results.join(ton_df, how="left")
            except Exception:
                # Si no calza, lo integramos despues de aplanar
                pass

    flat_df = flatten_results(results)
    if "tonality" in args.metrics and not ton_df.empty:
        if not {"tonality_tnr_db", "tonality_pr_db"}.issubset(flat_df.columns):
            flat_df = flat_df.merge(
                ton_df.reset_index(), on=["Recording", "Channel"], how="left"
            )

    csv_path = results_folder / "dp_dataset.csv"
    flat_df.to_csv(csv_path, index=False)
    print("  ✔ CSV generado:", csv_path)

    print("\n🎨 Graficos simples (promedio por archivo)...")
    plot_bar(flat_df, args.compare_label, plots_folder)

    print("\n📊 Comparaciones entre audios...")
    compare_pairs(flat_df, args.compare_label, results_folder)

    print("\n🎉 Pipeline completo ejecutado con exito.")
    print("Puedes revisar:")
    print(f"  • CSV: {csv_path}")
    print(f"  • Graficos: {plots_folder}")
    print(f"  • Comparaciones: {results_folder}/comparaciones_{args.compare_label}.csv")


if __name__ == "__main__":
    main()
