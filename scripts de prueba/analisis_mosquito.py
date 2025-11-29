import csv
from pathlib import Path

import librosa
import numpy as np
import mosqito as mq
from tkinter import Tk, filedialog

# Ocultar ventana principal
Tk().withdraw()

print("Selecciona un archivo de audio WAV...")
file_path = filedialog.askopenfilename(
    filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
)

if file_path == "":
    print("❌ No seleccionaste archivo.")
    exit()

print(f"🔍 Procesando: {file_path}")

# Cargar audio
signal, fs = librosa.load(file_path, sr=None)

# -----------------------
# LOUDNESS (ZWST)
# -----------------------
# loudness_zwst devuelve una tupla (N, N_specific, bark_axis)
N, N_specific, bark_axis = mq.loudness_zwst(signal, fs)
loudness = float(np.mean(np.atleast_1d(N)))

# -----------------------
# SHARPNESS (DIN)
# -----------------------
# En la versión 1.2.x la función pública es sharpness_din_st
sharp = float(np.mean(np.atleast_1d(mq.sharpness_din_st(signal, fs))))

# -----------------------
# ROUGHNESS (Daniel & Weber)
# -----------------------
R, R_specific, bark_axis_rough, time_axis = mq.roughness_dw(signal, fs)
rough = float(np.mean(np.atleast_1d(R)))

# -----------------------
# TONALITY (ECMA-418-1)
# -----------------------
# -----------------------
# TONALITY (ECMA-418-1)
# -----------------------
# La librería expone TNR (tone-to-noise ratio) y PR (prominence ratio)
tnr_total, tnr_tones, tnr_prom, tones_freqs = mq.tnr_ecma_st(signal, fs)
pr_total, pr_tones, pr_prom, pr_freqs = mq.pr_ecma_st(signal, fs)
tonality = float(np.mean(np.atleast_1d(tnr_total)))

# -----------------------
# GUARDAR EN CSV
# -----------------------
csv_row = {
    "archivo": file_path,
    "fs_hz": fs,
    "loudness_sones": loudness,
    "sharpness_acum": sharp,
    "roughness_asper": rough,
    "tonality_tnr_db": tonality,
    "tnr_tones_db": ";".join(f"{v:.3f}" for v in np.atleast_1d(tnr_tones)),
    "tones_freqs_hz": ";".join(f"{v:.1f}" for v in np.atleast_1d(tones_freqs)),
}

csv_path = Path(file_path).with_name(Path(file_path).stem + "_mosquito.csv")
write_header = not csv_path.exists()
with csv_path.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_row.keys())
    if write_header:
        writer.writeheader()
    writer.writerow(csv_row)

print("\n=== RESULTADOS PSICOACÚSTICOS ===")
print("Loudness (sones):", loudness)
print("Sharpness (acum):", sharp)
print("Roughness (asper):", rough)
print("Tonality (TNR, dB):", tonality)
# Si quieres ver tonos detectados:
# print("TNR tonos (dB):", tnr_tones)
# print("Frecuencias tonos (Hz):", tones_freqs)
print(f"CSV guardado en: {csv_path}")
