from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from visualization.plots_correlacion import parse_time_group


def plot_distributions_by_time(
    df: pd.DataFrame,
    descriptors: Iterable[str],
    run_slug: str,
    out_dir: Path,
    kind: str = "box",
    include_unknown: bool = False,
) -> list[Path]:
    """
    Genera boxplots/violin plots por descriptor, agrupados por time_group derivado de Recording.
    Omite grupos con <5 UTS.
    """
    paths: list[Path] = []
    if df is None or df.empty:
        print("[aviso] No hay datos UTS para distribuciones por momento del día.")
        return paths
    if "Recording" not in df.columns:
        print("[aviso] No hay columna Recording; no se generan distribuciones por momento del día.")
        return paths

    base = df.copy()
    base["Recording_norm"] = base["Recording"].apply(parse_time_group.__globals__["normalize_recording_name"])
    base["time_group"] = base["Recording"].apply(parse_time_group)
    # Diagnóstico
    sample_map = base[["Recording", "Recording_norm", "time_group"]].head(10).to_records(index=False)
    print(f"[info] (boxplot) Ejemplos Recording->norm->time_group: {list(sample_map)}")
    print(f"[info] (boxplot) Conteo por time_group antes de filtro: {base['time_group'].value_counts().to_dict()}")
    if not include_unknown:
        unknown_n = (base["time_group"] == "UNKNOWN").sum()
        if unknown_n:
            print(f"[aviso] Excluyendo UNKNOWN en boxplots (n={unknown_n}). Usa include_unknown=True para incluirlos.")
        base = base[base["time_group"] != "UNKNOWN"]
    if base.empty:
        print("[aviso] Sin datos tras filtrar UNKNOWN para boxplots.")
        return paths
    out_dir.mkdir(parents=True, exist_ok=True)

    for desc in descriptors:
        if desc not in base.columns:
            # Resolver *_mean si existe
            mean_col = f"{desc}_mean"
            if mean_col in base.columns:
                base[desc] = base[mean_col]
            else:
                continue
        sub = base[["time_group", desc]].dropna()
        counts = sub["time_group"].value_counts()
        valid_groups = counts[counts >= 5].index
        sub = sub[sub["time_group"].isin(valid_groups)]
        if sub.empty or sub["time_group"].nunique() < 1:
            print(f"[aviso] No se grafica {desc}: grupos insuficientes o <5 UTS por grupo.")
            continue

        plt.figure(figsize=(8, 6))
        if kind == "violin":
            sns.violinplot(data=sub, x="time_group", y=desc, inner="box", palette="Set2")
        else:
            sns.boxplot(data=sub, x="time_group", y=desc, palette="Set2")
        plt.xlabel("Momento del día (time_group)")
        plt.ylabel(desc)
        plt.title(f"Distribución de {desc} por momento del día (UTS)")
        plt.suptitle(f"{run_slug} | Unidad: UTS | Grupos con N>=5", fontsize=9)
        plt.tight_layout()

        out_png = out_dir / f"{desc}_by_time.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        paths.append(out_png)
    return paths
