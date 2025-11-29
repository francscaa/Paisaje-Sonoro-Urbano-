from __future__ import annotations

import numpy as np
import mosqito as mq


def compute_psycho(segment: np.ndarray, sr: int) -> dict:
    """Calcula métricas psicoacústicas básicas para un segmento."""
    metrics = {}
    try:
        N, *_ = mq.loudness_zwst(segment, sr)
        metrics["loudness_sones"] = float(np.mean(np.atleast_1d(N)))
    except Exception as exc:
        print(f"[aviso] Loudness fallo: {exc}")
        metrics["loudness_sones"] = np.nan
    try:
        S = mq.sharpness_din_st(segment, sr)
        metrics["sharpness_acum"] = float(np.mean(np.atleast_1d(S)))
    except Exception as exc:
        print(f"[aviso] Sharpness fallo: {exc}")
        metrics["sharpness_acum"] = np.nan
    try:
        R, *_ = mq.roughness_dw(segment, sr)
        metrics["roughness_asper"] = float(np.mean(np.atleast_1d(R)))
    except Exception as exc:
        print(f"[aviso] Roughness fallo: {exc}")
        metrics["roughness_asper"] = np.nan
    try:
        tnr_total, *_ = mq.tnr_ecma_st(segment, sr)
        pr_total, *_ = mq.pr_ecma_st(segment, sr)
        metrics["tonality_tnr_db"] = float(np.mean(np.atleast_1d(tnr_total)))
        metrics["tonality_pr_db"] = float(np.mean(np.atleast_1d(pr_total)))
    except Exception as exc:
        print(f"[aviso] Tonality fallo: {exc}")
        metrics["tonality_tnr_db"] = np.nan
        metrics["tonality_pr_db"] = np.nan
    return metrics
