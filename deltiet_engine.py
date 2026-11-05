"""Núcleo Delt-IEt Battery Lab.

Compatibilidade explícita com o código Colab B0005 fornecido pelo autor.
Não altera a definição matemática do motor legado.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def shannon_entropy(values):
    """Entropia de Shannon da distribuição de amplitudes (implementação original)."""
    v = np.abs(np.array(values, dtype=float)) + 1e-20
    if v.size == 0:
        raise ValueError("A curva de tensão está vazia.")
    total = v.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("A curva de tensão não contém amplitudes válidas.")
    p = v / total
    return float(-np.sum(p * np.log(p)))


def deltiet_legacy(entropy_series, window_norm=10, percentile=95):
    """Reprodução da função deltiet() do notebook original.

    Preserva: np.diff(..., prepend=...), rolling mean/std incluindo o ponto
    corrente, ddof=0, substituição de sigma pequeno por 1e-12 e percentil
    calculado sobre a série inteira (retrospectivo).
    """
    S = np.asarray(entropy_series, dtype=float)
    if S.ndim != 1 or len(S) < 3:
        raise ValueError("São necessários pelo menos 3 ciclos com entropia.")
    if not np.isfinite(S).all():
        raise ValueError("A série de entropia contém valores ausentes ou infinitos.")
    if window_norm < 2:
        raise ValueError("A janela de normalização precisa ser >= 2.")
    if not 0 < percentile < 100:
        raise ValueError("O percentil deve estar entre 0 e 100.")

    dS = np.diff(S, prepend=S[0])
    d2S = np.diff(dS, prepend=dS[0])
    mu = pd.Series(S).rolling(window_norm, min_periods=1).mean().to_numpy()
    sig = pd.Series(S).rolling(window_norm, min_periods=1).std(ddof=0).to_numpy()
    sig = np.where(sig < 1e-12, 1e-12, sig)
    Z = (S - mu) / sig
    index = Z * np.abs(d2S)
    theta = float(np.percentile(np.abs(index), percentile))
    events = (np.abs(index) > theta).astype(int)
    return {
        "delta_S": dS, "delta2_S": d2S, "Z_t": Z,
        "DeltIEt": index, "theta": theta, "event": events
    }


def analyze_curves(curves, cycle_ids=None, capacities=None,
                   window_norm=10, percentile=95):
    """Calcula entropia por curva de descarga e aplica o motor legado."""
    curves = list(curves)
    if len(curves) < 3:
        raise ValueError("O arquivo precisa conter pelo menos 3 ciclos.")
    cycles = list(range(1, len(curves) + 1)) if cycle_ids is None else list(cycle_ids)
    if len(cycles) != len(curves):
        raise ValueError("Quantidade de identificadores de ciclo diferente da quantidade de curvas.")
    caps = [np.nan] * len(curves) if capacities is None else list(capacities)
    if len(caps) != len(curves):
        raise ValueError("Quantidade de capacidades diferente da quantidade de curvas.")

    entropies = [shannon_entropy(c) for c in curves]
    out = deltiet_legacy(entropies, window_norm=window_norm, percentile=percentile)
    df = pd.DataFrame({
        "cycle": cycles,
        "capacity_Ah": caps,
        "entropy_St": entropies,
        "delta_S": out["delta_S"],
        "delta2_S": out["delta2_S"],
        "Z_t": out["Z_t"],
        "DeltIEt": out["DeltIEt"],
        "abs_DeltIEt": np.abs(out["DeltIEt"]),
        "theta": out["theta"],
        "event": out["event"]
    })
    return df


def extract_nasa_mat(file_obj, battery_key=None):
    """Extrai curvas Voltage_measured e Capacity dos arquivos NASA .mat."""
    from scipy.io import loadmat
    mat = loadmat(file_obj, simplify_cells=True)
    candidates = [k for k in mat.keys() if not k.startswith("__")]
    if battery_key is None:
        battery_key = next((k for k in candidates if isinstance(mat[k], dict) and "cycle" in mat[k]), None)
    if battery_key is None or battery_key not in mat:
        raise ValueError("Não encontrei a estrutura da bateria no .mat.")
    cycle_data = mat[battery_key]["cycle"]
    if isinstance(cycle_data, dict):
        cycle_data = [cycle_data]
    curves, caps, ids = [], [], []
    for c in cycle_data:
        if str(c.get("type", "")).lower() != "discharge":
            continue
        data = c.get("data", {})
        if "Voltage_measured" not in data:
            continue
        voltage = np.asarray(data["Voltage_measured"], dtype=float).reshape(-1)
        cap = data.get("Capacity", np.nan)
        try:
            cap = float(cap)
        except (TypeError, ValueError):
            cap = np.nan
        curves.append(voltage)
        caps.append(cap)
        ids.append(len(ids) + 1)
    if not curves:
        raise ValueError("Nenhum ciclo de descarga com Voltage_measured foi encontrado.")
    return ids, curves, caps, battery_key
