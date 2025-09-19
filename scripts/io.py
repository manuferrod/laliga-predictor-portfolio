# scripts/io.py
from pathlib import Path
import json
import pandas as pd
import streamlit as st

BASE = Path("outputs")

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

@st.cache_data
def load_csv(relpath: str) -> pd.DataFrame:
    p = BASE / relpath
    if not p.exists():
        raise FileNotFoundError(f"Falta {p}")
    try:
        df = pd.read_csv(p)
    except Exception:
        df = pd.read_csv(p, sep=";")
    return _norm_cols(df)

@st.cache_data
def load_json(relpath: str):
    p = BASE / relpath
    if not p.exists():
        raise FileNotFoundError(f"Falta {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def _pick_col(df: pd.DataFrame, *cands: str) -> str | None:
    """Devuelve el nombre real de la primera columna candidata que exista (case-insensitive).
       Si no hay match, intenta detectar una 'temporada' numérica (2000–2100) por heurística.
    """
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    # heurística: columna numérica con valores mayoritariamente entre 1990 y 2100
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            ok = s.dropna().between(1990, 2100).mean() > 0.6
            if ok:
                return c
    return None

@st.cache_data
def seasons() -> list[int]:
    """Intenta inferir temporadas disponibles desde distintos outputs."""
    # 1) classification_by_season_base.csv (preferido)
    p = BASE / "classification_by_season_base.csv"
    if p.exists():
        df = load_csv("classification_by_season_base.csv")
        # 👇 añadimos 'test_season' a los candidatos
        col = _pick_col(df, "test_season", "Season", "season", "Temporada")
        if col:
            s = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
            out = sorted(set(s.tolist()))
            if out:
                return out

    # 2) cumprofit_index.json (alternativa)
    p = BASE / "cumprofit_index.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            cand = set()
            if isinstance(data, dict):
                if "seasons" in data and isinstance(data["seasons"], (list, tuple)):
                    cand |= {int(x) for x in data["seasons"] if str(x).isdigit()}
                for k in data.keys():
                    if str(k).isdigit():
                        cand.add(int(k))
            out = sorted(cand)
            if out:
                return out
        except Exception:
            pass

    # 3) escaneo de ficheros de curvas
    vals = []
    curv = BASE / "cumprofit_curves"
    if curv.exists():
        for f in list(curv.glob("cumprofit_*.json")) + list(curv.glob("cumprofit_*.csv")):
            suf = f.stem.split("_")[-1]
            if suf.isdigit():
                vals.append(int(suf))
    return sorted(set(vals))

@st.cache_data
def load_roi(model: str) -> pd.DataFrame:
    """Carga roi_by_season_{model}.csv y normaliza nombres de columnas."""
    p = BASE / f"roi_by_season_{model}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = load_csv(f"roi_by_season_{model}.csv")

    # Normaliza Season/test_season -> Season
    season_col = _pick_col(df, "test_season", "Season", "season", "Temporada")
    if season_col and season_col != "Season":
        df = df.rename(columns={season_col: "Season"})

    # Normaliza ROI -> ROI
    if "ROI" not in df.columns:
        cand = next((c for c in df.columns if c.lower().startswith("roi")), None)
        if cand:
            df = df.rename(columns={cand: "ROI"})

    # Tipos
    if "ROI" in df.columns:
        df["ROI"] = pd.to_numeric(df["ROI"], errors="coerce")
    if "Season" in df.columns:
        df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")

    return df

@st.cache_data
def load_matchlog(model: str, season: int) -> pd.DataFrame:
    p = BASE / f"matchlogs_{model}" / f"matchlog_{season}.csv"
    return load_csv(str(p.relative_to(BASE))) if p.exists() else pd.DataFrame()
