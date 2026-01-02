# scripts/io.py
from __future__ import annotations

import numpy as np
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import datetime
from typing import Any, Iterable

# === Localización de outputs/ ===
BASE = (Path(__file__).resolve().parents[1] / "outputs").resolve()

# -------------------------
# Helpers
# -------------------------
def _p(*parts: Iterable[str | Path]) -> Path:
    return BASE.joinpath(*parts)

def ensure_outputs_dir() -> Path:
    BASE.mkdir(parents=True, exist_ok=True)
    return BASE

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def has_outputs() -> bool:
    return BASE.exists() and any(BASE.iterdir())

def get_data_last_update() -> str:
    """Devuelve la fecha del archivo más reciente en outputs/."""
    if not BASE.exists():
        return "Desconocida"
    try:
        # Filtramos solo archivos, ignorando carpetas
        files = [f for f in BASE.glob("**/*") if f.is_file()]
        if not files:
            return "Sin datos"
        latest = max(files, key=lambda f: f.stat().st_mtime)
        dt = datetime.datetime.fromtimestamp(latest.stat().st_mtime)
        meses = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 
                 7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}
        return f"{dt.day} de {meses.get(dt.month, 'Enero')}, {dt.year}"
    except Exception:
        return "Desconocida"

def _to_season_int(season) -> int:
    if season is None:
        raise ValueError("Temporada vacía")
    if isinstance(season, (int, np.integer)):
        return int(season)
    s = str(season).strip()
    try:
        return int(float(s))
    except Exception:
        import re
        m = re.search(r"(19|20)\d{2}", s)
        if m:
            return int(m.group(0))
        raise ValueError(f"Temporada no válida: {season!r}")

# -------------------------
# Lectura básica
# -------------------------
@st.cache_data
def load_csv(relpath: str, **kwargs) -> pd.DataFrame:
    p = _p(relpath)
    if not p.exists():
        # Intento fallback si no lo encuentra (ej: nombres antiguos vs nuevos)
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, **kwargs)
    except Exception:
        try:
            df = pd.read_csv(p, sep=";", **kwargs)
        except Exception:
            return pd.DataFrame()
    return _norm_cols(df)

@st.cache_data
def load_json(relpath: str) -> Any:
    p = _p(relpath)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

# -------------------------
# Descubrimiento de temporadas
# -------------------------
@st.cache_data
def list_seasons_from_matchlogs(tag: str = "base") -> list[int]:
    """Lista temporadas basándose en los archivos matchlogs_*.csv de la raíz."""
    if not BASE.exists():
        return []
    
    out = []
    # Diferenciamos base de market
    if tag in ("base", ""):
        pattern = "matchlogs_????.csv" # matchlogs_2025.csv
    elif tag in ("market", "bet365"):
        pattern = "matchlogs_market_????.csv"
    else:
        pattern = f"matchlogs_{tag}_????.csv"

    for f in BASE.glob(pattern):
        # Evitar falsos positivos (ej: market cuando buscamos base)
        if tag in ("base", "") and "market" in f.name:
            continue
            
        try:
            # Extraer año (los 4 dígitos del final del stem)
            year = f.stem.split("_")[-1]
            if year.isdigit():
                out.append(int(year))
        except Exception:
            pass
    return sorted(set(out))

@st.cache_data
def seasons() -> list[int]:
    """Infiere temporadas disponibles."""
    # 1. Intentar desde matchlogs base (lo más fiable ahora)
    ss = list_seasons_from_matchlogs("base")
    if ss:
        return ss
        
    # 2. Intentar desde clasificación
    df_cls = load_classification_by_season()
    if not df_cls.empty:
        col = next((c for c in df_cls.columns if c.lower() in ["season", "temporada", "test_season"]), None)
        if col:
            return sorted(df_cls[col].dropna().unique().astype(int).tolist())
            
    # 3. Fallback: buscar en curvas
    curv = _p("cumprofit_curves")
    if curv.exists():
        vals = []
        for f in curv.glob("cumprofit_*.json"):
            try:
                vals.append(int(f.stem.split("_")[1]))
            except: pass
        if vals:
            return sorted(set(vals))
            
    return []

@st.cache_data
def available_seasons() -> list[int]:
    return seasons()

@st.cache_data
def current_season() -> int | None:
    ss = seasons()
    return max(ss) if ss else None

# -------------------------
# Cargas específicas (Nuevos Nombres del Engine)
# -------------------------

@st.cache_data
def load_classification_by_season(tag: str = "base") -> pd.DataFrame:
    # Engine genera: classification_report_by_season.csv
    return load_csv("classification_report_by_season.csv")

@st.cache_data
def load_roc_by_season(tag: str = "base") -> pd.DataFrame:
    # El engine antiguo generaba CSV, el nuevo JSON: roc_curves_by_season.json
    # Si la UI espera DF, intentamos adaptar, si no, devolvemos vacío o cargamos JSON
    # NOTA: Si tu página Métricas espera un CSV con columnas, esto puede requerir ajuste en la página.
    # Por ahora intentamos devolver el JSON parseado a DF si es posible, o vacío.
    data = load_json("roc_curves_by_season.json")
    if data and isinstance(data, list):
         return pd.DataFrame(data)
    return pd.DataFrame() 

@st.cache_data
def load_roi_by_season(tag: str = "base") -> pd.DataFrame:
    # Engine genera: metrics_main_by_season.csv
    df = load_csv("metrics_main_by_season.csv")
    return df

@st.cache_data
def load_confusion_grid(tag: str = "base") -> dict:
    # Engine genera: confusion_matrices_by_season.json
    return load_json("confusion_matrices_by_season.json")

@st.cache_data
def load_matchlog(model: str, season: int) -> pd.DataFrame:
    try:
        s = _to_season_int(season)
    except:
        return pd.DataFrame()
    
    # Mapeo a estructura PLANA de outputs/
    if model in ("base", ""):
        return load_csv(f"matchlogs_{s}.csv")
    elif model in ("market", "bet365"):
        return load_csv(f"matchlogs_market_{s}.csv")
    else:
        return load_csv(f"matchlogs_{model}_{s}.csv")

# -------------------------
# Baseline Bet365
# -------------------------
@st.cache_data
def load_bet365_metrics_by_season() -> pd.DataFrame:
    # Engine genera: metrics_market_by_season.csv
    return load_csv("metrics_market_by_season.csv")

@st.cache_data
def load_bet365_grid() -> dict:
    # Engine genera: metrics_market_overall.json (o similar, ajusta si es grid)
    return load_json("metrics_market_overall.json")

@st.cache_data
def load_bet365_matchlog(season: int) -> pd.DataFrame:
    return load_matchlog("market", season)

@st.cache_data
def load_comparison_season(model_tag: str = "base") -> pd.DataFrame:
    # Este fichero quizás no se genera en el nuevo engine, devolver vacío para no romper
    return load_csv(f"comparison_season_{model_tag}_vs_bet365.csv")

# -------------------------
# Curvas cumprofit
# -------------------------
@st.cache_data
def load_cumprofit(season: int) -> pd.DataFrame:
    # Busca en outputs/cumprofit_curves/cumprofit_YYYY.json
    p_json = _p("cumprofit_curves", f"cumprofit_{int(season)}.json")
    
    if p_json.exists():
        try:
            obj = json.loads(p_json.read_text(encoding="utf-8"))
            # Adaptador rápido para el formato que espera la UI
            if "series" in obj:
                df = pd.DataFrame(obj["series"])
                # Renombrar claves cortas (json) a nombres UI si hace falta
                rename_map = {"i":"match_num", "d":"date", "m":"Model (BASE)", "b":"Bet365", "x":"x"}
                # Detectar qué claves vienen
                df = df.rename(columns=rename_map)
                
                # Asegurar columna 'x'
                if "match_num" in df.columns and "x" not in df.columns:
                    df["x"] = df["match_num"]
                elif "x" not in df.columns:
                    df["x"] = range(1, len(df)+1)
                
                # Asegurar nombres de series para el gráfico
                if "Model (BASE)" not in df.columns and "m" in df.columns: df["Model (BASE)"] = df["m"]
                if "Bet365" not in df.columns and "b" in df.columns: df["Bet365"] = df["b"]
                
                return df
        except Exception:
            pass
            
    # Fallback a CSV si existe
    p_csv = _p("cumprofit_curves", f"cumprofit_{int(season)}.csv")
    if p_csv.exists():
        df = pd.read_csv(p_csv)
        # Normalizar nombres
        renames = {"model_cum": "Model (BASE)", "bet365_cum": "Bet365", "match_num": "x"}
        df = df.rename(columns=renames)
        return df
        
    return pd.DataFrame()

# -------------------------
# Extras
# -------------------------
def _coerce_date_col(df: pd.DataFrame) -> pd.Series:
    for cand in ["Date", "date", "Fecha", "fecha"]:
        if cand in df.columns:
            return pd.to_datetime(df[cand], errors="coerce")
    return pd.to_datetime(pd.Series([None] * len(df)))

def _ensure_week_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc in {"matchweek", "week", "round", "jornada", "gw", "mw"}:
            if c != "Week":
                df.rename(columns={c: "Week"}, inplace=True)
            return df
    d = _coerce_date_col(df)
    if "Week" not in df.columns:
        df["Week"] = d.dt.isocalendar().week.astype("Int64")
    return df
