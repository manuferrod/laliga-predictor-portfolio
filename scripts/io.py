# scripts/io.py
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import streamlit as st
from typing import Any, Iterable

# === Localización robusta de outputs/ (independiente del cwd) ===
# .../repo-root/scripts/io.py  -> parents[1] == repo-root
BASE = (Path(__file__).resolve().parents[1] / "outputs").resolve()

# -------------------------
# Helpers de ruta y sanity
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

# -------------------------
# Lectura básica (fail-soft)
# -------------------------
@st.cache_data
def load_csv(relpath: str, **kwargs) -> pd.DataFrame:
    """
    Lee outputs/<relpath>. Devuelve DataFrame vacío si no existe
    o si hay error al leer. Normaliza columnas (strip).
    """
    p = _p(relpath)
    if not p.exists():
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
    """
    Lee outputs/<relpath>. Devuelve {} si no existe o hay error.
    """
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
def _pick_col(df: pd.DataFrame, *cands: str) -> str | None:
    low = {str(c).lower().strip(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    # Heurística: columna con años 1990..2100 mayoritariamente
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            if (s.dropna().between(1990, 2100).mean() > 0.6):
                return c
    return None

@st.cache_data
def seasons() -> list[int]:
    """
    Intenta inferir temporadas disponibles por este orden:
    1) classification_by_season_base.csv (col Season o test_season)
    2) cumprofit_index.json (si existe)
    3) ficheros en cumprofit_curves/
    4) ficheros en matchlogs_base/
    """
    # 1) Clasificación por temporada
    df_cls = load_csv("classification_by_season_base.csv")
    if not df_cls.empty:
        col = _pick_col(df_cls, "test_season", "Season", "season", "Temporada")
        if col:
            s = pd.to_numeric(df_cls[col], errors="coerce").dropna().astype(int)
            out = sorted(set(s.tolist()))
            if out:
                return out

    # 2) Índice de curvas (si lo tienes)
    idx = load_json("cumprofit_index.json")
    cand: set[int] = set()
    if isinstance(idx, list):
        for r in idx:
            try:
                cand.add(int(r.get("test_season")))
            except Exception:
                pass
    elif isinstance(idx, dict):
        for k in ("seasons", "test_seasons"):
            if isinstance(idx.get(k), (list, tuple)):
                for x in idx[k]:
                    try:
                        cand.add(int(x))
                    except Exception:
                        pass
    if cand:
        return sorted(cand)

    # 3) Curvas en carpeta
    vals = []
    curv = _p("cumprofit_curves")
    if curv.exists():
        for f in list(curv.glob("cumprofit_*.json")) + list(curv.glob("cumprofit_*.csv")):
            suf = f.stem.split("_")[-1]
            if suf.isdigit():
                vals.append(int(suf))
    if vals:
        return sorted(set(vals))

    # 4) Matchlogs del modelo base
    vals = []
    ml = _p("matchlogs_base")
    if ml.exists():
        for f in ml.glob("matchlog_*.csv"):
            suf = f.stem.split("_")[-1]
            if suf.isdigit():
                vals.append(int(suf))
    return sorted(set(vals))

@st.cache_data
def available_seasons() -> list[int]:
    return seasons()

@st.cache_data
def current_season() -> int | None:
    ss = seasons()
    return max(ss) if ss else None

# -------------------------
# Listados por carpeta
# -------------------------
@st.cache_data
def list_seasons_from_matchlogs(tag: str = "base") -> list[int]:
    folder = {"base": "matchlogs_base", "smote": "matchlogs_smote"}.get(tag, tag)
    root = _p(folder)
    if not root.exists():
        return []
    out = []
    for f in root.glob("matchlog_*.csv"):
        try:
            out.append(int(f.stem.split("_")[1]))
        except Exception:
            pass
    return sorted(set(out))

@st.cache_data
def list_seasons_from_curves() -> list[int]:
    root = _p("cumprofit_curves")
    if not root.exists():
        return []
    out = []
    for f in root.glob("cumprofit_*.json"):
        try:
            out.append(int(f.stem.split("_")[1]))
        except Exception:
            pass
    return sorted(set(out))

# -------------------------
# Cargas específicas (modelo)
# -------------------------
@st.cache_data
def load_classification_by_season(tag: str = "base") -> pd.DataFrame:
    return load_csv(f"classification_by_season_{tag}.csv")

@st.cache_data
def load_roc_by_season(tag: str = "base") -> pd.DataFrame:
    return load_csv(f"roc_by_season_{tag}.csv")

@st.cache_data
def load_roi_by_season(tag: str = "base") -> pd.DataFrame:
    df = load_csv(f"roi_by_season_{tag}.csv")
    if df.empty:
        return df
    # Normaliza Season/test_season -> Season
    col = _pick_col(df, "test_season", "Season", "season", "Temporada")
    if col and col != "Season":
        df = df.rename(columns={col: "Season"})
    # Normaliza ROI -> ROI
    if "ROI" not in df.columns:
        cand = next((c for c in df.columns if str(c).lower().startswith("roi")), None)
        if cand:
            df = df.rename(columns={cand: "ROI"})
    # Tipos
    if "ROI" in df.columns:
        df["ROI"] = pd.to_numeric(df["ROI"], errors="coerce")
    if "Season" in df.columns:
        df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")
    return df

@st.cache_data
def load_confusion_grid(tag: str = "base") -> dict:
    return load_json(f"confusion_grid_{tag}.json")

@st.cache_data
def load_classification_grid(tag: str = "base") -> dict:
    return load_json(f"classification_grid_{tag}.json")

@st.cache_data
def load_roc_grid(tag: str = "base") -> dict:
    return load_json(f"roc_grid_{tag}.json")

@st.cache_data
def load_matchlog(season: int, tag: str = "base") -> pd.DataFrame:
    folder = {"base": "matchlogs_base", "smote": "matchlogs_smote"}.get(tag, tag)
    return load_csv(f"{folder}/matchlog_{int(season)}.csv")

# -------------------------
# Baseline Bet365
# -------------------------
@st.cache_data
def load_bet365_metrics_by_season() -> pd.DataFrame:
    return load_csv("bet365_metrics_by_season.csv")

@st.cache_data
def load_bet365_grid() -> dict:
    return load_json("bet365_grid.json")

@st.cache_data
def load_bet365_matchlog(season: int) -> pd.DataFrame:
    return load_csv(f"bet365_matchlogs/matchlog_{int(season)}.csv")

@st.cache_data
def load_comparison_season(model_tag: str = "base") -> pd.DataFrame:
    return load_csv(f"comparison_season_{model_tag}_vs_bet365.csv")

# -------------------------
# Curvas cumprofit (modelo vs Bet365)
# -------------------------
def _cumprofit_from_any_json(obj: Any) -> pd.DataFrame:
    """
    Soporta varios formatos. Caso principal esperado:
      {"series":[{"i":..., "m":..., "b":..., ...}, ...]}
      donde:
        - i -> x (índice de partido)
        - m -> "Model (BASE)"
        - sm -> "Model (SMOTE)"  (opcional)
        - b -> "Bet365"
    Devuelve DF con columnas: x, Model (BASE), Model (SMOTE)?, Bet365?
    """
    if isinstance(obj, dict) and isinstance(obj.get("series"), list):
        rows = obj["series"]
        if rows and isinstance(rows[0], dict):
            df = pd.DataFrame(rows)
            df.columns = [str(c).strip() for c in df.columns]
            rename = {}
            if "i" in df.columns:  rename["i"]  = "x"
            if "m" in df.columns:  rename["m"]  = "Model (BASE)"
            if "sm" in df.columns: rename["sm"] = "Model (SMOTE)"
            if "b" in df.columns:  rename["b"]  = "Bet365"
            if rename:
                df = df.rename(columns=rename)
            keep = [c for c in ["x", "Model (BASE)", "Model (SMOTE)", "Bet365"] if c in df.columns]
            return df[keep] if keep else pd.DataFrame()

    # Fallbacks tolerantes
    if isinstance(obj, list):
        if len(obj) == 0:
            return pd.DataFrame()
        if isinstance(obj[0], dict):
            return pd.DataFrame(obj)
        return pd.DataFrame({"x": range(1, len(obj) + 1), "Model (BASE)": obj})

    if isinstance(obj, dict):
        # {"series":[{"name":..., "data":[...]} , ...]}
        ser = obj.get("series")
        if isinstance(ser, list):
            cols, x_ref = {}, None
            for s in ser:
                if not isinstance(s, dict):
                    continue
                name = str(s.get("name") or "").lower()
                data = s.get("data") or s.get("y") or s.get("values")
                if isinstance(s.get("x"), list) and x_ref is None:
                    x_ref = s["x"]
                if isinstance(data, list):
                    if "smote" in name:
                        cols["Model (SMOTE)"] = data
                    elif "bet365" in name or "benchmark" in name:
                        cols["Bet365"] = data
                    else:
                        cols["Model (BASE)"] = data
            if cols:
                n = max(len(v) for v in cols.values())
                for k in list(cols.keys()):
                    if len(cols[k]) < n:
                        cols[k] = cols[k] + [None] * (n - len(cols[k]))
                df = pd.DataFrame(cols)
                df.insert(0, "x", x_ref if (x_ref and len(x_ref) == n) else list(range(1, n + 1)))
                return df

        # dict con arrays por serie: {"base":[...], "bet365":[...]}
        cols = {}
        for k, v in obj.items():
            if isinstance(v, list):
                lk = str(k).lower()
                if "smote" in lk:
                    cols["Model (SMOTE)"] = v
                elif "bet365" in lk or "benchmark" in lk:
                    cols["Bet365"] = v
                else:
                    cols["Model (BASE)"] = v
        if cols:
            n = max(len(v) for v in cols.values())
            for k in list(cols.keys()):
                if len(cols[k]) < n:
                    cols[k] = cols[k] + [None] * (n - len(cols[k]))
            df = pd.DataFrame(cols)
            df.insert(0, "x", range(1, n + 1))
            return df

        # Último recurso: records/rows/items
        for key in ("records", "rows", "items"):
            if isinstance(obj.get(key), list):
                return pd.DataFrame(obj[key])

    return pd.DataFrame()

@st.cache_data
def load_cumprofit(season: int) -> pd.DataFrame:
    """
    Lee outputs/cumprofit_curves/cumprofit_<season>.json|csv y devuelve
    DataFrame con columnas:
      - x
      - Model (BASE) [opt]
      - Model (SMOTE) [opt]
      - Bet365       [opt]
    """
    p_json = _p("cumprofit_curves", f"cumprofit_{int(season)}.json")
    p_csv  = _p("cumprofit_curves", f"cumprofit_{int(season)}.csv")

    df = pd.DataFrame()
    if p_json.exists():
        try:
            obj = json.loads(p_json.read_text(encoding="utf-8"))
            df = _cumprofit_from_any_json(obj)
        except Exception:
            df = pd.DataFrame()

    if (df is None or df.empty) and p_csv.exists():
        try:
            df = pd.read_csv(p_csv)
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Normaliza eje X
    df = _norm_cols(df)
    if "x" not in df.columns:
        for c in ["match_num", "match_idx", "step", "round", "i", "index", "n", "Match"]:
            if c in df.columns:
                df = df.rename(columns={c: "x"})
                break
        else:
            df.insert(0, "x", range(1, len(df) + 1))

    # Orden y filtrado
    order = ["x", "Model (BASE)", "Model (SMOTE)", "Bet365"]
    keep = [c for c in order if c in df.columns]
    if len(keep) <= 1:  # solo 'x' => sin series
        return pd.DataFrame()
    return df[keep]

# -------------------------
# Extras útiles en páginas
# -------------------------
def _coerce_date_col(df: pd.DataFrame) -> pd.Series:
    for cand in ["Date", "date", "Fecha", "fecha"]:
        if cand in df.columns:
            return pd.to_datetime(df[cand], errors="coerce")
    return pd.to_datetime(pd.Series([None] * len(df)))

def _ensure_week_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intenta detectar columna de jornada; si no, crea ISO week a partir de la fecha.
    NO fuerza nombre si ya existe 'jornada' o similar.
    """
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
