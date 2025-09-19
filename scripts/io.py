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

def _cumprofit_from_any_json(obj) -> pd.DataFrame:
    """
    Convierte distintos formatos de JSON a un DataFrame:
      - lista de dicts -> DataFrame directo
      - dict con arrays por serie -> columnas por serie
      - dict anidado con 'y'/'values'/'data' -> usa esos arrays
    Devuelve DataFrame (puede estar vacío si no reconoce la estructura).
    """
    import pandas as pd

    if isinstance(obj, list):
        if len(obj) == 0:
            return pd.DataFrame()
        if isinstance(obj[0], dict):
            return pd.DataFrame(obj)
        # lista plana -> la tratamos como una serie única
        return pd.DataFrame({"x": range(1, len(obj) + 1), "Model (BASE)": obj})

    if isinstance(obj, dict):
        # 1) dict con arrays por serie
        cols = {}
        for k, v in obj.items():
            if isinstance(v, list):
                cols[k] = v
            elif isinstance(v, dict):
                for kk in ("y", "values", "data", "series"):
                    if kk in v and isinstance(v[kk], list):
                        cols[k] = v[kk]
                        break
        if cols:
            # iguala longitudes
            n = max(len(v) for v in cols.values())
            for k in list(cols.keys()):
                if len(cols[k]) < n:
                    cols[k] = cols[k] + [None] * (n - len(cols[k]))
            df = pd.DataFrame(cols)
            if "x" not in df.columns:
                df.insert(0, "x", range(1, len(df) + 1))
            return df

        # 2) dict con lista 'records'/'rows'/'items'
        for key in ("records", "rows", "items"):
            if isinstance(obj.get(key), list):
                return pd.DataFrame(obj[key])

    return pd.DataFrame()


@st.cache_data
def load_cumprofit(season: int) -> pd.DataFrame:
    """
    Carga y normaliza la curva de beneficio acumulado para una temporada:
      - intenta JSON (varios formatos), si falla usa CSV
      - normaliza eje 'x'
      - renombra series a: 'Model (BASE)', 'Model (SMOTE)', 'Bet365' (si existen)
    """
    import json
    p_json = BASE / "cumprofit_curves" / f"cumprofit_{season}.json"
    p_csv  = BASE / "cumprofit_curves" / f"cumprofit_{season}.csv"

    df = pd.DataFrame()
    if p_json.exists():
        try:
            # intento rápido: lista de records
            df = pd.read_json(p_json, orient="records")
        except Exception:
            # parseo manual y conversión laxa
            obj = json.loads(p_json.read_text(encoding="utf-8"))
            df = _cumprofit_from_any_json(obj)

    if (df is None or df.empty) and p_csv.exists():
        df = pd.read_csv(p_csv)

    if df is None or df.empty:
        return pd.DataFrame()

    df = _norm_cols(df)

    # Detectar/normalizar eje X
    for c in ["x", "match_idx", "step", "round", "i", "index", "n", "Match"]:
        if c in df.columns:
            if c != "x":
                df = df.rename(columns={c: "x"})
            break
    else:
        df.insert(0, "x", range(1, len(df) + 1))

    # Renombrar series a etiquetas estándar
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if c == "x":
            continue
        if "bet365" in lc or "benchmark" in lc:
            rename[c] = "Bet365"
        elif "smote" in lc:
            rename[c] = "Model (SMOTE)"
        elif "base" in lc or "model" in lc or "pred" in lc:
            rename[c] = "Model (BASE)"
    if rename:
        df = df.rename(columns=rename)

    keep = ["x"] + [c for c in ["Model (BASE)", "Model (SMOTE)", "Bet365"] if c in df.columns]
    # si ninguna serie coincide, deja todas salvo 'x'
    if len(keep) == 1:
        keep = [c for c in df.columns if c != "x"]
        keep.insert(0, "x")
    return df[keep]

@st.cache_data
def load_matchlog(model: str, season: int) -> pd.DataFrame:
    p = BASE / f"matchlogs_{model}" / f"matchlog_{season}.csv"
    return load_csv(str(p.relative_to(BASE))) if p.exists() else pd.DataFrame()
