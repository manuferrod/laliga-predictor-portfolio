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

def _rename_series(name: str) -> str:
    n = str(name or "").strip()
    l = n.lower()
    if "bet365" in l or "benchmark" in l:
        return "Bet365"
    if "smote" in l:
        return "Model (SMOTE)"
    if "base" in l or "model" in l or "pred" in l:
        return "Model (BASE)"
    return n or "Serie"

def _cumprofit_from_any_json(obj) -> pd.DataFrame:
    """
    Acepta varios formatos típicos:
      - lista de dicts -> DataFrame directo
      - dict con arrays por serie -> columnas por serie
      - dict con 'series': [{name, data|y|values, x?}, ...]
      - dict con 'records'/'rows'/'items'
    Devuelve DataFrame (posiblemente sólo con 'x' si no reconoce series).
    """
    if isinstance(obj, list):
        if not obj:
            return pd.DataFrame()
        if isinstance(obj[0], dict):
            return pd.DataFrame(obj)
        # lista plana -> serie única
        return pd.DataFrame({"x": range(1, len(obj) + 1), "Model (BASE)": obj})

    if isinstance(obj, dict):
        # Caso Highcharts / ECharts: {"series":[{"name":..., "data":[...] , "x": [...]}, ...]}
        if isinstance(obj.get("series"), list):
            cols, x_ref = {}, None
            for s in obj["series"]:
                if not isinstance(s, dict):
                    continue
                name = _rename_series(s.get("name"))
                data = None
                for k in ("data", "y", "values"):
                    if isinstance(s.get(k), list):
                        data = s[k]
                        break
                if data is None:
                    continue
                if isinstance(s.get("x"), list) and x_ref is None:
                    x_ref = s["x"]
                cols[name] = data
            if cols:
                n = max(len(v) for v in cols.values())
                for k in list(cols.keys()):
                    if len(cols[k]) < n:
                        cols[k] = cols[k] + [None] * (n - len(cols[k]))
                df = pd.DataFrame(cols)
                if x_ref is not None and len(x_ref) == n:
                    df.insert(0, "x", x_ref)
                else:
                    df.insert(0, "x", range(1, n + 1))
                return df

        # Dict con arrays por serie: {"base":[...], "smote":[...], "bet365":[...]}
        cols = {}
        for k, v in obj.items():
            if isinstance(v, list):
                cols[_rename_series(k)] = v
            elif isinstance(v, dict):
                for kk in ("y", "values", "data"):
                    if isinstance(v.get(kk), list):
                        cols[_rename_series(k)] = v[kk]
                        break
        if cols:
            n = max(len(v) for v in cols.values())
            for k in list(cols.keys()):
                if len(cols[k]) < n:
                    cols[k] = cols[k] + [None] * (n - len(cols[k]))
            df = pd.DataFrame(cols)
            df.insert(0, "x", range(1, n + 1))
            return df

        # Dict con 'records'/'rows'/'items'
        for key in ("records", "rows", "items"):
            if isinstance(obj.get(key), list):
                return pd.DataFrame(obj[key])

    return pd.DataFrame()

@st.cache_data
def load_cumprofit(season: int) -> pd.DataFrame:
    """
    Carga y normaliza la curva de beneficio acumulado:
      - intenta JSON (formato flexible), si no hay usa CSV
      - normaliza eje 'x'
      - renombra series a: 'Model (BASE)', 'Model (SMOTE)', 'Bet365'
    """
    import json
    p_json = BASE / "cumprofit_curves" / f"cumprofit_{season}.json"
    p_csv  = BASE / "cumprofit_curves" / f"cumprofit_{season}.csv"

    df = pd.DataFrame()
    if p_json.exists():
        try:
            df = pd.read_json(p_json, orient="records")
        except Exception:
            obj = json.loads(p_json.read_text(encoding="utf-8"))
            df = _cumprofit_from_any_json(obj)

    if (df is None or df.empty) and p_csv.exists():
        df = pd.read_csv(p_csv)

    if df is None or df.empty:
        return pd.DataFrame()

    df = _norm_cols(df)

    # Eje X
    for c in ["x", "match_idx", "step", "round", "i", "index", "n", "Match"]:
        if c in df.columns:
            if c != "x":
                df = df.rename(columns={c: "x"})
            break
    else:
        df.insert(0, "x", range(1, len(df) + 1))

    # Renombrar posibles series
    rename = {c: _rename_series(c) for c in df.columns if c != "x"}
    df = df.rename(columns=rename)

    # Si no hay ninguna de las etiquetas estándar, deja todas menos 'x'
    series_cols = [c for c in df.columns if c != "x"]
    if not series_cols:
        return pd.DataFrame()

    # Orden sugerido
    order = ["Model (BASE)", "Model (SMOTE)", "Bet365"]
    ordered = ["x"] + [c for c in order if c in df.columns] + [c for c in series_cols if c not in order]
    return df[ordered]

@st.cache_data
def load_matchlog(model: str, season: int) -> pd.DataFrame:
    p = BASE / f"matchlogs_{model}" / f"matchlog_{season}.csv"
    return load_csv(str(p.relative_to(BASE))) if p.exists() else pd.DataFrame()
