from pathlib import Path
import json
import pandas as pd
import streamlit as st

BASE = Path("outputs")

@st.cache_data
def load_json(relpath: str):
    p = BASE / relpath
    with p.open(encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_csv(relpath: str):
    return pd.read_csv(BASE / relpath)

@st.cache_data
def seasons() -> list[int]:
    # 1) de classification_by_season_base.csv
    p = BASE / "classification_by_season_base.csv"
    if p.exists():
        s = pd.read_csv(p)["Season"]
        s = pd.to_numeric(s, errors="coerce").dropna().astype(int).unique().tolist()
        return sorted(s)
    # 2) de cumprofit_index.json
    p = BASE / "cumprofit_index.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        out = set()
        if isinstance(data, dict):
            if "seasons" in data:
                out |= set(map(int, data["seasons"]))
            for k in data.keys():
                try: out.add(int(k))
                except: pass
        return sorted(out)
    # 3) patrón de ficheros
    out = []
    for f in (BASE / "cumprofit_curves").glob("cumprofit_*.json"):
        s = f.stem.split("_")[-1]
        if s.isdigit(): out.append(int(s))
    return sorted(set(out))

@st.cache_data
def load_cumprofit(season: int) -> pd.DataFrame:
    # prefer JSON
    p_json = BASE / "cumprofit_curves" / f"cumprofit_{season}.json"
    p_csv  = BASE / "cumprofit_curves" / f"cumprofit_{season}.csv"
    if p_json.exists():
        df = pd.read_json(p_json, orient="records")
    elif p_csv.exists():
        df = pd.read_csv(p_csv)
    else:
        return pd.DataFrame()

    df = df.copy()
    # eje x
    for c in ["match_idx","step","round","i","index","n","Match"]:
        if c in df.columns:
            df.rename(columns={c:"x"}, inplace=True)
            break
    if "x" not in df.columns:
        df.insert(0, "x", range(1, len(df) + 1))
    # series
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if "bet365" in lc or "benchmark" in lc: rename[c] = "Bet365"
        elif "smote" in lc:                     rename[c] = "Model (SMOTE)"
        elif "base" in lc or "model" in lc:     rename[c] = "Model (BASE)"
    df = df.rename(columns=rename)
    keep = ["x"] + [c for c in ["Model (BASE)","Model (SMOTE)","Bet365"] if c in df.columns]
    return df[keep]

@st.cache_data
def load_roi(model: str) -> pd.DataFrame:
    p = BASE / f"roi_by_season_{model}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data
def load_matchlog(model: str, season: int) -> pd.DataFrame:
    p = BASE / f"matchlogs_{model}" / f"matchlog_{season}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()
