# matchlogs.py — Visor de matchlogs (nuevos outputs)
from __future__ import annotations

import re
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# ================== Paths ==================
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

st.set_page_config(page_title="Matchlogs", page_icon="📋", layout="wide")
st.header("Matchlogs por temporada")

# ================== Helpers E/S ==================
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _list_seasons() -> list[int]:
    seasons = set()
    for p in OUT.glob("matchlogs_*.csv"):
        m = re.match(r"matchlogs_(\d{4})\.csv$", p.name)
        if m:
            seasons.add(int(m.group(1)))
    return sorted(seasons)

def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def _ensure_matchday(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "Matchday" not in df.columns:
        d = df.copy().reset_index(drop=True)
        d["Matchday"] = np.arange(1, len(d) + 1)
        return d
    return df

def _only_played_mask(df: pd.DataFrame) -> pd.Series:
    """Disputados = y_true ∈ {H,D,A}."""
    if "y_true" not in df.columns:
        return pd.Series(False, index=df.index)
    t = df["y_true"].astype(str).str.strip().str.upper()
    return t.isin({"H","D","A"})

def _source_file(source: str, season: int) -> Path:
    if source == "Modelo":
        return OUT / f"matchlogs_{season}.csv"
    else:
        return OUT / f"matchlogs_market_{season}.csv"

def _normalize_market_cols(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Unifica nombres para la vista cuando la fuente es Bet365."""
    if df.empty or source != "Bet365":
        return df
    d = df.copy()
    # Renombrar y_pred_market -> y_pred para homogenizar
    if "y_pred_market" in d.columns and "y_pred" not in d.columns:
        d = d.rename(columns={"y_pred_market": "y_pred"})
    # p*_mkt_pred -> proba_* (para homogeneizar con modelo)
    ren = {}
    if "pH_mkt_pred" in d.columns: ren["pH_mkt_pred"] = "proba_H"
    if "pD_mkt_pred" in d.columns: ren["pD_mkt_pred"] = "proba_D"
    if "pA_mkt_pred" in d.columns: ren["pA_mkt_pred"] = "proba_A"
    if ren:
        d = d.rename(columns=ren)
    return d

# ================== Comprobaciones ==================
OUT.mkdir(parents=True, exist_ok=True)
if not any(OUT.iterdir()):
    st.warning("No hay artefactos en `outputs/` todavía.")
    st.stop()

seasons = _list_seasons()
if not seasons:
    st.warning("No hay temporadas detectadas en `outputs/` (matchlogs_YYYY.csv).")
    st.stop()

# ================== Controles superiores ==================
c1, c2, c3 = st.columns([1,1,2])
with c1:
    source = st.radio("Fuente", ["Modelo", "Bet365"], horizontal=True)
with c2:
    sel = st.selectbox("Temporada", seasons, index=len(seasons)-1)
with c3:
    team = st.text_input("Filtrar por equipo (contiene)", placeholder="barcelona, betis, ...")

# ================== Carga y normalización ==================
path = _source_file(source, sel)
df = _read_csv(path)
if df.empty:
    st.info(f"No hay matchlog para {source} / {sel}.")
    st.stop()

df = _ensure_matchday(_ensure_date(df.copy()))
df = _normalize_market_cols(df, source)
df = df.sort_values(["Date","HomeTeam_norm","AwayTeam_norm"], na_position="last").reset_index(drop=True)

# ================== Filtros ==================
with st.expander("Filtros"):
    cols = st.columns(4)
    with cols[0]:
        only_bet_placed = st.checkbox(
            "Solo apuestas colocadas",
            value=False,
            help="Filtra filas con `bet_placed == True` si existe."
        )
    with cols[1]:
        only_played = st.checkbox(
            "Solo disputados",
            value=False,
            help="Filtra por partidos con resultado real (y_true en H/D/A)."
        )
    with cols[2]:
        # Jornada
        all_md = pd.to_numeric(df["Matchday"], errors="coerce").dropna().astype(int).sort_values().unique().tolist()
        wk_sel = st.selectbox("Jornada", options=["(todas)"] + all_md, index=0)
    with cols[3]:
        order_ev_desc = st.checkbox("Ordenar por EV (desc)", value=False)

# filtro por equipo
if team:
    t = team.strip()
    mask = pd.Series(False, index=df.index)
    for c in ["HomeTeam_norm", "AwayTeam_norm"]:
        if c in df.columns:
            mask = mask | df[c].astype(str).str.contains(t, case=False, na=False)
    df = df[mask]

# filtro solo apuestas colocadas
if only_bet_placed and "bet_placed" in df.columns:
    df = df[df["bet_placed"] == True]

# filtro solo disputados
if only_played:
    df = df[_only_played_mask(df)]

# filtro por jornada
if wk_sel != "(todas)":
    df = df[pd.to_numeric(df["Matchday"], errors="coerce").astype("Int64") == int(wk_sel)]

# orden final por EV
if order_ev_desc and "ev_pick" in df.columns:
    df = df.sort_values("ev_pick", ascending=False)

# ================== KPIs rápidos ==================
n_rows = int(len(df))
roi_pick = None
hit_rate = None
n_bets = None

# ROI del pick (profit)
if "profit" in df.columns and n_rows > 0:
    roi_pick = float(pd.to_numeric(df["profit"], errors="coerce").fillna(0).sum() / n_rows)

# Hit rate (si hay 'correct')
if "correct" in df.columns and n_rows > 0:
    hit_rate = float(pd.to_numeric(df["correct"], errors="coerce").where(lambda x: x.isin([0,1])).mean())

# Nº apuestas colocadas
if "bet_placed" in df.columns:
    n_bets = int((df["bet_placed"] == True).sum())

k1, k2, k3 = st.columns(3)
k1.metric("Filas visibles", f"{n_rows}")
if roi_pick is not None:
    k2.metric("ROI (pick)", f"{roi_pick:.1%}")
if hit_rate is not None:
    k3.metric("Acierto", f"{hit_rate:.1%}")

if n_bets is not None:
    st.caption(f"Apuestas colocadas en vista: **{n_bets}**")

st.divider()

# ================== Selección de columnas para vista ==================
# columnas propias de tus outputs (modelo/market)
preferred_cols = [
    # contexto
    "Date","Matchday","Season","HomeTeam_norm","AwayTeam_norm",
    # señales modelo / market
    "pred_key","y_true","y_pred",
    "proba_H","proba_D","proba_A",
    "pH_mkt","pD_mkt","pA_mkt",            # si vinieran en modelo
    "B365H","B365D","B365A","overround",
    # pick y valor esperado
    "odds_pick","p_pick","ev_pick","kelly_pick","bet_placed",
    # verificación y P&L
    "correct","profit","cum_profit_season",
]

# añade solo las que existan y luego el resto
ordered_cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
view = df[ordered_cols].copy()

# formateos suaves
if "Date" in view.columns:
    view["Date"] = pd.to_datetime(view["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
for c in ["proba_H","proba_D","proba_A","pH_mkt","pD_mkt","pA_mkt","p_pick","kelly_pick","ev_pick"]:
    if c in view.columns:
        view[c] = pd.to_numeric(view[c], errors="coerce").round(4)
for c in ["profit","cum_profit_season","odds_pick","B365H","B365D","B365A","overround"]:
    if c in view.columns:
        view[c] = pd.to_numeric(view[c], errors="coerce").round(3)

# renombrado amigable
rename_map = {
    "Date": "Fecha",
    "Matchday": "Jornada",
    "HomeTeam_norm": "Local",
    "AwayTeam_norm": "Visitante",
    "y_true": "Resultado real",
    "y_pred": "Predicción",
    "proba_H": "p(H)",
    "proba_D": "p(D)",
    "proba_A": "p(A)",
    "B365H": "Bet365 H",
    "B365D": "Bet365 D",
    "B365A": "Bet365 A",
    "overround": "Overround",
    "odds_pick": "Cuota pick",
    "p_pick": "p(pick)",
    "ev_pick": "EV pick",
    "kelly_pick": "Kelly",
    "bet_placed": "Apuesta colocada",
    "correct": "Acierto",
    "profit": "Beneficio",
    "cum_profit_season": "Beneficio acumulado (temp)",
}
view = view.rename(columns=rename_map)

st.dataframe(view, use_container_width=True, hide_index=True)

# ================== Descargas ==================
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        "Descargar CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name=f"matchlog_{sel}_{source.lower()}.csv",
        mime="text/csv",
    )
with col_dl2:
    st.download_button(
        "Descargar JSON",
        data=view.to_json(orient="records", force_ascii=False).encode("utf-8"),
        file_name=f"matchlog_{sel}_{source.lower()}.json",
        mime="application/json",
    )
