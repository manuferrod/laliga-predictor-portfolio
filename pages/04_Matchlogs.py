# matchlogs.py
from __future__ import annotations

import sys, importlib.util, json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# --- import robusto del módulo scripts/io ---
ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    # Preferido
    from scripts.io import (
        ensure_outputs_dir,
        has_outputs,
        seasons as load_seasons,
        load_matchlog,
        _ensure_week_col,
        _coerce_date_col,
        BASE as OUTPUTS_BASE,
    )
except Exception:
    # Fallback defensivo
    spec = importlib.util.spec_from_file_location("io", SCRIPTS_DIR / "io.py")
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)  # type: ignore
    ensure_outputs_dir = io.ensure_outputs_dir
    has_outputs        = io.has_outputs
    load_seasons       = io.seasons
    load_matchlog      = io.load_matchlog
    _ensure_week_col   = io._ensure_week_col
    _coerce_date_col   = io._coerce_date_col
    OUTPUTS_BASE       = io.BASE

st.set_page_config(page_title="Matchlogs", page_icon="📋", layout="wide")
st.header("Matchlogs por temporada")

# --- existencia de outputs ---
ensure_outputs_dir()
if not has_outputs():
    st.warning("No hay artefactos en `outputs/` todavía.")
    st.stop()

# --- temporadas disponibles ---
seas = load_seasons()
if not seas:
    st.warning("No hay temporadas detectadas en `outputs/`.")
    st.stop()

# --- controles superiores ---
c1, c2, c3 = st.columns([1,1,2])
with c1:
    model = st.radio("Modelo", ["base", "smote"], horizontal=True)
with c2:
    sel = st.selectbox("Temporada", seas, index=len(seas)-1)
with c3:
    team = st.text_input("Filtrar por equipo (contiene)", placeholder="barcelona, betis, ...")

# --- carga del matchlog ---
df = load_matchlog(model, sel)
if df.empty:
    st.info(f"No hay matchlog para {model} / {sel}.")
    st.stop()

# --- normaliza fecha/jornada ---
df = df.copy()
df = _ensure_week_col(df)  # crea/renombra 'Week' si es necesario
# preserva 'jornada' si existe
has_jornada = "jornada" in df.columns
date_ser = _coerce_date_col(df)
df["Date_dt"] = pd.to_datetime(date_ser, errors="coerce")
df = df.sort_values(["Date_dt", "HomeTeam_norm", "AwayTeam_norm"], na_position="last").reset_index(drop=True)

# --- filtros opcionales ---
with st.expander("Filtros"):
    cols = st.columns(4)
    with cols[0]:
        only_value = st.checkbox("Solo value bets", value=False, help="Filtra filas con `use_value=True` si existe.")
    with cols[1]:
        only_played = st.checkbox("Solo disputados", value=False, help="Filtra por partidos con resultado real disponible.")
    with cols[2]:
        # elegir jornada (prioriza 'jornada' y si no, 'Week')
        if has_jornada:
            weeks_all = pd.to_numeric(df["jornada"], errors="coerce").dropna().astype(int).sort_values().unique().tolist()
            label_week = "Jornada"
        else:
            weeks_all = pd.to_numeric(df["Week"], errors="coerce").dropna().astype(int).sort_values().unique().tolist()
            label_week = "Week"
        wk_sel = st.selectbox(label_week, options=["(todas)"] + weeks_all, index=0)
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

# filtro por value bets
if only_value and "use_value" in df.columns:
    df = df[df["use_value"] == True]

# filtro por disputados
if only_played:
    if "true_result" in df.columns:
        df = df[df["true_result"].isin([0,1,2])]
    elif "FTR" in df.columns:
        df = df[df["FTR"].astype(str).str.upper().isin(["H","D","A"])]

# filtro por jornada/semana
if wk_sel != "(todas)":
    if has_jornada:
        df = df[pd.to_numeric(df["jornada"], errors="coerce").astype("Int64") == int(wk_sel)]
    else:
        df = df[pd.to_numeric(df["Week"], errors="coerce").astype("Int64") == int(wk_sel)]

# orden final
if order_ev_desc and "value_ev" in df.columns:
    df = df.sort_values("value_ev", ascending=False)

# --- KPIs rápidos ---
n_rows = int(len(df))
roi_pick = roi_value = None

# ROI del pick del modelo si existe net_profit
if "net_profit" in df.columns and n_rows > 0:
    try:
        roi_pick = float(pd.to_numeric(df["net_profit"], errors="coerce").fillna(0).sum() / n_rows)
    except Exception:
        roi_pick = None

# ROI de value bets si existen columnas
if {"use_value", "value_net_profit"}.issubset(df.columns):
    mask_v = df["use_value"] == True
    n_v = int(mask_v.sum())
    if n_v > 0:
        roi_value = float(pd.to_numeric(df.loc[mask_v, "value_net_profit"], errors="coerce").fillna(0).sum() / n_v)

k1, k2, k3 = st.columns(3)
k1.metric("Filas visibles", f"{n_rows}")
if roi_pick is not None:
    k2.metric("ROI pick modelo", f"{roi_pick:.1%}")
if roi_value is not None:
    k3.metric("ROI value bets", f"{roi_value:.1%}")

st.divider()

# --- selección de columnas amigables ---
ordered_cols = []
# clave temporal y contexto
for c in ["Date", "Date_dt", "jornada", "Week", "HomeTeam_norm", "AwayTeam_norm"]:
    if c in df.columns: ordered_cols.append(c)
# predicción del modelo
for c in ["Pred", "predicted_result", "predicted_prob", "predicted_odds", "edge"]:
    if c in df.columns: ordered_cols.append(c)
# value bets
for c in ["value_pick", "value_ev", "value_prob", "value_odds", "use_value"]:
    if c in df.columns: ordered_cols.append(c)
# cuotas
for c in ["B365H", "B365D", "B365A"]:
    if c in df.columns: ordered_cols.append(c)
# verdad y métricas
for c in ["true_result", "Correct", "value_correct", "bet_return", "net_profit", "value_bet_return", "value_net_profit", "Cum_net_profit"]:
    if c in df.columns: ordered_cols.append(c)

# añade el resto (si no estaban)
ordered_cols += [c for c in df.columns if c not in ordered_cols]

view = df[ordered_cols].copy()
# formateos suaves
if "Date_dt" in view.columns:
    view["Date_dt"] = pd.to_datetime(view["Date_dt"], errors="coerce").dt.strftime("%Y-%m-%d")
if "value_ev" in view.columns:
    view["value_ev"] = pd.to_numeric(view["value_ev"], errors="coerce").round(3)
if "edge" in view.columns:
    view["edge"] = pd.to_numeric(view["edge"], errors="coerce").round(3)

st.dataframe(view, use_container_width=True, hide_index=True)

# --- descargas ---
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        "Descargar CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name=f"matchlog_{sel}_{model}.csv",
        mime="text/csv",
    )
with col_dl2:
    st.download_button(
        "Descargar JSON",
        data=view.to_json(orient="records", force_ascii=False).encode("utf-8"),
        file_name=f"matchlog_{sel}_{model}.json",
        mime="application/json",
    )
