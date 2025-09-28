# Home.py
from __future__ import annotations

import sys, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# --- import robusto del módulo io ---
ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    # preferimos import explícito (cuando corre desde streamlit run)
    from scripts.io import (
        ensure_outputs_dir,
        has_outputs,
        available_seasons,
        current_season,
        load_matchlog,
        load_cumprofit,
        load_roi_by_season,
        load_bet365_metrics_by_season,
        _ensure_week_col,   # helpers internos, los usamos igual
        _coerce_date_col,
    )
except Exception:
    # fallback por si el import relativo falla en algún entorno
    spec = importlib.util.spec_from_file_location("io", SCRIPTS_DIR / "io.py")
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)  # type: ignore
    ensure_outputs_dir      = io.ensure_outputs_dir
    has_outputs             = io.has_outputs
    available_seasons       = io.available_seasons
    current_season          = io.current_season
    load_matchlog           = io.load_matchlog
    load_cumprofit          = io.load_cumprofit
    load_roi_by_season      = io.load_roi_by_season
    load_bet365_metrics_by_season = io.load_bet365_metrics_by_season
    _ensure_week_col        = io._ensure_week_col
    _coerce_date_col        = io._coerce_date_col

st.set_page_config(page_title="LaLiga 1X2 · Modelo vs Bet365", page_icon="⚽", layout="wide")
st.title("LaLiga 1X2 · Modelo vs Bet365")

# --- chequeo de outputs/ ---
ensure_outputs_dir()
if not has_outputs():
    st.warning(
        "No se encontraron artefactos en `outputs/`. "
        "Sube los ficheros generados por el repo de motor o espera a la sincronización."
    )
    st.stop()

# === Selector de temporada ===
seasons = available_seasons()
if not seasons:
    st.info("No pude inferir temporadas disponibles. ¿Están los CSV/JSON en `outputs/`?")
    st.stop()

default_season = current_season() or seasons[-1]
season = st.selectbox("Temporada", seasons, index=seasons.index(default_season))

st.caption(f"Portada enfocada en: **{season}**")

tab_resumen, tab_privado = st.tabs(["📊 Resumen", "🔒 Zona privada"])

# =========================================================
# TAB: RESUMEN
# =========================================================
with tab_resumen:
    # Matchlog del modelo BASE
    df = load_matchlog("base", season)
    if df.empty:
        st.info(f"No encontré matchlogs del modelo BASE para la temporada {season}.")
        st.stop()

    # Normaliza fecha y jornada/semana
    df = _ensure_week_col(df)
    df["Date_dt"] = _coerce_date_col(df)

    # Resultado real: preferimos 'true_result' (0/1/2); si no, 'target' o textual 'FTR'
    result_col = None
    for c in ["true_result", "target", "FTR", "Result", "ftr", "resultado"]:
        if c in df.columns:
            result_col = c
            break

    played_mask = pd.Series(False, index=df.index)
    if result_col in ("true_result", "target"):
        played_mask = pd.to_numeric(df[result_col], errors="coerce").isin([0, 1, 2])
    elif result_col is not None:
        played_mask = df[result_col].astype(str).str.upper().isin(["H", "D", "A"])

    total_played = int(played_mask.sum())
    total_matches = int(len(df))

    # ROI temporada (BASE, SMOTE y Bet365 si están)
    roi_base  = load_roi_by_season("base")
    roi_smote = load_roi_by_season("smote")
    roi_b365  = load_bet365_metrics_by_season()

    def _pick_roi(df_roi, col="ROI"):
        if df_roi.empty:
            return None
        # normaliza nombre de temporada
        sc = "Season" if "Season" in df_roi.columns else "test_season" if "test_season" in df_roi.columns else None
        if sc is None:
            return None
        sub = df_roi[pd.to_numeric(df_roi[sc], errors="coerce") == season]
        if sub.empty or col not in sub.columns:
            return None
        try:
            return float(pd.to_numeric(sub[col], errors="coerce").iloc[0])
        except Exception:
            return None

    r_base  = _pick_roi(roi_base, "ROI")
    r_smote = _pick_roi(roi_smote, "ROI")
    r_b365  = _pick_roi(roi_b365.rename(columns={"test_season":"Season","roi":"ROI"}) if not roi_b365.empty else pd.DataFrame(), "ROI")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Partidos disputados", f"{total_played}/{total_matches}")
    if r_base  is not None: c2.metric("ROI modelo BASE",  f"{r_base*100:.2f}%")
    if r_smote is not None: c3.metric("ROI modelo SMOTE", f"{r_smote*100:.2f}%")
    if r_b365  is not None: c4.metric("ROI Bet365",       f"{r_b365*100:.2f}%")

    st.divider()

    # Filtro por jornada/semana
    weeks = df["Week"].dropna().astype(int).sort_values().unique().tolist()
    col_toggle, col_wk = st.columns([1, 2])
    with col_toggle:
        mostrar_todo = st.toggle("Mostrar toda la temporada", value=True)
    if mostrar_todo:
        df_view = df.copy()
        titulo = f"Temporada {season}"
    else:
        with col_wk:
            wk = st.select_slider("Filtrar por jornada/semana", options=weeks, value=weeks[-1])
        df_view = df[df["Week"].astype("Int64") == wk].copy()
        titulo = f"Temporada {season} — Jornada/Semana {wk}"

    # Resumen de resultados (solo jugados)
    sub = df_view[played_mask.reindex(df_view.index, fill_value=False)]
    home_w = away_w = draws = 0
    if result_col in ("true_result", "target"):
        vals = pd.to_numeric(sub[result_col], errors="coerce")
        home_w = int((vals == 2).sum())
        draws  = int((vals == 1).sum())
        away_w = int((vals == 0).sum())
    elif result_col is not None:
        s = sub[result_col].astype(str).str.upper()
        home_w = int((s == "H").sum())
        draws  = int((s == "D").sum())
        away_w = int((s == "A").sum())

    d1, d2, d3 = st.columns(3)
    d1.metric("Victorias HOME", home_w)
    d2.metric("Empates", draws)
    d3.metric("Victorias AWAY", away_w)

    # Tabla de partidos (selección mínima de columnas)
    cols_basic = []
    for c in ["Date", "Week", "jornada", "HomeTeam_norm", "AwayTeam_norm",
              result_col, "Pred", "edge", "value_pick", "value_ev", "B365H", "B365D", "B365A",
              "net_profit", "value_net_profit"]:
        if c and c in df_view.columns:
            cols_basic.append(c)
    if "Date" not in cols_basic and "Date_dt" in df_view.columns:
        cols_basic = ["Date_dt"] + [c for c in cols_basic if c != "Date_dt"]

    st.subheader(titulo)
    st.dataframe(df_view[cols_basic], use_container_width=True, hide_index=True)

    # Curvas acumuladas (si existen)
    curv = load_cumprofit(season)
    if not curv.empty:
        st.subheader("Beneficio acumulado (Modelo vs Bet365)")
        st.line_chart(curv.set_index("x"), height=420)
        with st.expander("Ver datos de la curva"):
            st.dataframe(curv, use_container_width=True, hide_index=True)

# =========================================================
# TAB: ZONA PRIVADA (PIN)
# =========================================================
with tab_privado:
    st.write("Introduce tu PIN para ver las predicciones de la próxima jornada.")
    PIN_CORRECTO = st.secrets.get("APP_PIN", "")
    ok = st.session_state.get("pin_ok", False)

    if not ok:
        pin = st.text_input("PIN", type="password")
        if st.button("Entrar"):
            if PIN_CORRECTO and pin == PIN_CORRECTO:
                st.session_state["pin_ok"] = True
                ok = True
            else:
                st.error("PIN incorrecto.")

    if ok:
        df = load_matchlog("base", season)
        if df.empty:
            st.info("No hay matchlog disponible.")
            st.stop()

        df = _ensure_week_col(df)
        d = _coerce_date_col(df)
        today = pd.Timestamp.now(tz="Europe/Madrid").normalize().tz_localize(None)

        # Futuros = sin resultado o con Date >= hoy
        no_target = pd.Series(True, index=df.index)
        if "true_result" in df.columns:
            no_target = ~pd.to_numeric(df["true_result"], errors="coerce").isin([0, 1, 2])
        elif "target" in df.columns:
            no_target = ~pd.to_numeric(df["target"], errors="coerce").isin([0, 1, 2])
        elif "FTR" in df.columns:
            no_target = ~df["FTR"].astype(str).str.upper().isin(["H", "D", "A"])

        future_mask = no_target | (pd.to_datetime(d, errors="coerce") >= today)
        fut = df[future_mask].copy()
        if fut.empty:
            st.info("Ahora mismo no hay partidos futuros en outputs.")
            st.stop()

        # Si no existen value_pick/value_ev intentamos construirlos (solo si hay proba por clase)
        for k in ["value_pick", "value_ev"]:
            if k not in fut.columns:
                fut[k] = np.nan

        if fut["value_ev"].isna().all():
            # Intento de cálculo desde probabilidades si existieran
            pH = next((c for c in ["p_H", "proba_H", "prob_H", "proba_home", "pHome"] if c in fut.columns), None)
            pD = next((c for c in ["p_D", "proba_D", "prob_D", "proba_draw", "pDraw"] if c in fut.columns), None)
            pA = next((c for c in ["p_A", "proba_A", "prob_A", "proba_away", "pAway"] if c in fut.columns), None)
            if all([pH, pD, pA]) and all(c in fut.columns for c in ["B365H", "B365D", "B365A"]):
                evH = pd.to_numeric(fut[pH], errors="coerce") * pd.to_numeric(fut["B365H"], errors="coerce") - 1
                evD = pd.to_numeric(fut[pD], errors="coerce") * pd.to_numeric(fut["B365D"], errors="coerce") - 1
                evA = pd.to_numeric(fut[pA], errors="coerce") * pd.to_numeric(fut["B365A"], errors="coerce") - 1
                mat = np.vstack([evA.fillna(-np.inf), evD.fillna(-np.inf), evH.fillna(-np.inf)])
                arg = np.argmax(mat, axis=0)
                fut["value_ev"] = np.take_along_axis(mat, arg[np.newaxis, :], axis=0).ravel()
                fut["value_pick"] = np.where(arg == 2, "H", np.where(arg == 1, "D", "A"))

        # Orden por EV descendente
        fut = fut.sort_values("value_ev", ascending=False)

        # Selector de jornada para predicciones (última por defecto)
        weeks = fut["Week"].dropna().astype(int).sort_values().unique().tolist()
        wk = st.selectbox("Jornada/Semana", weeks, index=len(weeks) - 1)

        show = fut[fut["Week"].astype("Int64") == wk]
        cols = [c for c in ["Date", "Week", "jornada", "HomeTeam_norm", "AwayTeam_norm",
                            "Pred", "value_pick", "value_ev", "B365H", "B365D", "B365A"] if c in show.columns]
        st.subheader(f"Predicciones privadas · Temporada {season} · Jornada {wk}")
        st.dataframe(show[cols], use_container_width=True, hide_index=True)

        st.caption("Nota: EV = p*odds - 1. `value_pick` = clase con mayor EV entre A/D/H.")
