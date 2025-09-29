# Home.py — LaLiga 1X2 · 25/26 (solo temporada actual, sin "value")
from __future__ import annotations

import sys, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- import robusto de scripts/io.py ----
try:
    from scripts.io import (
        ensure_outputs_dir,
        has_outputs,
        available_seasons,
        current_season,
        load_matchlog,             # load_matchlog(model, season)
        load_cumprofit,            # load_cumprofit(season)
        load_roi_by_season,        # load_roi_by_season(model)
        load_bet365_metrics_by_season,
        load_csv,                  # genérico outputs/<file>.csv
        _ensure_week_col,
        _coerce_date_col,
    )
except Exception:
    spec = importlib.util.spec_from_file_location("io", SCRIPTS_DIR / "io.py")
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)  # type: ignore
    ensure_outputs_dir                  = io.ensure_outputs_dir
    has_outputs                         = io.has_outputs
    available_seasons                   = io.available_seasons
    current_season                      = io.current_season
    load_matchlog                       = io.load_matchlog
    load_cumprofit                      = io.load_cumprofit
    load_roi_by_season                  = io.load_roi_by_season
    load_bet365_metrics_by_season       = io.load_bet365_metrics_by_season
    load_csv                            = getattr(io, "load_csv", None)
    _ensure_week_col                    = io._ensure_week_col
    _coerce_date_col                    = io._coerce_date_col

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LaLiga 1X2 · 25/26", page_icon="⚽", layout="wide")
st.title("LaLiga 1X2 · 25/26")

ensure_outputs_dir()
if not has_outputs():
    st.warning("No se encontraron artefactos en `outputs/`. Sube/sincroniza y recarga.")
    st.stop()

# Temporada actual (solo esta en Home)
seasons = available_seasons()
cur_season = current_season() or (seasons[-1] if seasons else None)
if not cur_season:
    st.info("No pude inferir la temporada actual.")
    st.stop()

# ========= Filtros (debajo del título) =========
st.subheader("Filtros")
colf1, colf2, colf3 = st.columns([1, 1, 6])

with colf1:
    model = st.radio("Modelo", ["base", "smote"], horizontal=True)

# jornadas dependen del modelo en la temporada actual
_logs_tmp = _ensure_week_col(load_matchlog(model, cur_season))
jornadas = (
    pd.to_numeric(_logs_tmp.get("Week"), errors="coerce")
    .dropna().astype(int).sort_values().unique().tolist()
    if not _logs_tmp.empty else []
)

with colf2:
    jornada = st.selectbox("Jornada", jornadas if jornadas else [None],
                           index=len(jornadas)-1 if jornadas else 0)

# Tabs: pública (datos presentes) y privada (próxima jornada)
tab_public, tab_private = st.tabs(["📊 Temporada actual", "🔒 Zona privada (próxima jornada)"])

# =============== TAB PÚBLICA ===============
with tab_public:
    st.caption(f"Temporada: **{cur_season}** · Modelo: **{model.upper()}**")

    # 1) KPIs de temporada (sin métricas de value)
    df = load_matchlog(model, cur_season).copy()
    if df.empty:
        st.warning(f"No hay matchlogs de {model.upper()} para {cur_season}.")
    else:
        df = _ensure_week_col(df)
        # mantenemos Date_dt solo si lo necesitas para ordenar internamente; no se mostrará
        df["Date_dt"] = _coerce_date_col(df)

        # Partidos disputados (según columna de resultado)
        res_col = next((c for c in ["true_result","target","FTR","Result","ftr","resultado"] if c in df.columns), None)
        played_mask = pd.Series(False, index=df.index)
        if res_col in ("true_result","target"):
            played_mask = pd.to_numeric(df[res_col], errors="coerce").isin([0,1,2])
        elif res_col:
            played_mask = df[res_col].astype(str).str.upper().isin(["H","D","A"])
        n_played = int(played_mask.sum())

        # Acierto y # aciertos (si existe 'Correct')
        hit_rate = np.nan
        n_hits = np.nan
        if "Correct" in df.columns:
            corr = pd.to_numeric(df["Correct"], errors="coerce")
            hit_rate = float(corr.mean()) if len(corr) else np.nan
            n_hits = int(corr.sum()) if corr.notna().any() else np.nan

        # ROI por partido (media de net_profit sobre partidos disputados)
        roi_por_partido = np.nan
        cum_profit = np.nan
        if "net_profit" in df.columns and n_played > 0:
            net = pd.to_numeric(df["net_profit"], errors="coerce").fillna(0)
            cum_profit = float(net.sum())
            roi_por_partido = float(net.sum() / n_played)

        # ROI (agregado de temporada desde roi_by_season_{model})
        roi_model_temp = None
        roi_by_season = load_roi_by_season(model)
        season_col = next((c for c in ["Season","test_season","season"] if c in (roi_by_season.columns if not roi_by_season.empty else [])), None)
        if season_col:
            row = roi_by_season[pd.to_numeric(roi_by_season[season_col], errors="coerce") == pd.to_numeric(cur_season)]
            if not row.empty:
                roi_col = "roi" if "roi" in row.columns else next((c for c in row.columns if str(c).lower().startswith("roi")), None)
                if roi_col:
                    roi_model_temp = float(pd.to_numeric(row[roi_col], errors="coerce").iloc[0])

        # KPIs — con "Beneficio acumulado" en la derecha, bajo ROI
        k1, k2, k3, k4, k5 = st.columns(5)
        # Col 1: Partidos disputados + aciertos Y/X (con accuracy)
        k1.metric("Partidos disputados", f"{n_played}")
        with k1:
            if not np.isnan(hit_rate) and not np.isnan(n_hits):
                st.caption(f"Aciertos: **{int(n_hits)}/{n_played}** ({hit_rate:.1%})")
        # Col 2: Accuracy independiente (opcional)
        if not np.isnan(hit_rate):
            k2.metric("Acierto", f"{hit_rate:.1%}")
        # Col 3: # aciertos (métrica separada si lo quieres a la vista)
        if not np.isnan(n_hits):
            k3.metric("# aciertos", f"{int(n_hits)}")
        # Col 4: ROI por partido
        if not np.isnan(roi_por_partido):
            k4.metric("ROI por partido", f"{roi_por_partido:.1%}")
        # Col 5: ROI (agregado) y debajo Beneficio acumulado
        if roi_model_temp is not None:
            k5.metric("ROI", f"{roi_model_temp:.2%}")
        with k5:
            if not np.isnan(cum_profit):
                st.caption(f"Beneficio acumulado: **{cum_profit:,.2f}**")

    st.divider()

    # 2) Tabla por jornada (SIN columnas de value ni Date_dt)
    st.subheader("Partidos — jornada seleccionada")
    dfj = df.copy()
    if not dfj.empty and jornada is not None:
        dfj = dfj[pd.to_numeric(dfj["Week"], errors="coerce").astype("Int64") == int(jornada)]

    # columnas visibles (sin value_* ni Partido_con_valor ni Date_dt)
    cols_show = [c for c in [
        "Date","Week","jornada",
        "HomeTeam_norm","AwayTeam_norm",
        "Pred",
        "B365H","B365D","B365A",
        "Correct","net_profit"
    ] if c in dfj.columns]

    if not dfj.empty:
        st.dataframe(dfj[cols_show], use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar jornada (CSV)",
            dfj[cols_show].to_csv(index=False).encode("utf-8"),
            file_name=f"matchlog_{model}_{cur_season}_J{jornada or 'all'}.csv"
        )
    else:
        st.info("No hay filas para esa jornada.")

    st.divider()

    # 3) Curva acumulada (recortada hasta la jornada)
    st.subheader("Beneficio acumulado (temporada actual)")
    curves = load_cumprofit(cur_season)
    if not curves.empty:
        d = curves.copy()
        d.columns = [str(c).strip() for c in d.columns]
        x_col = "x"
        if "x" not in d.columns:
            for cand in ("match_num","index","i","step","round","n"):
                if cand in d.columns:
                    x_col = cand
                    break
            else:
                d.insert(0, "x", range(1, len(d)+1))
                x_col = "x"
        # mostramos solo serie del modelo + (si existe) Bet365
        keep = [c for c in d.columns if c.lower().find(model) >= 0 or c.lower().find("bet365") >= 0 or c == x_col]
        if len(keep) <= 1:
            keep = [x_col] + [c for c in d.columns if c != x_col]
        d = d[keep].copy()
        if jornada is not None and len(d) >= int(jornada):
            d = d.iloc[:int(jornada)]
        long = d.melt(id_vars=x_col, var_name="Serie", value_name="Beneficio")
        fig = px.line(long, x=x_col, y="Beneficio", color="Serie")
        fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), legend_title_text="")
        fig.update_xaxes(title_text="Partidos (acumulado)")
        fig.update_yaxes(title_text="Beneficio")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Ver datos de la curva"):
            st.dataframe(d, use_container_width=True, hide_index=True)
    else:
        st.info("No encontré curvas de cumprofit para la temporada actual.")

# =============== TAB PRIVADA: próxima jornada (sin columnas de value) ===============
with tab_private:
    st.write("Introduce tu PIN para ver las predicciones **de la próxima jornada**.")
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
        # Preferimos predictions_current_<modelo>.csv (futura jornada)
        dfp = pd.DataFrame()
        if callable(load_csv):
            try:
                dfp = load_csv(f"predictions_current_{model}.csv")
            except Exception:
                dfp = pd.DataFrame()

        # Fallback: próxima = max(Week)+1 del histórico del modelo
        if dfp.empty:
            df_all = _ensure_week_col(load_matchlog(model, cur_season).copy())
            if not df_all.empty:
                last_week = pd.to_numeric(df_all["Week"], errors="coerce").dropna().astype(int).max()
                dfp = df_all[df_all["Week"].astype("Int64") == (last_week + 1)]

        if dfp.empty:
            st.info("No hay predicciones para la próxima jornada todavía.")
        else:
            dfp = _ensure_week_col(dfp)
            cols = [c for c in [
                "Date","Week","jornada",
                "HomeTeam_norm","AwayTeam_norm",
                "Pred",
                "B365H","B365D","B365A"
            ] if c in dfp.columns]
            st.subheader(f"Predicciones (privadas) · {cur_season} · {model.upper()}")
            st.dataframe(dfp[cols], use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar predicciones (CSV)",
                dfp[cols].to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_{model}_{cur_season}_proxima.csv"
            )
