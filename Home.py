# Home.py — LaLiga 1X2 · 25/26 (solo temporada actual)
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

# Utilidades para detección de 'value' al vuelo
def _compute_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura columnas: value_ev, value_pick, Partido_con_valor (bool)
    Si ya existe 'use_value', usa eso para 'Partido_con_valor'.
    Si no, intenta calcular EV con p_H/p_D/p_A y cuotas B365.
    """
    if df.empty:
        return df

    df = df.copy()

    # Si ya existe 'use_value', úsalo directamente
    if "use_value" in df.columns:
        if "Partido_con_valor" not in df.columns:
            df["Partido_con_valor"] = df["use_value"].fillna(False).astype(bool)
        return df

    # Si no hay value_ev/value_pick, intentamos calcular
    have_ev = "value_ev" in df.columns
    have_pick = "value_pick" in df.columns

    # Buscar columnas de probas
    pH = next((c for c in ["p_H","proba_H","prob_H","proba_home","pHome"] if c in df.columns), None)
    pD = next((c for c in ["p_D","proba_D","prob_D","proba_draw","pDraw"] if c in df.columns), None)
    pA = next((c for c in ["p_A","proba_A","prob_A","proba_away","pAway"] if c in df.columns), None)

    if all([pH,pD,pA]) and all(c in df.columns for c in ["B365H","B365D","B365A"]):
        # calcula EV para cada signo
        evH = pd.to_numeric(df[pH], errors="coerce") * pd.to_numeric(df["B365H"], errors="coerce") - 1
        evD = pd.to_numeric(df[pD], errors="coerce") * pd.to_numeric(df["B365D"], errors="coerce") - 1
        evA = pd.to_numeric(df[pA], errors="coerce") * pd.to_numeric(df["B365A"], errors="coerce") - 1

        mat = np.vstack([
            evA.fillna(-np.inf).to_numpy(),
            evD.fillna(-np.inf).to_numpy(),
            evH.fillna(-np.inf).to_numpy()
        ])
        arg = np.argmax(mat, axis=0)  # 0->A, 1->D, 2->H
        best_ev = np.take_along_axis(mat, arg[np.newaxis,:], axis=0).ravel()

        # Crear columnas faltantes
        if not have_ev:
            df["value_ev"] = best_ev
        if not have_pick:
            df["value_pick"] = np.where(arg==2,"H", np.where(arg==1,"D","A"))

        # Partido con valor si EV>0 (umbral base; ajusta si quieres exigir margen)
        df["Partido_con_valor"] = (df["value_ev"] > 0).fillna(False)

    else:
        # Si no podemos calcular EV, al menos añade la bandera si ya viene en columnas existentes
        if "value_ev" in df.columns:
            df["Partido_con_valor"] = (pd.to_numeric(df["value_ev"], errors="coerce") > 0).fillna(False)
        elif "edge" in df.columns:
            # heurística: edge>0 indica valor
            df["Partido_con_valor"] = (pd.to_numeric(df["edge"], errors="coerce") > 0).fillna(False)
        else:
            df["Partido_con_valor"] = False

    return df

# =============== TAB PÚBLICA ===============
with tab_public:
    st.caption(f"Temporada: **{cur_season}** · Modelo: **{model.upper()}**")

    # 1) KPIs de temporada
    df = load_matchlog(model, cur_season).copy()
    if df.empty:
        st.warning(f"No hay matchlogs de {model.upper()} para {cur_season}.")
    else:
        df = _ensure_week_col(df)
        df["Date_dt"] = _coerce_date_col(df)
        df = _compute_value_columns(df)

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

        # ROI partidos con valor (media de value_net_profit sobre Partido_con_valor==True)
        roi_partidos_con_valor = np.nan
        cum_profit_value = np.nan
        n_value = int(df["Partido_con_valor"].sum()) if "Partido_con_valor" in df.columns else 0
        if n_value > 0:
            if "value_net_profit" in df.columns:
                vnet = pd.to_numeric(df.loc[df["Partido_con_valor"], "value_net_profit"], errors="coerce").fillna(0)
                cum_profit_value = float(vnet.sum())
                roi_partidos_con_valor = float(vnet.mean())
            else:
                # si no existe value_net_profit, aproximamos con net_profit de esos partidos
                vnet = pd.to_numeric(df.loc[df["Partido_con_valor"], "net_profit"], errors="coerce").fillna(0)
                cum_profit_value = float(vnet.sum())
                roi_partidos_con_valor = float(vnet.mean())

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

        # KPIs
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Partidos disputados", f"{n_played}")
        if not np.isnan(hit_rate):          k2.metric("Acierto", f"{hit_rate:.1%}")
        if not np.isnan(n_hits):            k3.metric("# aciertos", f"{int(n_hits)}")
        if not np.isnan(roi_por_partido):   k4.metric("ROI por partido", f"{roi_por_partido:.1%}")
        if not np.isnan(roi_partidos_con_valor): 
            k5.metric("ROI partidos con valor", f"{roi_partidos_con_valor:.1%}")
        if roi_model_temp is not None:      k6.metric("ROI", f"{roi_model_temp:.2%}")

        c1, c2, c3 = st.columns(3)
        if not np.isnan(cum_profit):        c1.metric("Beneficio acumulado", f"{cum_profit:,.2f}")
        c2.metric("# partidos con valor", f"{n_value}")
        if not np.isnan(cum_profit_value):  c3.metric("Beneficio valor acumulado", f"{cum_profit_value:,.2f}")

    st.divider()

    # 2) Tabla por jornada (incluye bandera 'Partido_con_valor')
    st.subheader("Partidos y señales — jornada seleccionada")
    dfj = df.copy()
    if not dfj.empty and jornada is not None:
        dfj = dfj[pd.to_numeric(dfj["Week"], errors="coerce").astype("Int64") == int(jornada)]

    # Asegura columnas de value en la tabla
    dfj = _compute_value_columns(dfj)

    cols_show = [c for c in [
        "Date","Date_dt","Week","jornada",
        "HomeTeam_norm","AwayTeam_norm",
        "Pred","value_pick","value_ev","Partido_con_valor",
        "B365H","B365D","B365A",
        "Correct","net_profit","value_net_profit"
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

# =============== TAB PRIVADA: próxima jornada ===============
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
            # Calcula columnas de value al vuelo si faltan
            dfp = _compute_value_columns(dfp)

            wk_next = pd.to_numeric(dfp["Week"], errors="coerce").dropna().astype(int).unique().tolist()
            title_wk = wk_next[0] if wk_next else "próxima"
            st.subheader(f"Predicciones (privadas) · {cur_season} · {model.upper()} · Jornada {title_wk}")

            cols = [c for c in [
                "Date","Week","jornada",
                "HomeTeam_norm","AwayTeam_norm",
                "Pred","value_pick","value_ev","Partido_con_valor",
                "B365H","B365D","B365A"
            ] if c in dfp.columns]
            st.dataframe(dfp[cols], use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar predicciones (CSV)",
                dfp[cols].to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_{model}_{cur_season}_J{title_wk}.csv"
            )
