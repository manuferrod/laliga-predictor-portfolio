# Home.py
import sys, importlib.util
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- import robusto del módulo io ---
ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import scripts.io as io
except Exception:
    spec = importlib.util.spec_from_file_location("io", SCRIPTS_DIR / "io.py")
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)  # type: ignore

st.set_page_config(page_title="LaLiga 1X2 · Modelo vs Bet365", page_icon="⚽", layout="wide")
st.title("LaLiga 1X2 · Modelo vs Bet365")

# === Temporada actual (máxima disponible) ===
season = io.current_season()
if season is None:
    st.warning("Aún no hay datos en outputs/.")
    st.stop()

st.caption(f"Temporada enfocada en portada: **{season}** (actual)")

tab_resumen, tab_privado = st.tabs(["📊 Resumen", "🔒 Zona privada"])

# =============== TAB: RESUMEN ===============
with tab_resumen:
    # Cargamos matchlog del modelo BASE (para resumen de resultados)
    df = io.load_matchlog("base", season)
    if df.empty:
        st.info(f"No encontré matchlogs para la temporada {season}.")
        st.stop()

    # Normaliza fecha y semana/jornada
    df = io._ensure_week_col(df)
    if "Date" in df.columns:
        df["Date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["Date_dt"] = pd.to_datetime(df.iloc[:,0], errors="coerce")  # fallback
    # Resultado real: target (0=A,1=D,2=H) o FTR (H/D/A)
    result_col = None
    for c in ["target","y_true","Result","FTR","ftr","resultado"]:
        if c in df.columns:
            result_col = c
            break

    # KPI temporales de la temporada
    played_mask = pd.Series(False, index=df.index)
    if result_col == "target":
        played_mask = df["target"].isin([0,1,2])
    elif result_col is not None:
        played_mask = df[result_col].astype(str).str.upper().isin(["H","D","A"])

    total_played = int(played_mask.sum())
    total_matches = int(len(df))
    # ROI temporada (BASE y SMOTE si existe)
    roi_base = io.load_roi("base")
    roi_smote = io.load_roi("smote")
    roi_b = roi_base.loc[roi_base.get("Season") == season, "ROI"]
    roi_s = roi_smote.loc[roi_smote.get("Season") == season, "ROI"]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Partidos disputados", f"{total_played}/{total_matches}")
    if not roi_b.empty: c2.metric("ROI modelo BASE", f"{float(roi_b.iloc[0]):.1%}")
    if not roi_s.empty: c3.metric("ROI modelo SMOTE", f"{float(roi_s.iloc[0]):.1%}")

    st.divider()

    # Filtro por semana/jornada
    weeks = df["Week"].dropna().astype(int).sort_values().unique().tolist()
    mostrar_todo = st.toggle("Mostrar toda la temporada", value=True)
    if mostrar_todo:
        df_view = df.copy()
        titulo = f"Temporada {season}"
    else:
        wk = st.select_slider("Filtrar por semana/jornada", options=weeks, value=weeks[-1])
        df_view = df[df["Week"].astype("Int64") == wk].copy()
        titulo = f"Temporada {season} — Semana/Jornada {wk}"

    # Resumen de resultados (solo jugados)
    sub = df_view[played_mask.reindex(df_view.index, fill_value=False)]
    home_w = away_w = draws = 0
    if result_col == "target":
        home_w = int((sub["target"] == 2).sum())
        draws  = int((sub["target"] == 1).sum())
        away_w = int((sub["target"] == 0).sum())
    elif result_col is not None:
        s = sub[result_col].astype(str).str.upper()
        home_w = int((s == "H").sum())
        draws  = int((s == "D").sum())
        away_w = int((s == "A").sum())

    d1,d2,d3 = st.columns(3)
    d1.metric("Victorias HOME", home_w)
    d2.metric("Empates", draws)
    d3.metric("Victorias AWAY", away_w)

    # Tabla de partidos (selección mínima de columnas)
    cols_basic = []
    for c in ["Date","Week","HomeTeam_norm","AwayTeam_norm", result_col, "value_pick", "value_ev", "B365H","B365D","B365A"]:
        if c and c in df_view.columns:
            cols_basic.append(c)
    if "Date" not in cols_basic and "Date_dt" in df_view.columns:
        cols_basic = ["Date_dt"] + [c for c in cols_basic if c != "Date_dt"]

    st.subheader(titulo)
    st.dataframe(df_view[cols_basic], use_container_width=True, hide_index=True)

    # (Opcional) pequeña curva acumulada de esta temporada (si tienes JSON/CSV de curvas)
    try:
        curv = io.load_cumprofit(season)
        if not curv.empty:
            long = curv.melt(id_vars="x", var_name="Serie", value_name="Beneficio")
            fig = px.line(long, x="x", y="Beneficio", color="Serie", title="Beneficio acumulado (YTD)")
            fig.update_layout(legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

# ============= TAB: ZONA PRIVADA (PIN) =============
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
        # Usamos matchlog BASE de la temporada actual para localizar FUTUROS (sin resultado)
        df = io.load_matchlog("base", season)
        if df.empty:
            st.info("No hay matchlog disponible.")
            st.stop()

        df = io._ensure_week_col(df)
        d = io._coerce_date_col(df)
        today = pd.Timestamp.now(tz="Europe/Madrid").normalize().tz_localize(None)

        # Futuros = sin resultado o con Date >= hoy
        no_target = pd.Series(True, index=df.index)
        if "target" in df.columns:
            no_target = ~df["target"].isin([0,1,2])
        elif "FTR" in df.columns:
            no_target = ~df["FTR"].astype(str).str.upper().isin(["H","D","A"])

        future_mask = no_target | (pd.to_datetime(d, errors="coerce") >= today)
        fut = df[future_mask].copy()
        if fut.empty:
            st.info("Ahora mismo no hay partidos futuros en outputs.")
            st.stop()

        # EV / pick (si no vienen dados, los calculamos)
        for k in ["value_pick","value_ev"]:
            if k not in fut.columns:
                fut[k] = np.nan

        # Probabilidades y cuotas -> EV por clase
        def _first_col(df, cands):
            for c in cands:
                if c in df.columns:
                    return c
            return None

        # intentamos construir value_pick/value_ev si faltan
        have_ev = fut["value_ev"].notna().any()
        if not have_ev:
            pH = _first_col(fut, ["p_H","proba_H","prob_H","proba_home","pHome"])
            pD = _first_col(fut, ["p_D","proba_D","prob_D","proba_draw","pDraw"])
            pA = _first_col(fut, ["p_A","proba_A","prob_A","proba_away","pAway"])
            if all([pH,pD,pA]) and all(c in fut.columns for c in ["B365H","B365D","B365A"]):
                evH = fut[pH]*fut["B365H"] - 1
                evD = fut[pD]*fut["B365D"] - 1
                evA = fut[pA]*fut["B365A"] - 1
                fut["value_ev"] = np.vstack([evA, evD, evH]).max(axis=0)
                arg = np.vstack([evA, evD, evH]).argmax(axis=0)
                fut["value_pick"] = np.where(arg==2,"H", np.where(arg==1,"D","A"))

        # Ordenamos por EV descendente
        fut = fut.sort_values("value_ev", ascending=False)

        # Selección de semana para las predicciones (próxima jornada)
        weeks = fut["Week"].dropna().astype(int).sort_values().unique().tolist()
        wk = st.selectbox("Jornada/Semana", weeks, index=len(weeks)-1)

        show = fut[fut["Week"].astype("Int64") == wk]
        cols = [c for c in ["Date","Week","HomeTeam_norm","AwayTeam_norm","value_pick","value_ev","B365H","B365D","B365A"] if c in show.columns]
        st.subheader(f"Predicciones privadas · Temporada {season} · Jornada {wk}")
        st.dataframe(show[cols], use_container_width=True, hide_index=True)

        st.caption("Nota: EV = p*odds - 1. value_pick = clase con mayor EV entre A/D/H.")
