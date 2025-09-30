# Home.py — LaLiga 1X2 · 25/26 (temporada actual, KPIs con acc_test y Y/X desde 'Correct', ROI/ROI por partido juntos, STAKE select)
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
# Mismo ancho para los tres filtros
colf1, colf2, colf3 = st.columns([1, 1, 1])

with colf1:
    model = st.radio("Modelo", ["base", "smote"], horizontal=True)

_logs_tmp = _ensure_week_col(load_matchlog(model, cur_season))
jornadas = (
    pd.to_numeric(_logs_tmp.get("Week"), errors="coerce")
    .dropna().astype(int).sort_values().unique().tolist()
    if not _logs_tmp.empty else []
)

with colf2:
    jornada = st.selectbox(
        "Jornada",
        jornadas if jornadas else [None],
        index=len(jornadas) - 1 if jornadas else 0
    )

with colf3:
    stake = st.selectbox(
        "STAKE",
        options=list(range(1, 11)),
        index=0,
        help="€ por apuesta. Afecta solo al Beneficio (ROI no cambia)."
    )

# Tabs: pública (datos presentes) y privada (próxima jornada)
tab_public, tab_private = st.tabs(["📊 Temporada actual", "🔒 Zona privada (próxima jornada)"])

# ===== Helpers =====
def _euros(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}{abs(x):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")

def _load_metrics_by_season_for_model(model: str) -> pd.DataFrame:
    """Carga outputs/metrics_by_season*.csv según modelo."""
    fname = "metrics_by_season.csv" if model == "base" else "metrics_by_season_smote.csv"
    df = pd.DataFrame()
    if callable(load_csv):
        try:
            df = load_csv(fname)
        except Exception:
            df = pd.DataFrame()
    if df.empty:
        path = Path("outputs") / fname
        if path.exists():
            try:
                df = pd.read_csv(path)
            except Exception:
                df = pd.DataFrame()
    return df

def _extract_correct_series(df: pd.DataFrame) -> pd.Series:
    """
    Extrae una serie booleana de acierto a partir de la columna 'Correct':
      ✓ / ✔ / True / 1  -> True
      ✗ / ✘ / False / 0 -> False
      otros / NaN       -> NaN
    Si no existe 'Correct', devuelve serie vacía (NaN).
    """
    if df.empty or "Correct" not in df.columns:
        return pd.Series(index=df.index, dtype="float")

    s = df["Correct"]
    t = s.astype(str).str.strip().str.lower()

    true_tokens  = {"✓", "✔", "true", "1", "si", "sí", "y", "acierto", "correct", "correcto"}
    false_tokens = {"✗", "✘", "false", "0", "no", "n", "fallo", "incorrect", "incorrecto"}

    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[t.isin(true_tokens)]  = 1.0
    out[t.isin(false_tokens)] = 0.0

    # Si hay booleanos/números verdaderos, intenta mapearlos
    if s.dtype == bool:
        out = s.astype(float)
    else:
        num = pd.to_numeric(s, errors="coerce")
        out = out.fillna(num.where(num.isin([0, 1]), np.nan).astype("float"))

    return out

# =============== TAB PÚBLICA ===============
with tab_public:
    st.caption(f"Temporada: **{cur_season}** · Modelo: **{model.upper()}**")

    # 1) KPIs de temporada
    df = load_matchlog(model, cur_season).copy()
    if df.empty:
        st.warning(f"No hay matchlogs de {model.upper()} para {cur_season}.")
    else:
        df = _ensure_week_col(df)
        df["Date_dt"] = _coerce_date_col(df)  # no se muestra

        # Partidos jugados y aciertos/fallos a partir de 'Correct' (✓/✗)
        corr_series = _extract_correct_series(df)  # float {1.0, 0.0, NaN}
        played_mask = corr_series.notna()
        n_played = int(played_mask.sum())
        n_hits = int((corr_series == 1.0).sum()) if n_played > 0 else 0

        # Acierto (%) desde metrics_by_season (acc_test) para temporada actual
        acc_pct = float("nan")
        metrics_df = _load_metrics_by_season_for_model(model)
        if not metrics_df.empty and {"test_season", "acc_test"}.issubset(metrics_df.columns):
            row = metrics_df[pd.to_numeric(metrics_df["test_season"], errors="coerce") == pd.to_numeric(cur_season)]
            if not row.empty:
                acc_pct = float(pd.to_numeric(row["acc_test"], errors="coerce").iloc[0])

        # ROI por partido y Beneficio base (sobre partidos jugados)
        roi_por_partido = float("nan")
        beneficio_base = float("nan")
        if "net_profit" in df.columns and n_played > 0:
            net = pd.to_numeric(df.loc[played_mask, "net_profit"], errors="coerce").fillna(0.0)
            beneficio_base = float(net.sum())
            roi_por_partido = float(net.sum() / n_played)

        # ROI (agregado de temporada desde roi_by_season_{model})
        roi_model_temp = None
        roi_by_season = load_roi_by_season(model)
        if not roi_by_season.empty:
            season_col_r = next((c for c in ["Season","test_season","season"] if c in roi_by_season.columns), None)
            if season_col_r:
                row = roi_by_season[pd.to_numeric(roi_by_season[season_col_r], errors="coerce") == pd.to_numeric(cur_season)]
                if not row.empty:
                    roi_col = "roi" if "roi" in row.columns else next((c for c in row.columns if str(c).lower().startswith("roi")), None)
                    if roi_col:
                        roi_model_temp = float(pd.to_numeric(row[roi_col], errors="coerce").iloc[0])

        # Beneficio escalado por STAKE (€)
        beneficio = float("nan")
        if not np.isnan(beneficio_base):
            beneficio = beneficio_base * float(stake)

        # KPIs — orden: Partidos, Acierto, ROI, ROI por partido; Beneficio € bajo ROI
        k1, k2, k3, k4 = st.columns(4)

        # Col 1: Partidos disputados
        k1.metric("Partidos disputados", f"{n_played}")

        # Col 2: Acierto (métrico) + Y/X debajo, un poco más grande y más negro
        if not np.isnan(acc_pct):
            k2.metric("Acierto", f"{acc_pct:.1%}")
        else:
            hit_rate_fb = float(corr_series[played_mask].mean()) if n_played > 0 else float("nan")
            k2.metric("Acierto", f"{hit_rate_fb:.1%}" if not np.isnan(hit_rate_fb) else "—")
        with k2:
            if n_played > 0:
                st.markdown(
                    f"<div style='margin-top:0.10rem;font-size:1.05rem;font-weight:600;'>"
                    f"{n_hits}/{n_played}</div>",
                    unsafe_allow_html=True
                )

        # Col 3: ROI (agregado) + Beneficio €
        if roi_model_temp is not None:
            k3.metric("ROI", f"{roi_model_temp:.2%}")
        else:
            k3.metric("ROI", "—")
        with k3:
            if not np.isnan(beneficio):
                st.markdown(
                    f"<div style='margin-top:0.25rem;font-size:1.05rem;color:var(--text-color);'>"
                    f"<strong>Beneficio</strong>: {_euros(beneficio)}</div>",
                    unsafe_allow_html=True
                )

        # Col 4: ROI por partido (a la derecha de ROI)
        if not np.isnan(roi_por_partido):
            k4.metric("ROI por partido", f"{roi_por_partido:.1%}")
        else:
            k4.metric("ROI por partido", "—")

    st.divider()

    # 2) Tabla por jornada (renombrada y con columnas en castellano)
    titulo_jornada = f"Jornada {int(jornada)}" if jornada is not None else "Jornada —"
    st.subheader(f"Partidos — {titulo_jornada}")

    dfj = df.copy()
    if not dfj.empty and jornada is not None:
        dfj = dfj[pd.to_numeric(dfj["Week"], errors="coerce").astype("Int64") == int(jornada)]

    # columnas base de la tabla (sin Week ni 'jornada', como pediste)
    cols_show = [c for c in [
        "Date",
        # "Week",        # <- eliminado
        # "jornada",     # <- eliminado
        "HomeTeam_norm","AwayTeam_norm",
        "Pred",
        "B365H","B365D","B365A",
        "Correct","net_profit"
    ] if c in dfj.columns]

    # renombrado a español (solo en la vista)
    rename_map = {
        "Date": "Fecha",
        "HomeTeam_norm": "Local",
        "AwayTeam_norm": "Visitante",
        "Pred": "Predicción",
        "B365H": "Bet365 H",
        "B365D": "Bet365 D",
        "B365A": "Bet365 A",
        "Correct": "Acierto",
        "net_profit": "Beneficio neto",
    }

    if not dfj.empty:
        view = dfj[cols_show].copy()

        # ESCALAR el beneficio neto por STAKE (solo visualización y CSV)
        if "net_profit" in view.columns:
            view["net_profit"] = pd.to_numeric(view["net_profit"], errors="coerce").fillna(0.0) * float(stake)

        dfj_vista = view.rename(columns=rename_map)

        st.dataframe(dfj_vista, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar jornada (CSV)",
            dfj_vista.to_csv(index=False).encode("utf-8"),
            file_name=f"matchlog_{model}_{cur_season}_J{jornada or 'all'}.csv"
        )

        # --- Resumen de métricas de la jornada (letra pequeña) ---
        corr_series_week = _extract_correct_series(dfj)
        wk_played_mask = corr_series_week.notna()
        wk_n_played = int(wk_played_mask.sum())
        wk_n_hits = int((corr_series_week == 1.0).sum()) if wk_n_played > 0 else 0
        wk_hit_rate = float(corr_series_week[wk_played_mask].mean()) if wk_n_played > 0 else float("nan")

        wk_roi_por_partido = float("nan")
        wk_beneficio_base = float("nan")
        if "net_profit" in dfj.columns and wk_n_played > 0:
            net_wk = pd.to_numeric(dfj.loc[wk_played_mask, "net_profit"], errors="coerce").fillna(0.0)
            wk_beneficio_base = float(net_wk.sum())
            wk_roi_por_partido = float(net_wk.sum() / wk_n_played)

        wk_beneficio = wk_beneficio_base * float(stake) if not np.isnan(wk_beneficio_base) else float("nan")
        # ROI jornada (porcentaje total de la jornada; con stake unitario coincide con ROI por partido)
        wk_roi_pct = (wk_beneficio_base / wk_n_played) if wk_n_played > 0 else float("nan")

        wk_hit_rate_txt = f"{wk_hit_rate:.1%}" if not np.isnan(wk_hit_rate) else "—"
        wk_roi_por_partido_txt = f"{wk_roi_por_partido:.1%}" if not np.isnan(wk_roi_por_partido) else "—"
        wk_roi_pct_txt = f"{wk_roi_pct:.1%}" if not np.isnan(wk_roi_pct) else "—"
        wk_beneficio_txt = _euros(wk_beneficio) if not np.isnan(wk_beneficio) else "—"

        st.markdown(
            f"""
            <div style="margin-top:.5rem; font-size:0.95rem;">
              <strong>Resumen jornada</strong> — 
              Partidos: <strong>{wk_n_played}</strong> · 
              Aciertos: <strong>{wk_n_hits}/{wk_n_played}</strong> ({wk_hit_rate_txt}) · 
              ROI jornada: <strong>{wk_roi_pct_txt}</strong> · 
              ROI por partido: <strong>{wk_roi_por_partido_txt}</strong> · 
              Beneficio: <strong>{wk_beneficio_txt}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.info("No hay filas para esa jornada.")

    st.divider()

    # 3) Trayectoria de beneficio (modelo & Bet365) — temporada actual
    st.subheader(f"Trayectoria de beneficio (modelo & Bet365) — {cur_season}")
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
                d.insert(0, "x", range(1, len(d) + 1))
                x_col = "x"

        # mostrar solo serie del modelo + Bet365 (si existe)
        keep = [c for c in d.columns if c.lower().find(model) >= 0 or c.lower().find("bet365") >= 0 or c == x_col]
        if len(keep) <= 1:
            keep = [x_col] + [c for c in d.columns if c != x_col]
        d = d[keep].copy()

        if jornada is not None and len(d) >= int(jornada):
            d = d.iloc[:int(jornada)]

        long = d.melt(id_vars=x_col, var_name="Serie", value_name="Beneficio")
        fig = px.line(long, x=x_col, y="Beneficio", color="Serie")
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
        fig.update_xaxes(title_text="Partidos (acumulado)")
        fig.update_yaxes(title_text="Beneficio (stake=1)")
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
