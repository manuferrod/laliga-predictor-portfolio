# Home.py — LaLiga 1X2 · 25/26 (alineado con los NUEVOS outputs)
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

st.set_page_config(page_title="LaLiga 1X2 · 25/26", page_icon="⚽", layout="wide")
st.title("LaLiga 1X2 · 25/26")

# ================== I/O ==================
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _list_matchlog_seasons() -> list[int]:
    seasons = set()
    for p in OUT.glob("matchlogs_*.csv"):
        m = re.match(r"matchlogs_(\d{4})\.csv$", p.name)
        if m:
            seasons.add(int(m.group(1)))
    return sorted(seasons)

# ================== Normalización mínima ==================
def _ensure_matchday(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "Matchday" not in df.columns:
        # si no existiera, deriva 1..n por fecha
        if "Date" in df.columns:
            d = df.copy()
            d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
            d = d.sort_values("Date").reset_index(drop=True)
            d["Matchday"] = np.arange(1, len(d) + 1)
            return d
        d = df.copy().reset_index(drop=True)
        d["Matchday"] = np.arange(1, len(d) + 1)
        return d
    return df

def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def _correct_series(df: pd.DataFrame) -> pd.Series:
    # tus matchlogs traen 'correct' (0/1 o boolean)
    if "correct" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float")
    s = df["correct"]
    if s.dtype == bool:
        return s.astype(float)
    return pd.to_numeric(s, errors="coerce").where(lambda x: x.isin([0, 1]), np.nan).astype("float")

def _profit_cols(df: pd.DataFrame) -> tuple[str|None, str|None]:
    # (profit_por_partido, cum_profit_temporada)
    p = "profit" if "profit" in df.columns else None
    c = "cum_profit_season" if "cum_profit_season" in df.columns else None
    return p, c

def _beneficio_acum_por_jornada(df: pd.DataFrame, stake: float = 1.0) -> pd.DataFrame:
    """Devuelve DF: Jornada | Beneficio (acumulado) usando profit/cum_profit_season agrupado por Matchday."""
    if df.empty:
        return pd.DataFrame(columns=["Jornada", "Beneficio"])
    d = _ensure_matchday(_ensure_date(df.copy()))
    p_col, c_col = _profit_cols(d)

    if c_col:
        # tomar el valor al final de cada Matchday
        d = d.sort_values(["Matchday", "Date"])
        last = d.groupby(pd.to_numeric(d["Matchday"], errors="coerce").astype("Int64"))[c_col].last().dropna()
        last.index = last.index.astype(int)
        out = (pd.to_numeric(last, errors="coerce") * float(stake)).reset_index()
        out.columns = ["Jornada", "Beneficio"]
        return out

    if p_col:
        d[p_col] = pd.to_numeric(d[p_col], errors="coerce").fillna(0.0)
        by_w = d.groupby(pd.to_numeric(d["Matchday"], errors="coerce").astype("Int64"))[p_col].sum().dropna()
        by_w.index = by_w.index.astype(int)
        acum = by_w.sort_index().cumsum() * float(stake)
        return pd.DataFrame({"Jornada": acum.index.values, "Beneficio": acum.values})

    return pd.DataFrame(columns=["Jornada", "Beneficio"])

def _euros(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    sign = "-" if x < 0 else ""
    return f"{sign}{abs(x):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")

# ================== Arranque ==================
if not OUT.exists() or not any(OUT.iterdir()):
    st.warning("No se encontraron artefactos en `outputs/`. Sube/sincroniza y recarga.")
    st.stop()

seasons = _list_matchlog_seasons()
if not seasons:
    st.warning("No hay `matchlogs_YYYY.csv` en `outputs/`.")
    st.stop()

cur_season = seasons[-1]

# ========= Filtros =========
st.subheader("Filtros")
colf1, colf2 = st.columns([1, 1])

df_tmp = _ensure_matchday(_ensure_date(_read_csv(OUT / f"matchlogs_{cur_season}.csv")))
weeks = pd.to_numeric(df_tmp.get("Matchday"), errors="coerce").dropna().astype(int) if not df_tmp.empty else pd.Series(dtype=int)
jornada_opts = sorted(weeks.unique().tolist())
default_idx = (len(jornada_opts) - 1) if jornada_opts else 0

with colf1:
    jornada = st.selectbox("Jornada", jornada_opts if jornada_opts else [1], index=default_idx)

with colf2:
    stake = st.selectbox("STAKE", options=list(range(1, 11)), index=0,
                         help="€ por apuesta. Afecta solo al Beneficio (ROI no cambia).")

tab_public, tab_private = st.tabs(["📊 Temporada actual", "🔒 Zona privada (próxima jornada)"])

# ================== TAB PÚBLICA ==================
with tab_public:
    st.caption(f"Temporada: **{cur_season}**")

    # 1) KPIs principales
    df = _ensure_matchday(_ensure_date(_read_csv(OUT / f"matchlogs_{cur_season}.csv")))
    if df.empty:
        st.warning(f"No hay matchlogs para {cur_season}.")
    else:
        corr = _correct_series(df)
        played_mask = corr.notna()
        n_played = int(played_mask.sum())
        n_hits = int((corr == 1.0).sum()) if n_played > 0 else 0

        # accuracy y roi desde metrics_main_by_season.csv
        metrics = _read_csv(OUT / "metrics_main_by_season.csv")
        acc_pct = float("nan")
        roi_temp = float("nan")
        if not metrics.empty:
            row = metrics[pd.to_numeric(metrics["Season"], errors="coerce") == pd.to_numeric(cur_season)]
            if not row.empty:
                acc_pct = float(pd.to_numeric(row["accuracy"], errors="coerce").iloc[0])
                roi_temp = float(pd.to_numeric(row["roi"], errors="coerce").iloc[0])

        # ROI por partido y beneficio total desde matchlogs (profit o cum acumulado final)
        p_col, c_col = _profit_cols(df)
        beneficio_base = float("nan")
        roi_por_partido = float("nan")
        if n_played > 0:
            if p_col:
                net = pd.to_numeric(df.loc[played_mask, p_col], errors="coerce").fillna(0.0)
                beneficio_base = float(net.sum())
                roi_por_partido = float(net.sum() / n_played)
            elif c_col:
                total = float(pd.to_numeric(df[c_col], errors="coerce").fillna(0.0).iloc[-1])
                beneficio_base = total
                roi_por_partido = total / n_played

        beneficio = beneficio_base * float(stake) if not np.isnan(beneficio_base) else float("nan")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Partidos disputados", f"{n_played}")
        k2.metric("Acierto", f"{acc_pct:.1%}" if not np.isnan(acc_pct) else ("—" if n_played == 0 else f"{float(corr[played_mask].mean()):.1%}"))
        with k2:
            if n_played > 0:
                st.markdown(f"<div style='margin-top:0.10rem;font-size:1.05rem;font-weight:600;'>{n_hits}/{n_played}</div>", unsafe_allow_html=True)
        k3.metric("ROI", f"{roi_temp:.2%}" if not np.isnan(roi_temp) else "—")
        with k3:
            if not np.isnan(beneficio):
                st.markdown(f"<div style='margin-top:0.25rem;font-size:1.05rem;'><strong>Beneficio</strong>: {_euros(beneficio)}</div>", unsafe_allow_html=True)
        k4.metric("ROI por partido", f"{roi_por_partido:.1%}" if not np.isnan(roi_por_partido) else "—")

    st.divider()

    # 2) Tabla por jornada
    st.subheader(f"Partidos — Jornada {int(jornada)}")
    dfj = df.copy()
    if not dfj.empty:
        dfj = dfj[pd.to_numeric(dfj["Matchday"], errors="coerce").astype("Int64") == int(jornada)]

        cols_show = [
            c for c in [
                "Date","Matchday",
                "HomeTeam_norm","AwayTeam_norm",
                "pred_key","y_true","y_pred",
                "proba_H","proba_D","proba_A",
                "B365H","B365D","B365A",
                "correct","profit"
            ] if c in dfj.columns
        ]

        view = dfj[cols_show].copy()
        # escalar beneficio por stake
        if "profit" in view.columns:
            view["profit"] = pd.to_numeric(view["profit"], errors="coerce").fillna(0.0) * float(stake)

        rename_map = {
            "Date": "Fecha",
            "Matchday": "Jornada",
            "HomeTeam_norm": "Local",
            "AwayTeam_norm": "Visitante",
            "pred_key": "Pred clave",
            "y_true": "Resultado real",
            "y_pred": "Predicción",
            "proba_H": "p(H)",
            "proba_D": "p(D)",
            "proba_A": "p(A)",
            "B365H": "Bet365 H",
            "B365D": "Bet365 D",
            "B365A": "Bet365 A",
            "correct": "Acierto",
            "profit": "Beneficio neto",
        }
        dfj_vista = view.rename(columns=rename_map)

        st.dataframe(dfj_vista, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar jornada (CSV)",
            dfj_vista.to_csv(index=False).encode("utf-8"),
            file_name=f"matchlog_{cur_season}_J{jornada}.csv"
        )

        # Resumen rápido de la jornada
        corr_w = _correct_series(dfj)
        wk_mask = corr_w.notna()
        wk_n_played = int(wk_mask.sum())
        wk_n_hits = int((corr_w == 1.0).sum()) if wk_n_played > 0 else 0
        wk_hit_rate = float(corr_w[wk_mask].mean()) if wk_n_played > 0 else float("nan")
        wk_beneficio_base = float(pd.to_numeric(dfj.loc[wk_mask, "profit"], errors="coerce").fillna(0.0).sum()) if "profit" in dfj.columns and wk_n_played > 0 else float("nan")
        wk_beneficio = wk_beneficio_base * float(stake) if not np.isnan(wk_beneficio_base) else float("nan")
        wk_roi = (wk_beneficio_base / wk_n_played) if wk_n_played > 0 else float("nan")

        st.markdown(
            f"""
            <div style="margin-top:.5rem; font-size:0.95rem;">
              <strong>Resumen jornada</strong> — 
              Partidos: <strong>{wk_n_played}</strong> · 
              Aciertos: <strong>{wk_n_hits}/{wk_n_played}</strong> ({f"{wk_hit_rate:.1%}" if not np.isnan(wk_hit_rate) else "—"}) · 
              ROI: <strong>{f"{wk_roi:.1%}" if not np.isnan(wk_roi) else "—"}</strong> · 
              Beneficio: <strong>{_euros(wk_beneficio)}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("No hay filas para esa jornada.")

    st.divider()

    # 3) Trayectoria de beneficio (Modelo & Bet365) — acumulado por jornada
    st.subheader("Trayectoria de beneficio (Modelo & Bet365)")

    modelo_acum = _beneficio_acum_por_jornada(df, stake=stake)

    df_mkt = _ensure_matchday(_ensure_date(_read_csv(OUT / f"matchlogs_market_{cur_season}.csv")))
    if not df_mkt.empty:
        bet365_acum = _beneficio_acum_por_jornada(df_mkt, stake=stake)
    else:
        bet365_acum = pd.DataFrame(columns=["Jornada", "Beneficio"])

    plot_df = pd.DataFrame()
    if not modelo_acum.empty:
        a = modelo_acum.copy(); a["Serie"] = "Modelo"
        plot_df = pd.concat([plot_df, a], ignore_index=True)
    if not bet365_acum.empty:
        b = bet365_acum.copy(); b["Serie"] = "Bet365"
        plot_df = pd.concat([plot_df, b], ignore_index=True)

    if not plot_df.empty:
        if jornada is not None:
            plot_df = plot_df[plot_df["Jornada"] <= int(jornada)]
        fig = px.line(plot_df, x="Jornada", y="Beneficio", color="Serie")
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
        fig.update_xaxes(title_text="Jornadas")
        fig.update_yaxes(title_text=f"Beneficio (STAKE = {stake})")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Ver datos de la curva"):
            pivot = plot_df.pivot_table(index="Jornada", columns="Serie", values="Beneficio", aggfunc="last").reset_index()
            cols = ["Jornada"] + [c for c in ["Modelo", "Bet365"] if c in pivot.columns]
            st.dataframe(pivot[cols], use_container_width=True, hide_index=True)
    else:
        st.info("No pude construir la trayectoria acumulada por jornada (faltan datos).")

# ================== TAB PRIVADA (próxima jornada) ==================
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
        # Nuevo formato: future_predictions_<YYYY>.csv
        year_tag = cur_season  # el año de inicio de temporada
        dfp = _read_csv(OUT / f"future_predictions_{year_tag}.csv")

        if dfp.empty:
            st.info("No hay predicciones futuras disponibles todavía.")
        else:
            dfp = _ensure_date(_ensure_matchday(dfp))
            cols = [c for c in [
                "Date","Matchday",
                "HomeTeam_norm","AwayTeam_norm",
                "pred_key","y_pred",
                "pH_pred","pD_pred","pA_pred",
                "conf_maxprob","entropy","margin_top12",
                "B365H","B365D","B365A"  # por si decides incluir cuotas aquí en el futuro
            ] if c in dfp.columns]

            st.subheader(f"Predicciones (privadas) · {cur_season}")
            st.dataframe(dfp[cols] if cols else dfp, use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar predicciones (CSV)",
                (dfp[cols] if cols else dfp).to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_{cur_season}_proxima.csv"
            )
