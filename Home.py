# Home.py — LaLiga 1X2 · 25/26 (temporada actual con nuevos outputs)
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

# =============== Helpers de E/S y normalización ===============
def ensure_outputs_dir():
    OUT.mkdir(parents=True, exist_ok=True)

def has_outputs() -> bool:
    return OUT.exists() and any(OUT.iterdir())

def _list_matchlog_seasons() -> list[int]:
    seasons = set()
    for p in OUT.glob("matchlogs_*.csv"):
        m = re.match(r"matchlogs_(\d{4})\.csv$", p.name)
        if m:
            seasons.add(int(m.group(1)))
    return sorted(seasons)

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _ensure_week_col(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "Week" in df.columns:
        return df
    # compat: jornada/jornada_num/Wk/Matchweek
    for cand in ["jornada", "Matchweek", "Wk", "mw", "week"]:
        if cand in df.columns:
            return df.rename(columns={cand: "Week"})
    return df

def _coerce_date_col(df: pd.DataFrame) -> pd.Series:
    for c in ["Date", "date", "Fecha"]:
        if c in df.columns:
            return pd.to_datetime(df[c], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)

def _extract_correct_series(df: pd.DataFrame) -> pd.Series:
    """Mapea 'Correct' a {1,0,NaN}. Acepta ✓/✔/True/1 y ✗/✘/False/0."""
    if df.empty or "Correct" not in df.columns:
        return pd.Series(index=df.index, dtype="float")
    s = df["Correct"]
    t = s.astype(str).str.strip().str.lower()
    true_tokens  = {"✓", "✔", "true", "1", "si", "sí", "y", "acierto", "correct", "correcto"}
    false_tokens = {"✗", "✘", "false", "0", "no", "n", "fallo", "incorrect", "incorrecto"}
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[t.isin(true_tokens)]  = 1.0
    out[t.isin(false_tokens)] = 0.0
    if s.dtype == bool:
        return s.astype(float)
    num = pd.to_numeric(s, errors="coerce")
    return out.fillna(num.where(num.isin([0, 1]), np.nan).astype("float"))

def _euros(x: float) -> str:
    if np.isnan(x):
        return "—"
    sign = "-" if x < 0 else ""
    return f"{sign}{abs(x):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")

def _beneficio_acum_por_jornada(df: pd.DataFrame, stake: float = 1.0) -> pd.DataFrame:
    """Devuelve DataFrame: Jornada | Beneficio (acumulado hasta esa jornada)."""
    if df.empty or "net_profit" not in df.columns:
        return pd.DataFrame(columns=["Jornada", "Beneficio"])
    d = _ensure_week_col(df.copy())
    d["Date_dt"] = _coerce_date_col(d)
    d = d.sort_values(["Week", "Date_dt"])
    d["net_profit"] = pd.to_numeric(d["net_profit"], errors="coerce").fillna(0.0)
    by_w = d.groupby(pd.to_numeric(d["Week"], errors="coerce").astype("Int64"))["net_profit"].sum().dropna()
    by_w.index = by_w.index.astype(int)
    acum = by_w.sort_index().cumsum() * float(stake)
    return pd.DataFrame({"Jornada": acum.index.values, "Beneficio": acum.values})

def _load_metrics_main_by_season() -> pd.DataFrame:
    """Carga outputs/metrics_main_by_season.csv (acc_test, roi, etc.)."""
    return _read_csv(OUT / "metrics_main_by_season.csv")

def _season_col(df: pd.DataFrame) -> str | None:
    for c in ["Season", "season", "test_season", "temporada"]:
        if c in df.columns:
            return c
    return None

# =============== Comprobaciones iniciales ===============
ensure_outputs_dir()
if not has_outputs():
    st.warning("No se encontraron artefactos en `outputs/`. Sube/sincroniza y recarga.")
    st.stop()

seasons = _list_matchlog_seasons()
if not seasons:
    st.warning("No hay `matchlogs_YYYY.csv` en `outputs/`.")
    st.stop()

# Temporada actual (última disponible por convención)
cur_season = seasons[-1]

# ========= Filtros =========
st.subheader("Filtros")
colf1, colf2 = st.columns([1, 1])

with colf1:
    jornada_opts = []
    df_tmp = _ensure_week_col(_read_csv(OUT / f"matchlogs_{cur_season}.csv"))
    if not df_tmp.empty and "Week" in df_tmp.columns:
        jornada_opts = (
            pd.to_numeric(df_tmp["Week"], errors="coerce")
            .dropna().astype(int).sort_values().unique().tolist()
        )
    jornada = st.selectbox(
        "Jornada",
        jornada_opts if jornada_opts else [None],
        index=len(jornada_opts) - 1 if jornada_opts else 0
    )

with colf2:
    stake = st.selectbox(
        "STAKE",
        options=list(range(1, 11)),
        index=0,
        help="€ por apuesta. Afecta solo al Beneficio (ROI no cambia)."
    )

tab_public, tab_private = st.tabs(["📊 Temporada actual", "🔒 Zona privada (próxima jornada)"])

# =============== TAB PÚBLICA ===============
with tab_public:
    st.caption(f"Temporada: **{cur_season}**")

    # 1) KPIs de temporada (desde matchlogs + metrics_main_by_season.csv)
    df = _read_csv(OUT / f"matchlogs_{cur_season}.csv").copy()
    if df.empty:
        st.warning(f"No hay matchlogs para {cur_season}.")
    else:
        df = _ensure_week_col(df)
        df["Date_dt"] = _coerce_date_col(df)

        corr = _extract_correct_series(df)
        played_mask = corr.notna()
        n_played = int(played_mask.sum())
        n_hits = int((corr == 1.0).sum()) if n_played > 0 else 0

        # Acierto oficial (acc_test) y ROI desde metrics_main_by_season.csv
        acc_pct = float("nan")
        roi_temp = float("nan")
        m = _load_metrics_main_by_season()
        sc = _season_col(m)
        if not m.empty and sc:
            row = m[pd.to_numeric(m[sc], errors="coerce") == pd.to_numeric(cur_season)]
            if not row.empty:
                # acc_test
                acc_col = next((c for c in row.columns if str(c).lower() in ("acc_test","accuracy_test","acc")), None)
                if acc_col:
                    acc_pct = float(pd.to_numeric(row[acc_col], errors="coerce").iloc[0])
                # roi
                roi_col = next((c for c in row.columns if str(c).lower().startswith("roi")), None)
                if roi_col:
                    roi_temp = float(pd.to_numeric(row[roi_col], errors="coerce").iloc[0])

        # ROI por partido y beneficio total (desde matchlogs)
        roi_por_partido = float("nan")
        beneficio_base = float("nan")
        if "net_profit" in df.columns and n_played > 0:
            net = pd.to_numeric(df.loc[played_mask, "net_profit"], errors="coerce").fillna(0.0)
            beneficio_base = float(net.sum())
            roi_por_partido = float(net.sum() / n_played)

        beneficio = beneficio_base * float(stake) if not np.isnan(beneficio_base) else float("nan")

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Partidos disputados", f"{n_played}")

        if not np.isnan(acc_pct):
            k2.metric("Acierto", f"{acc_pct:.1%}")
        else:
            hit_rate_fb = float(corr[played_mask].mean()) if n_played > 0 else float("nan")
            k2.metric("Acierto", f"{hit_rate_fb:.1%}" if not np.isnan(hit_rate_fb) else "—")
        with k2:
            if n_played > 0:
                st.markdown(
                    f"<div style='margin-top:0.10rem;font-size:1.05rem;font-weight:600;'>"
                    f"{n_hits}/{n_played}</div>",
                    unsafe_allow_html=True
                )

        k3.metric("ROI", f"{roi_temp:.2%}" if not np.isnan(roi_temp) else "—")
        with k3:
            if not np.isnan(beneficio):
                st.markdown(
                    f"<div style='margin-top:0.25rem;font-size:1.05rem;color:var(--text-color);'>"
                    f"<strong>Beneficio</strong>: {_euros(beneficio)}</div>",
                    unsafe_allow_html=True
                )

        k4.metric("ROI por partido", f"{roi_por_partido:.1%}" if not np.isnan(roi_por_partido) else "—")

    st.divider()

    # 2) Tabla por jornada
    titulo_jornada = f"Jornada {int(jornada)}" if jornada is not None else "Jornada —"
    st.subheader(f"Partidos — {titulo_jornada}")

    dfj = df.copy()
    if not dfj.empty and jornada is not None:
        dfj = dfj[pd.to_numeric(dfj["Week"], errors="coerce").astype("Int64") == int(jornada)]

    cols_show = [c for c in [
        "Date",
        "HomeTeam_norm","AwayTeam_norm",
        "Pred",
        "B365H","B365D","B365A",
        "Correct","net_profit"
    ] if c in dfj.columns]

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
        if "net_profit" in view.columns:
            view["net_profit"] = pd.to_numeric(view["net_profit"], errors="coerce").fillna(0.0) * float(stake)
        dfj_vista = view.rename(columns=rename_map)
        st.dataframe(dfj_vista, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar jornada (CSV)",
            dfj_vista.to_csv(index=False).encode("utf-8"),
            file_name=f"matchlog_{cur_season}_J{jornada or 'all'}.csv"
        )

        # Resumen jornada
        corr_w = _extract_correct_series(dfj)
        wk_mask = corr_w.notna()
        wk_n_played = int(wk_mask.sum())
        wk_n_hits = int((corr_w == 1.0).sum()) if wk_n_played > 0 else 0
        wk_hit_rate = float(corr_w[wk_mask].mean()) if wk_n_played > 0 else float("nan")
        wk_beneficio_base = float(pd.to_numeric(dfj.loc[wk_mask, "net_profit"], errors="coerce").fillna(0.0).sum()) if "net_profit" in dfj.columns and wk_n_played > 0 else float("nan")
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

    bet365_df = _read_csv(OUT / f"matchlogs_market_{cur_season}.csv")
    if not bet365_df.empty:
        bet365_df = _ensure_week_col(bet365_df)
        bet365_df["Date_dt"] = _coerce_date_col(bet365_df)
        bet365_acum = _beneficio_acum_por_jornada(bet365_df, stake=stake)
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

# =============== TAB PRIVADA (próxima jornada) ===============
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
        # Nuevo formato: future_predictions_<YYYY>.csv / .json
        # Para 25/26 usamos <YYYY>=2025 (primer año de la temporada)
        year_tag = cur_season  # tus matchlogs_YYYY usan el año de inicio de la temporada
        dfp = _read_csv(OUT / f"future_predictions_{year_tag}.csv")

        if dfp.empty:
            st.info("No hay predicciones futuras disponibles todavía.")
        else:
            dfp = _ensure_week_col(dfp)
            cols = [c for c in [
                "Date","Week","jornada",
                "HomeTeam_norm","AwayTeam_norm",
                "Pred",
                "B365H","B365D","B365A"
            ] if c in dfp.columns]

            st.subheader(f"Predicciones (privadas) · {cur_season}")
            st.dataframe(dfp[cols] if cols else dfp, use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar predicciones (CSV)",
                (dfp[cols] if cols else dfp).to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_{cur_season}_proxima.csv"
            )
