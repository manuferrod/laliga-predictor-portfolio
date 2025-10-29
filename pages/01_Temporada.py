# Home.py — LaLiga 1X2 · 25/26 (solo jornadas COMPLETADAS en público)
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

st.set_page_config(page_title="LaLiga 1X2 · 25/26", page_icon="⚽", layout="wide")
st.title("LaLiga 1X2 · 25/26")

# ============= I/O y utilidades =============
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

def _ensure_matchday(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "Matchday" not in df.columns:
        d = df.copy().reset_index(drop=True)
        if "Date" in d.columns:
            d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
            d = d.sort_values("Date").reset_index(drop=True)
        d["Matchday"] = np.arange(1, len(d) + 1)
        return d
    return df

def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def _correct_series(df: pd.DataFrame) -> pd.Series:
    if "correct" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float")
    s = df["correct"]
    if s.dtype == bool:
        return s.astype(float)
    return pd.to_numeric(s, errors="coerce").where(lambda x: x.isin([0, 1]), np.nan).astype("float")

def _profit_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    p = "profit" if "profit" in df.columns else None
    c = "cum_profit_season" if "cum_profit_season" in df.columns else None
    return p, c

def _ytrue_played_mask(df: pd.DataFrame) -> pd.Series:
    """
    'Jugado' estricto: y_true en {H,D,A}. Excluye 0, vacío o NaN.
    """
    if "y_true" not in df.columns:
        return pd.Series(False, index=df.index)
    t = df["y_true"].astype(str).str.strip().str.upper()
    return t.isin({"H", "D", "A"})

def _complete_matchdays(df: pd.DataFrame) -> list[int]:
    """
    Jornadas COMPLETADAS: todas sus filas tienen y_true en {H,D,A}.
    """
    if df.empty or "Matchday" not in df.columns:
        return []
    grp = df.groupby("Matchday")
    n_total = grp.size()
    n_played = grp.apply(lambda g: _ytrue_played_mask(g).sum())
    complete = n_played[n_played == n_total].index.astype(int).tolist()
    return sorted(complete)

def _beneficio_acum_por_jornada(df: pd.DataFrame, stake: float = 1.0) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Jornada", "Beneficio"])
    d = _ensure_matchday(_ensure_date(df.copy()))
    p_col, c_col = _profit_cols(d)
    if c_col:
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

def _season_label(start_year: int) -> str:
    """Convierte 2025 -> '25/26'."""
    a = start_year % 100
    b = (start_year + 1) % 100
    return f"{a:02d}/{b:02d}"

# ============= Arranque / comprobaciones =============
if not OUT.exists() or not any(OUT.iterdir()):
    st.warning("No se encontraron artefactos en `outputs/`. Sube/sincroniza y recarga.")
    st.stop()

seasons = _list_matchlog_seasons()
if not seasons:
    st.warning("No hay `matchlogs_YYYY.csv` en `outputs/`.")
    st.stop()

cur_season = seasons[-1]
season_lbl = _season_label(cur_season)

# ============= Filtros =============
st.subheader("Filtros")
colf1, colf2 = st.columns([1, 1])

# Cargamos matchlogs y determinamos jornadas COMPLETADAS
df_tmp = _ensure_matchday(_ensure_date(_read_csv(OUT / f"matchlogs_{cur_season}.csv")))
completed_matchdays = _complete_matchdays(df_tmp)

with colf1:
    if completed_matchdays:
        jornada = st.selectbox("Jornada", completed_matchdays, index=len(completed_matchdays) - 1)
    else:
        jornada = st.selectbox("Jornada", [1], index=0)

with colf2:
    stake = st.selectbox("STAKE", options=list(range(1, 11)), index=0,
                         help="€ por apuesta. Afecta solo al Beneficio (ROI no cambia).")

tab_public, tab_private = st.tabs(["📊 Temporada actual", "🔒 Zona privada (próxima jornada)"])

# ============= TAB PÚBLICA =============
with tab_public:
    st.caption(f"Temporada: **{cur_season}**")

    # KPIs solo con jornadas COMPLETADAS
    df = _ensure_matchday(_ensure_date(_read_csv(OUT / f"matchlogs_{cur_season}.csv")))
    if df.empty:
        st.warning(f"No hay matchlogs para {cur_season}.")
    else:
        df_public = df[df["Matchday"].isin(completed_matchdays)].copy() if completed_matchdays else df.iloc[0:0].copy()

        corr = _correct_series(df_public)
        n_played = len(df_public)
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

        # ROI por partido y beneficio total desde df_public
        p_col, c_col = _profit_cols(df_public)
        beneficio_base = float("nan")
        roi_por_partido = float("nan")
        if n_played > 0:
            if p_col:
                net = pd.to_numeric(df_public[p_col], errors="coerce").fillna(0.0)
                beneficio_base = float(net.sum())
                roi_por_partido = float(net.sum() / n_played)
            elif c_col:
                total = float(pd.to_numeric(df_public[c_col], errors="coerce").fillna(0.0).iloc[-1])
                beneficio_base = total
                roi_por_partido = total / n_played

        beneficio = beneficio_base * float(stake) if not np.isnan(beneficio_base) else float("nan")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Partidos disputados", f"{n_played}")
        if not np.isnan(acc_pct):
            k2.metric("Acierto", f"{acc_pct:.1%}")
        else:
            hit_rate_fb = float(corr.mean()) if n_played > 0 else float("nan")
            k2.metric("Acierto", f"{hit_rate_fb:.1%}" if not np.isnan(hit_rate_fb) else "—")
        with k2:
            if n_played > 0:
                st.markdown(
                    f"<div style='margin-top:0.10rem;font-size:1.05rem;font-weight:600;'>{n_hits}/{n_played}</div>",
                    unsafe_allow_html=True
                )
        k3.metric("ROI", f"{roi_temp:.2%}" if not np.isnan(roi_temp) else "—")
        with k3:
            if not np.isnan(beneficio):
                st.markdown(
                    f"<div style='margin-top:0.25rem;font-size:1.05rem;'><strong>Beneficio</strong>: {_euros(beneficio)}</div>",
                    unsafe_allow_html=True
                )
        k4.metric("ROI por partido", f"{roi_por_partido:.1%}" if not np.isnan(roi_por_partido) else "—")

    st.divider()

    # Tabla por jornada (solo jornadas COMPLETADAS)
    st.subheader(f"Partidos — Jornada {int(jornada)}")
    dfj = df[df["Matchday"].isin(completed_matchdays)].copy()
    dfj = dfj[pd.to_numeric(dfj["Matchday"], errors="coerce").astype("Int64") == int(jornada)]
    if dfj.empty:
        st.info("Esa jornada todavía no se ha disputado. Las predicciones se muestran solo en la zona privada.")
    else:
        # columnas visibles (sin 'pred_key')
        cols_show = [
            c for c in [
                "Date","Matchday",
                "HomeTeam_norm","AwayTeam_norm",
                # "pred_key",   # ← oculto
                "y_true","y_pred",
                "proba_H","proba_D","proba_A",
                "B365H","B365D","B365A",
                "correct","profit"
            ] if c in dfj.columns
        ]
        view = dfj[cols_show].copy()

        # Formato fecha sin hora
        if "Date" in view.columns:
            view["Date"] = pd.to_datetime(view["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

        # Escalar beneficio por stake
        if "profit" in view.columns:
            view["profit"] = pd.to_numeric(view["profit"], errors="coerce").fillna(0.0) * float(stake)

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

        # Resumen de la jornada (tamaño de letra ↑)
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
            <div style="margin-top:.5rem; font-size:1.10rem;">
              <strong>Resumen jornada</strong> — 
              Partidos: <strong>{wk_n_played}</strong> · 
              Aciertos: <strong>{wk_n_hits}/{wk_n_played}</strong> ({f"{wk_hit_rate:.1%}" if not np.isnan(wk_hit_rate) else "—"}) · 
              ROI: <strong>{f"{wk_roi:.1%}" if not np.isnan(wk_roi) else "—"}</strong> · 
              Beneficio: <strong>{_euros(wk_beneficio)}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # Trayectoria de beneficio (Modelo & Bet365) — SOLO jornadas COMPLETADAS
    st.subheader("Trayectoria de beneficio (Modelo & Bet365)")

    df_public = df[df["Matchday"].isin(completed_matchdays)].copy()
    modelo_acum = _beneficio_acum_por_jornada(df_public, stake=stake)

    df_mkt_all = _ensure_matchday(_ensure_date(_read_csv(OUT / f"matchlogs_market_{cur_season}.csv")))
    if not df_mkt_all.empty:
        completed_mkt = _complete_matchdays(df_mkt_all)
        df_mkt_public = df_mkt_all[df_mkt_all["Matchday"].isin(completed_mkt)].copy()
        bet365_acum = _beneficio_acum_por_jornada(df_mkt_public, stake=stake)
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

# ============= TAB PRIVADA (próxima jornada) =============
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

    # === Logos: helpers y ruta local (ajusta si usas otra carpeta) ===
    LOGOS_DIR = ROOT / "pages" / "logos"

    def _team_logo_path(team: str) -> Path | None:
        """Devuelve la ruta al escudo (busca png/jpg/jpeg/webp)."""
        if not team:
            return None
        team_norm = (
            str(team)
            .lower()
            .replace(" ", "_")
            .replace(".", "")
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("ñ", "n")
        )
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = LOGOS_DIR / f"{team_norm}{ext}"
            if p.exists():
                return p
        return None

    def _logo_html(team: str, size: int = 80) -> str:
        """Devuelve <img> embebido base64 para mostrar el logo."""
        path = _team_logo_path(team)
        if not path:
            return f"<div style='width:{size}px;height:{size}px;display:inline-block;'></div>"
        import base64
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return f"<img src='data:{mime};base64,{data}' width='{size}' height='{size}' style='object-fit:contain;'/>"

    # === Colores por equipo (soporta franjas bicolores) + helpers ===
    TEAM_COLORS = {
        # usa 1 o 2 tonos por equipo
        "real_madrid": ["#FEBE10"],
        "barcelona": ["#A50044", "#004D98"],           # grana + azul
        "atletico_madrid": ["#D1002D", "#1D3A94"],     # rojo + azul
        "athletic_bilbao": ["#E2231A", "#FFFFFF"],     # rojo + blanco
        "sevilla": ["#D50032", "#FFFFFF"],
        "real_betis": ["#009150", "#FFFFFF"],
        "valencia": ["#F49F1C", "#000000"],
        "villarreal": ["#F2E600"],
        "real_sociedad": ["#0056A6", "#FFFFFF"],
        "celta": ["#7EB7E6"],
        "girona": ["#D50032", "#FFFFFF"],
        "osasuna": ["#002D62", "#E5002D"],
        "mallorca": ["#C8102E", "#000000"],
        "rayo_vallecano": ["#FFFFFF", "#D50032"],
        "getafe": ["#0059B3"],
        "alaves": ["#003D8F", "#FFFFFF"],
        "las_palmas": ["#FFDD00", "#1B75BB"],
        "leganes": ["#2CA6E0", "#FFFFFF"],
        "levante": ["#1E2A78", "#A50044"],
        "granada": ["#D50032", "#FFFFFF"],
        # añade más si los necesitas
    }

    def _norm_team_key(name: str) -> str:
        return (
            (name or "")
            .lower()
            .replace(" ", "_")
            .replace(".", "")
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("ñ", "n")
        )

    def team_colors(name: str, fallback: list[str] = ["#1f77b4"]) -> list[str]:
        """Devuelve lista de colores del equipo (1 o 2 tonos)."""
        return TEAM_COLORS.get(_norm_team_key(name), fallback)

    def blend_color(c1: str, c2: str, ratio: float = 0.5) -> str:
        """Mezcla dos colores hex a uno intermedio (para fill)."""
        h1, h2 = c1.lstrip("#"), c2.lstrip("#")
        r = round(int(h1[0:2], 16) * (1-ratio) + int(h2[0:2], 16) * ratio)
        g = round(int(h1[2:4], 16) * (1-ratio) + int(h2[2:4], 16) * ratio)
        b = round(int(h1[4:6], 16) * (1-ratio) + int(h2[4:6], 16) * ratio)
        return f"#{r:02X}{g:02X}{b:02X}"

    def hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
        """Convierte #RRGGBB → rgba(r,g,b,a)."""
        h = hex_color.lstrip("#")
        r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    if ok:
        year_tag = cur_season  # año de inicio de la temporada
        dfp = _read_csv(OUT / f"future_predictions_{year_tag}.csv")

        if dfp.empty:
            st.info("No hay predicciones futuras disponibles todavía.")
        else:
            dfp = _ensure_date(_ensure_matchday(dfp))

            # Determinar la(s) jornada(s) en el fichero; usamos la moda (valor más frecuente)
            md_series = pd.to_numeric(dfp.get("Matchday"), errors="coerce")
            jornada_priv = int(md_series.mode().iloc[0]) if md_series.notna().any() else "-"
            st.subheader(f"Predicciones jornada {jornada_priv} · {season_lbl}")

            # Construir vista sin horas y con renombrados, ocultando pred_key
            cols = [c for c in [
                "Date","Matchday",
                "HomeTeam_norm","AwayTeam_norm",
                # "pred_key",   # ← oculto
                "y_pred",
                "pH_pred","pD_pred","pA_pred",
                "conf_maxprob","entropy","margin_top12"
            ] if c in dfp.columns]

            viewp = dfp[cols].copy()
            if "Date" in viewp.columns:
                viewp["Date"] = pd.to_datetime(viewp["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

            rename_map_p = {
                "Date": "Fecha",
                "Matchday": "Jornada",
                "HomeTeam_norm": "Local",
                "AwayTeam_norm": "Visitante",
                "y_pred": "Predicción",
                "pH_pred": "p(H)",
                "pD_pred": "p(D)",
                "pA_pred": "p(A)",
            }
            viewp = viewp.rename(columns=rename_map_p)

            st.dataframe(viewp, use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar predicciones (CSV)",
                viewp.to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_{cur_season}_J{jornada_priv}.csv"
            )

        # ===================== RADAR + BARRAS (zona privada) =====================
        st.divider()
        st.subheader("Perfil del partido (Radar + Barras)")

        # --- Cargar CSV del radar prematch de la temporada actual ---
        radar_csv = OUT / "radar_prematch" / f"radar_prematch_{cur_season}.csv"
        radar_df = _read_csv(radar_csv)
        if radar_df.empty:
            st.info("No encuentro el CSV de radar prematch para esta temporada.")
        else:
            # Normalizar fecha para emparejar
            if "Date" in radar_df.columns:
                radar_df["Date"] = pd.to_datetime(radar_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

            # Selector de partido (por defecto, el primero de la tabla)
            opciones = []
            for _, rsel in viewp.iterrows():
                opciones.append(f"{rsel.get('Fecha','?')} — {rsel['Local']} vs {rsel['Visitante']}")
            if not opciones:
                st.info("No hay filas en la tabla de predicciones para seleccionar un partido.")
            else:
                pick = st.selectbox("Selecciona partido", opciones, index=0)

                # Parse selección
                m = re.match(r"(.+?)\s+—\s+(.+?)\s+vs\s+(.+)$", pick)
                sel_date, sel_home, sel_away = None, None, None
                if m:
                    sel_date = m.group(1).strip()
                    sel_home = m.group(2).strip()
                    sel_away = m.group(3).strip()

                # Buscar fila correspondiente en radar_df
                cand = radar_df.copy()
                # renombre por seguridad (incluimos cuotas y overround)
                ren = {}
                for c in ["HomeTeam_norm","AwayTeam_norm","Season","Date","B365H","B365D","B365A","overround"]:
                    for cc in [c, c.lower(), c.upper()]:
                        if cc in cand.columns:
                            ren[cc] = c
                cand = cand.rename(columns=ren)
                # coerción tipos clave
                if "Season" in cand.columns:
                    cand["Season"] = pd.to_numeric(cand["Season"], errors="coerce").astype("Int64")
                if "Date" in cand.columns:
                    cand["Date"] = pd.to_datetime(cand["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

                mask = (pd.to_numeric(cand.get("Season"), errors="coerce") == pd.to_numeric(cur_season)) \
                       & (cand.get("HomeTeam_norm","") == sel_home) \
                       & (cand.get("AwayTeam_norm","") == sel_away)
                if sel_date and "Date" in cand.columns:
                    mask = mask & (cand["Date"] == sel_date)

                row = cand.loc[mask].copy()
                if row.empty:
                    # fallback (sin fecha exacta): primera coincidencia por equipos
                    mask2 = (cand.get("HomeTeam_norm","") == sel_home) & (cand.get("AwayTeam_norm","") == sel_away)
                    row = cand.loc[mask2].sort_values("Date").tail(1).copy()

                if row.empty:
                    st.warning("No encontré métricas de radar para ese partido.")
                else:
                    r = row.iloc[0].to_dict()

                    # ===== Encabezado con escudos + cuotas (tipografía algo mayor) =====
                    def _fmt_odds(x):
                        try:
                            return f"{float(x):.2f}"
                        except Exception:
                            return "—"

                    h_odds = _fmt_odds(r.get("B365H"))
                    d_odds = _fmt_odds(r.get("B365D"))
                    a_odds = _fmt_odds(r.get("B365A"))

                    # Tamaños tipográficos y alturas
                    LOGO_SIZE = 84
                    NAME_FS   = "1.20rem"
                    ODDS_FS   = "1.05rem"
                    VS_FS     = "2.00rem"
                    # Altura del bloque superior (logo + nombre) para alinear la 'Cuota Empate'
                    SPACER_PX = LOGO_SIZE - 30  # ajusta si necesitas

                    c_logo1, c_vs, c_logo2 = st.columns([1, 0.3, 1])
                    with c_logo1:
                        st.markdown(
                            f"""
                            <div style='text-align:center;'>
                                {_logo_html(sel_home, LOGO_SIZE)}
                                <div style='font-weight:700;margin-top:6px;font-size:{NAME_FS}'>{sel_home}</div>
                                <div style='margin-top:4px;color:#666;font-size:{ODDS_FS}'>Cuota Local: {h_odds}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with c_vs:
                        st.markdown(
                            f"""
                            <div style='text-align:center;'>
                                <div style='font-size:{VS_FS};font-weight:800;margin-top:18px;'>VS</div>
                                <div style='height:{SPACER_PX}px;'></div>
                                <div style='margin-top:4px;color:#666;font-size:{ODDS_FS}'>Cuota Empate: {d_odds}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with c_logo2:
                        st.markdown(
                            f"""
                            <div style='text-align:center;'>
                                {_logo_html(sel_away, LOGO_SIZE)}
                                <div style='font-weight:700;margin-top:6px;font-size:{NAME_FS}'>{sel_away}</div>
                                <div style='margin-top:4px;color:#666;font-size:{ODDS_FS}'>Cuota Visitante: {a_odds}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Paletas por equipo
                    home_palette = team_colors(sel_home)
                    away_palette = team_colors(sel_away)

                    home_col = home_palette[0]
                    away_col = away_palette[0]
                    # Relleno del radar: si hay 2 tonos, usa mezcla; si no, usa el principal
                    home_fill = hex_to_rgba(blend_color(*home_palette, 0.5), 0.25) if len(home_palette) > 1 else hex_to_rgba(home_col, 0.25)
                    away_fill = hex_to_rgba(blend_color(*away_palette, 0.5), 0.25) if len(away_palette) > 1 else hex_to_rgba(away_col, 0.25)

                    # ---------------- RADAR ----------------
                    import plotly.graph_objects as go

                    radar_axes = [
                        ("xG (7)", "home_avg_xg_last7_norm", "away_avg_xg_last7_norm",
                         "home_avg_xg_last7", "away_avg_xg_last7"),
                        ("On Target (7)", "home_avg_shotsontarget_last7_norm", "away_avg_shotsontarget_last7_norm",
                         "home_avg_shotsontarget_last7", "away_avg_shotsontarget_last7"),
                        ("Corners (7)", "home_avg_corners_last7_norm", "away_avg_corners_last7_norm",
                         "home_avg_corners_last7", "away_avg_corners_last7"),
                        ("Efectividad", "home_effectiveness_norm", "away_effectiveness_norm",
                         "home_effectiveness", "away_effectiveness"),
                        ("Forma Pts (6)", "home_form_points_6_norm", "away_form_points_6_norm",
                         "home_form_points_6", "away_form_points_6"),
                        ("Forma GD (6)", "home_form_gd_6_norm", "away_form_gd_6_norm",
                         "home_form_gd_6", "away_form_gd_6"),
                        ("Elo", "h_elo_norm", "a_elo_norm", "h_elo", "a_elo"),
                        ("Rend. relativo", "home_relative_perf_norm", "away_relative_perf_norm",
                         "home_relative_perf", "away_relative_perf"),
                    ]

                    # Si alguna *_norm no existe, la normalizamos on-the-fly con rangos fijos
                    ranges = {
                        "home_avg_xg_last7": (0, 4), "away_avg_xg_last7": (0, 4),
                        "home_avg_shotsontarget_last7": (0, 12), "away_avg_shotsontarget_last7": (0, 12),
                        "home_avg_corners_last7": (0, 12), "away_avg_corners_last7": (0, 12),
                        "home_effectiveness": (0, 1), "away_effectiveness": (0, 1),
                        "home_form_points_6": (0, 18), "away_form_points_6": (0, 18),
                        "home_form_gd_6": (-10, 10), "away_form_gd_6": (-10, 10),
                        "h_elo": (1450, 2150), "a_elo": (1450, 2150),
                        "home_relative_perf": (0, 2), "away_relative_perf": (0, 2),
                    }
                    def norm_val(raw_col, v):
                        lo, hi = ranges[raw_col]
                        if v is None or pd.isna(v):
                            return np.nan
                        return float(np.clip((float(v)-lo) / (hi-lo+1e-12), 0, 1))

                    thetas, home_vals, away_vals, hover_home, hover_away = [], [], [], [], []
                    for label, h_norm, a_norm, h_raw, a_raw in radar_axes:
                        thetas.append(label)
                        hv = r.get(h_norm)
                        av = r.get(a_norm)
                        if hv is None and h_raw in r:
                            hv = norm_val(h_raw, r.get(h_raw))
                        if av is None and a_raw in r:
                            av = norm_val(a_raw, r.get(a_raw))
                        home_vals.append(hv)
                        away_vals.append(av)
                        hover_home.append(f"{label}: {r.get(h_raw, '—')}")
                        hover_away.append(f"{label}: {r.get(a_raw, '—')}")

                    # cerrar el polígono
                    thetas_loop = thetas + [thetas[0]]
                    home_loop = home_vals + [home_vals[0]]
                    away_loop = away_vals + [away_vals[0]]

                    c1, c2 = st.columns([1,1])
                    with c1:
                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=home_loop, theta=thetas_loop, name=sel_home,
                            line=dict(color=home_col, width=3),
                            fill="toself", fillcolor=home_fill,
                            hovertext=hover_home+hover_home[:1]
                        ))
                        fig_radar.add_trace(go.Scatterpolar(
                            r=away_loop, theta=thetas_loop, name=sel_away,
                            line=dict(color=away_col, width=3),
                            fill="toself", fillcolor=away_fill,
                            hovertext=hover_away+hover_away[:1]
                        ))
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                            margin=dict(l=10, r=10, t=30, b=10), legend_title_text=""
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                    # ---------------- BARRAS (butterfly) ----------------
                    # Métricas para barras: usamos *_norm para longitud, y brutas para texto
                    bars_spec = [
                        # label, home_raw, away_raw, home_norm, away_norm
                        ("Puntos totales", "home_total_points_cum", "away_total_points_cum",
                         "home_total_points_cum_norm", "away_total_points_cum_norm"),
                        ("% puntos posibles", "home_points_pct", "away_points_pct",
                         "home_points_pct_norm", "away_points_pct_norm"),
                        ("Posición (inv.)", "home_prev_position", "away_prev_position",
                         "home_prev_position_norm", "away_prev_position_norm"),
                        ("GD acumulado", "home_gd_cum", "away_gd_cum",
                         "home_gd_cum_norm", "away_gd_cum_norm"),
                        ("Tiros (7)", "home_avg_shots_last7", "away_avg_shots_last7",
                         "home_avg_shots_last7_norm", "away_avg_shots_last7_norm"),
                        ("Corners (7)", "home_avg_corners_last7", "away_avg_corners_last7",
                         "home_avg_corners_last7_norm", "away_avg_corners_last7_norm"),
                        ("Faltas (7, inv.)", "home_avg_fouls_last7", "away_avg_fouls_last7",
                         "home_avg_fouls_last7_norm", "away_avg_fouls_last7_norm"),
                        ("Amarillas (7, inv.)", "home_avg_yellows_last7", "away_avg_yellows_last7",
                         "home_avg_yellows_last7_norm", "away_avg_yellows_last7_norm"),
                        ("Prob. implícita", "pimp1", "pimp2",
                         "pimp1_norm", "pimp2_norm"),
                    ]

                    rows = []
                    for label, hr, ar, hn, an in bars_spec:
                        hraw = r.get(hr); araw = r.get(ar)
                        hnm = r.get(hn);  anm = r.get(an)
                        def _try(v):
                            return np.nan if v is None or (isinstance(v, str) and v.strip() == "") else float(v)
                        rows.append({
                            "Métrica": label,
                            "Home_norm": -_try(hnm) if not pd.isna(_try(hnm)) else np.nan,  # negativo (izq)
                            "Away_norm":  _try(anm),
                            "Home_txt":  hraw,
                            "Away_txt":  araw,
                        })
                    bars_df = pd.DataFrame(rows)

                    with c2:
                        if bars_df.dropna(subset=["Home_norm","Away_norm"], how="all").empty:
                            st.info("No hay suficientes métricas normalizadas para construir las barras.")
                        else:
                            cats = bars_df["Métrica"].tolist()
                            home_x = bars_df["Home_norm"].fillna(0.0).tolist()
                            away_x = bars_df["Away_norm"].fillna(0.0).tolist()
                            home_text = [f"{sel_home}: {v if v is not None else '—'}" for v in bars_df["Home_txt"]]
                            away_text = [f"{sel_away}: {v if v is not None else '—'}" for v in bars_df["Away_txt"]]

                            # Patrón de rayas si hay 2 tonos
                            home_marker = dict(color=home_col)
                            away_marker = dict(color=away_col)
                            if len(home_palette) > 1:
                                home_marker["pattern"] = dict(shape="/", fgcolor=home_palette[1], solidity=0.4)
                            if len(away_palette) > 1:
                                away_marker["pattern"] = dict(shape="/", fgcolor=away_palette[1], solidity=0.4)

                            fig_bar = go.Figure()
                            fig_bar.add_bar(
                                x=home_x, y=cats, name=sel_home, orientation="h",
                                hovertext=home_text, hoverinfo="text",
                                marker=home_marker
                            )
                            fig_bar.add_bar(
                                x=away_x, y=cats, name=sel_away, orientation="h",
                                hovertext=away_text, hoverinfo="text",
                                marker=away_marker
                            )
                            # Eje X simétrico [-1,1]
                            fig_bar.update_layout(
                                barmode="relative",
                                xaxis=dict(range=[-1, 1], tickvals=[-1,-0.5,0,0.5,1],
                                           ticktext=["100%","","0","", "100%"]),
                                margin=dict(l=10, r=10, t=30, b=10), legend_title_text=""
                            )
                            fig_bar.update_yaxes(autorange="reversed")  # arriba la primera métrica
                            st.plotly_chart(fig_bar, use_container_width=True)
