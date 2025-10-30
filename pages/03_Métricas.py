# pages/03_Métricas.py — Métricas y ROI por temporada (nuevos outputs)
from __future__ import annotations

import sys, importlib.util
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

# ===== Paths / import defensivo (por si sigues usando scripts/io en otros sitios) =====
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
OUT = ROOT / "outputs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import scripts.io as io  # intento normal (no estrictamente necesario aquí)
except Exception:
    spec = importlib.util.spec_from_file_location("io", SCRIPTS_DIR / "io.py")
    io = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(io)  # type: ignore
    except Exception:
        io = None  # si falla, seguimos sin él

# =================== Página ===================
st.set_page_config(page_title="Métricas", page_icon="📊")

# --- Emojis en el sidebar, robusto en todas las páginas ---
def add_sidebar_icons(mapping: dict[str, str]):
    # mapping = {"Home":"🏠", "Temporada":"📅", ...}
    items_js = ",".join([f'["{k}","{v}"]' for k, v in mapping.items()])
    js_code = """
    <script>
    const mapping = new Map([{items_js}]);
    let tries = 0;
    const iv = setInterval(() => {{
      const nav = window.parent.document.querySelector('[data-testid="stSidebarNav"] ul');
      if (!nav) {{ if (++tries>20) clearInterval(iv); return; }}
      const spans = nav.querySelectorAll('li a span');
      spans.forEach(span => {{
        const label = span.textContent.trim();
        const ico = mapping.get(label);
        if (ico && !span.dataset.iconApplied) {{
          span.dataset.iconApplied = "1";
          // evita duplicar si ya tenía emoji manual
          if (!label.startsWith(ico)) {{
            span.textContent = `${{ico}} ${{label}}`;
          }}
        }}
      }});
      if (++tries>20) clearInterval(iv);
    }}, 300);
    </script>
    """.format(items_js=items_js)

    components.html(js_code, height=0)

# 👉 Define aquí tus iconos (los textos deben coincidir EXACTO con los nombres del sidebar)
SIDEBAR_ICONS = {
    "Home": "🏠",
    "Temporada Actual": "📅",
    "Historico": "📈",
    "Métricas": "📊",
    "Matchlogs": "🧾",
}

add_sidebar_icons(SIDEBAR_ICONS)

st.header("Métricas y ROI por temporada")

# =================== Helpers ===================
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _normalize_season(df: pd.DataFrame) -> pd.DataFrame:
    if "Season" in df.columns:
        df = df.copy()
        df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")
    return df

def _load_main_metrics() -> pd.DataFrame:
    """
    metrics_main_by_season.csv → columnas esperadas:
    Season,accuracy,logloss,brier,roi,n_bets,n_wins,hit_rate,avg_odds_win,avg_overround,avg_conf,avg_entropy,avg_margin
    """
    df = _read_csv(OUT / "metrics_main_by_season.csv")
    if df.empty:
        return df
    df = _normalize_season(df)
    # deja sólo columnas conocidas si existen
    keep = [c for c in [
        "Season","accuracy","roi","n_bets","n_wins","hit_rate",
        "logloss","brier","avg_odds_win","avg_overround","avg_conf","avg_entropy","avg_margin"
    ] if c in df.columns]
    return df[keep] if keep else df

def _load_market_metrics() -> pd.DataFrame:
    """
    metrics_market_by_season.csv → columnas esperadas:
    Season,accuracy,logloss,brier,n_scored,roi,n_bets,n_wins,hit_rate,avg_odds_win,avg_overround
    """
    df = _read_csv(OUT / "metrics_market_by_season.csv")
    if df.empty:
        return df
    df = _normalize_season(df)
    keep = [c for c in [
        "Season","accuracy","roi","n_bets","n_wins","hit_rate",
        "logloss","brier","n_scored","avg_odds_win","avg_overround"
    ] if c in df.columns]
    return df[keep] if keep else df

def _build_roi_long(main_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve DF largo: Season | ROI | Serie (Modelo / Bet365)
    """
    blocks = []
    if not main_df.empty and {"Season","roi"}.issubset(main_df.columns):
        a = main_df[["Season","roi"]].copy().rename(columns={"roi":"ROI"})
        a["Serie"] = "Modelo"
        blocks.append(a)
    if not market_df.empty and {"Season","roi"}.issubset(market_df.columns):
        b = market_df[["Season","roi"]].copy().rename(columns={"roi":"ROI"})
        b["Serie"] = "Bet365"
        blocks.append(b)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=["Season","ROI","Serie"])

def _merge_key_metrics(main_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Une métricas clave por temporada (sufijos _model / _bet365).
    """
    left = main_df.add_suffix("_model") if not main_df.empty else pd.DataFrame()
    right = market_df.add_suffix("_bet365") if not market_df.empty else pd.DataFrame()
    if not left.empty:
        left = left.rename(columns={"Season_model":"Season"})
    if not right.empty:
        right = right.rename(columns={"Season_bet365":"Season"})
    if left.empty and right.empty:
        return pd.DataFrame()
    if left.empty:
        return right
    if right.empty:
        return left
    out = pd.merge(left, right, on="Season", how="outer").sort_values("Season")
    return out

# =================== Carga de datos ===================
main_df = _load_main_metrics()
market_df = _load_market_metrics()

if main_df.empty and market_df.empty:
    st.info("No se encontraron métricas en `outputs/`.\n\n"
            "Asegúrate de generar:\n"
            "- outputs/metrics_main_by_season.csv\n"
            "- outputs/metrics_market_by_season.csv (opcional, para baseline Bet365)")
    st.stop()

roi_long = _build_roi_long(main_df, market_df)
if roi_long.empty:
    st.info("No se pudo construir la tabla de ROI por temporada (faltan columnas 'Season' y/o 'roi').")
    st.stop()

# =================== Vistas ===================

# Tabla ROI por temporada (Modelo vs Bet365)
with st.expander("Ver tabla ROI por temporada", expanded=False):
    tbl = roi_long.sort_values(["Season","Serie"]).copy()
    # formato % en ROI
    if "ROI" in tbl.columns:
        tbl["ROI"] = pd.to_numeric(tbl["ROI"], errors="coerce")
    st.dataframe(tbl, use_container_width=True, hide_index=True)

# Barras ROI por temporada
try:
    fig = px.bar(
        roi_long.sort_values("Season"),
        x="Season", y="ROI", color="Serie", barmode="group",
        title="ROI por temporada"
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"No pude dibujar el gráfico de barras: {e}")

# Línea de evolución del ROI
try:
    line_df = roi_long.sort_values(["Serie", "Season"])
    fig2 = px.line(line_df, x="Season", y="ROI", color="Serie", markers=True,
                   title="Evolución del ROI por temporada")
    fig2.update_yaxes(tickformat=".0%")
    fig2.update_layout(legend_title_text="")
    st.plotly_chart(fig2, use_container_width=True)
except Exception:
    pass

# (Opcional) Métricas clave por temporada
with st.expander("Métricas clave por temporada (tabla unificada)", expanded=False):
    merged = _merge_key_metrics(main_df, market_df)
    if merged.empty:
        st.info("No hay métricas clave suficientes para mostrar.")
    else:
        # Orden de columnas: Season | modelo (accuracy, roi, hit_rate, n_bets, n_wins) | bet365 (...)
        preferred_order = [
            "Season",
            "accuracy_model","roi_model","hit_rate_model","n_bets_model","n_wins_model",
            "accuracy_bet365","roi_bet365","hit_rate_bet365","n_bets_bet365","n_wins_bet365",
            "avg_overround_model","avg_overround_bet365",
            "logloss_model","logloss_bet365","brier_model","brier_bet365",
            "avg_conf_model","avg_entropy_model","avg_margin_model","n_scored_bet365","avg_odds_win_model","avg_odds_win_bet365"
        ]
        cols = [c for c in preferred_order if c in merged.columns] + [c for c in merged.columns if c not in preferred_order]
        view = merged[cols].copy()

        # Formatos numéricos amables
        for c in view.columns:
            lc = str(c).lower()
            if lc.endswith(("accuracy_model","accuracy_bet365","roi_model","roi_bet365","hit_rate_model","hit_rate_bet365")):
                view[c] = pd.to_numeric(view[c], errors="coerce")
            elif "overround" in lc or "logloss" in lc or "brier" in lc or "avg_" in lc or "odds" in lc:
                view[c] = pd.to_numeric(view[c], errors="coerce")

        st.dataframe(view, use_container_width=True, hide_index=True)
