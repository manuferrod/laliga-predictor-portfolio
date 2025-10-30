# pages/01_Historico.py
from __future__ import annotations
import streamlit.components.v1 as components
import sys, re
from pathlib import Path
import streamlit as st
import plotly.express as px
import pandas as pd

# ----------------- Paths base -----------------
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

st.set_page_config(page_title="Histórico", page_icon="📈", layout="wide")

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
    "Histórico": "📈",
    "Métricas": "📊",
    "Matchlogs": "🧾",
}

add_sidebar_icons(SIDEBAR_ICONS)

st.header("Curvas de beneficio acumulado")

# ----------------- Utils locales -----------------
def ensure_outputs_dir():
    OUT.mkdir(parents=True, exist_ok=True)

def has_outputs() -> bool:
    return OUT.exists() and any(OUT.iterdir())

def _list_matchlog_seasons() -> list[int]:
    seasons = set()
    # matchlogs_YYYY.csv
    for p in OUT.glob("matchlogs_*.csv"):
        m = re.match(r"matchlogs_(\d{4})\.csv$", p.name)
        if m:
            seasons.add(int(m.group(1)))
    # por si solo existen market:
    for p in OUT.glob("matchlogs_market_*.csv"):
        m = re.match(r"matchlogs_market_(\d{4})\.csv$", p.name)
        if m:
            seasons.add(int(m.group(1)))
    return sorted(seasons)

def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"No pude leer {path.name}: {e}")
    return None

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # nombres consistentes
    cols = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)
    # fecha a datetime si existe
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def _order_index(df: pd.DataFrame) -> pd.Series:
    """Devuelve la serie de orden (Jornada preferente, si no por fecha, si no por índice)."""
    if "Matchweek" in df.columns:
        # asegurar numérico
        try:
            ord_ser = pd.to_numeric(df["Matchweek"], errors="coerce")
            # si hay empates por jornada (varios días), establecemos orden estable por fecha si existe
            if "Date" in df.columns:
                df2 = df.copy()
                df2["_ord"] = ord_ser
                df2 = df2.sort_values(["_ord", "Date"], kind="stable")
                df2["_x"] = range(1, len(df2) + 1)
                # reindex a orden original
                return df2.set_index(df2.index)["_x"].sort_index()
            return ord_ser
        except Exception:
            pass
    if "Date" in df.columns:
        # ordenar por fecha, asignar 1..n
        df2 = df.sort_values("Date").copy()
        df2["_x"] = range(1, len(df2) + 1)
        return df2.set_index(df2.index)["_x"].sort_index()
    # fallback: índice natural
    return pd.Series(range(1, len(df) + 1), index=df.index)

def _find_profit_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Devuelve (profit_col, cumprofit_col).
    Busca nombres comunes: 'profit','benefit','beneficio','ret','gain' y 'cumprofit','cum_benefit','cum_beneficio'
    """
    profit_candidates = ["profit", "benefit", "beneficio", "ret", "gain"]
    cum_candidates = ["cumprofit", "cum_profit", "cumbenefit", "cum_benefit", "cum_beneficio", "beneficio_acum", "benefit_cum"]
    profit_col = next((c for c in profit_candidates if c in df.columns), None)
    cum_col = next((c for c in cum_candidates if c in df.columns), None)
    return profit_col, cum_col

def load_cumprofit_from_matchlogs(season: int) -> pd.DataFrame:
    """
    Construye un DF con columnas:
      - x (1..n)
      - Modelo (serie acumulada)
      - Bet365 (si existe matchlogs_market)
    """
    model_path = OUT / f"matchlogs_{season}.csv"
    mkt_path   = OUT / f"matchlogs_market_{season}.csv"

    df_model = _read_csv_safe(model_path)
    df_mkt   = _read_csv_safe(mkt_path)

    if df_model is None and df_mkt is None:
        return pd.DataFrame()

    out = pd.DataFrame({"x": []})

    # --- Modelo ---
    if df_model is not None and not df_model.empty:
        df_model = _normalize_columns(df_model)
        x_ser = _order_index(df_model)
        profit_col, cum_col = _find_profit_columns(df_model)
        if cum_col and cum_col in df_model.columns:
            serie_model = df_model[cum_col].sort_index()
            # si hay desorden, ordenar por x:
            try:
                tmp = pd.DataFrame({"x": x_ser, "v": serie_model})
                tmp = tmp.sort_values("x")
                out["x"] = tmp["x"].to_list()
                out["Modelo"] = tmp["v"].to_list()
            except Exception:
                out["x"] = range(1, len(serie_model) + 1)
                out["Modelo"] = serie_model.values
        elif profit_col and profit_col in df_model.columns:
            tmp = df_model.copy()
            tmp["x"] = x_ser
            tmp = tmp.sort_values("x")
            tmp["Modelo"] = pd.to_numeric(tmp[profit_col], errors="coerce").fillna(0).cumsum()
            out = tmp[["x", "Modelo"]].copy()
        else:
            st.warning(f"'{model_path.name}' no tiene columnas de beneficio reconocibles (profit/cumprofit).")
    # --- Bet365 (market) ---
    if df_mkt is not None and not df_mkt.empty:
        df_mkt = _normalize_columns(df_mkt)
        x_ser_mkt = _order_index(df_mkt)
        profit_col, cum_col = _find_profit_columns(df_mkt)
        if "x" not in out.columns:
            out["x"] = _order_index(df_mkt).to_list()
        if cum_col and cum_col in df_mkt.columns:
            tmp = df_mkt.copy()
            tmp["x"] = x_ser_mkt
            tmp = tmp.sort_values("x")
            out = pd.merge(
                out, tmp[["x", cum_col]].rename(columns={cum_col: "Bet365"}),
                on="x", how="outer"
            ).sort_values("x")
        elif profit_col and profit_col in df_mkt.columns:
            tmp = df_mkt.copy()
            tmp["x"] = x_ser_mkt
            tmp = tmp.sort_values("x")
            tmp["Bet365"] = pd.to_numeric(tmp[profit_col], errors="coerce").fillna(0).cumsum()
            out = pd.merge(out, tmp[["x", "Bet365"]], on="x", how="outer").sort_values("x")
        else:
            st.warning(f"'{mkt_path.name}' no tiene columnas de beneficio reconocibles (profit/cumprofit).")

    # Rellenos y orden
    if "x" not in out.columns or out.empty:
        return pd.DataFrame()
    out = out.sort_values("x")
    # Asegura enteros para x si procede
    try:
        out["x"] = pd.to_numeric(out["x"], errors="coerce")
    except Exception:
        pass
    # Reset index y limpieza
    out = out.reset_index(drop=True)
    return out

# ----------------- Flujo de página -----------------
ensure_outputs_dir()
if not has_outputs():
    st.warning("No se encontraron artefactos en `outputs/`. "
               "Sube los ficheros generados por el motor o espera a la sincronización.")
    st.stop()

# Temporadas a partir de los nuevos nombres de matchlogs
seasons = _list_matchlog_seasons()
if not seasons:
    st.warning("No se detectaron matchlogs en `outputs/` (matchlogs_YYYY.csv / matchlogs_market_YYYY.csv).")
    st.stop()

sel = st.selectbox("Temporada", seasons, index=len(seasons) - 1)

# Cargar curvas a partir de matchlogs
df_curve = load_cumprofit_from_matchlogs(sel)
if df_curve is None or df_curve.empty:
    st.info(f"No pude construir la curva para la temporada {sel}. "
            "Revisa que existan columnas de beneficio (profit o cumprofit) en los matchlogs.")
    st.stop()

# Normaliza estructura esperada: x + series
df_curve.columns = [str(c).strip() for c in df_curve.columns]
if "x" not in df_curve.columns:
    df_curve.insert(0, "x", range(1, len(df_curve) + 1))

series_cols = [c for c in df_curve.columns if c != "x"]
if not series_cols:
    st.warning("No hay series para graficar ('Modelo', 'Bet365').")
    st.dataframe(df_curve.head(), use_container_width=True, hide_index=True)
    st.stop()

# Plot
long = df_curve.melt(id_vars="x", var_name="Serie", value_name="Beneficio")
fig = px.line(long, x="x", y="Beneficio", color="Serie",
              title=f"Beneficio acumulado — Temporada {sel}")
fig.update_layout(legend_title_text="")
fig.update_xaxes(title_text="Jornadas")
fig.update_yaxes(title_text="Beneficio")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Ver datos"):
    st.dataframe(df_curve, use_container_width=True, hide_index=True)
