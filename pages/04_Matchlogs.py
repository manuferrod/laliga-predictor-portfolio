# matchlogs.py — Visor de matchlogs (solo partidos disputados; sin fugas de predicciones)
from __future__ import annotations
import streamlit.components.v1 as components
import re
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Matchlogs", page_icon="📋", layout="wide")

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

st.header("Matchlogs por temporada")

# ================== Local helpers ==================
def _find_outputs_dir() -> Path:
    """
    Detecta la carpeta 'outputs' de forma robusta:
    - mismo nivel que el repo (parents[1]/outputs si la página está en /pages)
    - directorio del archivo actual / 'outputs'
    - CWD / 'outputs'
    - buscar hacia arriba hasta 4 niveles
    Devuelve la PRIMERA coincidencia existente; si ninguna existe, devuelve la primera candidata.
    """
    here = Path(__file__).resolve()
    candidates: list[Path] = []

    # 1) Si el archivo está en /pages, el repo root suele ser parents[1]
    candidates.append(here.parents[1] / "outputs")  # .../repo/outputs
    # 2) Mismo directorio del archivo
    candidates.append(here.parent / "outputs")
    # 3) CWD/outputs
    candidates.append(Path.cwd() / "outputs")
    # 4) Buscar hacia arriba varios niveles
    for i in range(2, 6):
        candidates.append(here.parents[i] / "outputs" if len(here.parents) > i else here.parent / "outputs")

    # Devuelve la primera que exista; si ninguna existe, devuelve la 1ª candidata
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

OUT = _find_outputs_dir()

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _list_seasons(out_dir: Path) -> list[int]:
    seasons = set()
    for p in out_dir.glob("matchlogs_*.csv"):
        m = re.match(r"matchlogs_(\d{4})\.csv$", p.name)
        if m:
            seasons.add(int(m.group(1)))
    return sorted(seasons)

def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def _ensure_matchday(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "Matchday" not in df.columns:
        d = df.copy().reset_index(drop=True)
        d["Matchday"] = np.arange(1, len(d) + 1)
        return d
    return df

def _only_played_mask(df: pd.DataFrame) -> pd.Series:
    """Disputados = y_true ∈ {H,D,A}."""
    if "y_true" not in df.columns:
        return pd.Series(False, index=df.index)
    t = df["y_true"].astype(str).str.strip().str.upper()
    return t.isin({"H","D","A"})

def _source_file(source: str, season: int, out_dir: Path) -> Path:
    return out_dir / (f"matchlogs_{season}.csv" if source == "Modelo" else f"matchlogs_market_{season}.csv")

def _normalize_market_cols(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Unifica nombres para la vista cuando la fuente es Bet365."""
    if df.empty or source != "Bet365":
        return df
    d = df.copy()
    if "y_pred_market" in d.columns and "y_pred" not in d.columns:
        d = d.rename(columns={"y_pred_market": "y_pred"})
    ren = {}
    if "pH_mkt_pred" in d.columns: ren["pH_mkt_pred"] = "proba_H"
    if "pD_mkt_pred" in d.columns: ren["pD_mkt_pred"] = "proba_D"
    if "pA_mkt_pred" in d.columns: ren["pA_mkt_pred"] = "proba_A"
    if ren:
        d = d.rename(columns=ren)
    return d

# ================== Comprobaciones ==================
st.caption(f"Ruta de outputs detectada: `{OUT}`")

has_model_logs = any(OUT.glob("matchlogs_*.csv"))
has_market_logs = any(OUT.glob("matchlogs_market_*.csv"))
if not has_model_logs and not has_market_logs:
    st.warning("No se encontraron `matchlogs_*.csv` ni `matchlogs_market_*.csv` en la carpeta detectada de `outputs/`.\n"
               "Verifica la ruta anterior y que los ficheros existan.")
    st.stop()

seasons = _list_seasons(OUT)
if not seasons:
    st.warning("No hay temporadas detectadas (no hay ficheros `matchlogs_YYYY.csv`).")
    st.stop()

# ================== Controles superiores ==================
c1, c2, c3 = st.columns([1,1,2])
with c1:
    source = st.radio("Fuente", ["Modelo", "Bet365"], horizontal=True)
with c2:
    sel = st.selectbox("Temporada", seasons, index=len(seasons)-1)
with c3:
    team = st.text_input("Filtrar por equipo (contiene)", placeholder="barcelona, betis, ...")

# ================== Carga y normalización ==================
path = _source_file(source, sel, OUT)
df = _read_csv(path)
if df.empty:
    st.info(f"No hay matchlog para {source} / {sel}.")
    st.stop()

df = _ensure_matchday(_ensure_date(df.copy()))
df = _normalize_market_cols(df, source)
df = df.sort_values(["Date","HomeTeam_norm","AwayTeam_norm"], na_position="last").reset_index(drop=True)

# 🚫 POLÍTICA PÚBLICA: ocultar partidos futuros (sin resultado real) y, por tanto, sus predicciones
mask_played = _only_played_mask(df)
df = df[mask_played].reset_index(drop=True)
st.caption("🔒 Por política pública, esta vista solo muestra **partidos disputados**. Las predicciones futuras están en el área privada.")

if df.empty:
    st.info("No hay partidos disputados para los filtros seleccionados.")
    st.stop()

# ================== Filtros ==================
with st.expander("Filtros"):
    cols = st.columns(3)
    with cols[0]:
        # filtro por equipo (texto)
        pass
    with cols[1]:
        # Jornada
        all_md = pd.to_numeric(df["Matchday"], errors="coerce").dropna().astype(int).sort_values().unique().tolist()
        wk_sel = st.selectbox("Jornada", options=["(todas)"] + all_md, index=0)
    with cols[2]:
        order_ev_desc = st.checkbox("Ordenar por EV (desc)", value=False)

# filtro por equipo
if team:
    t = team.strip()
    mask = pd.Series(False, index=df.index)
    for c in ["HomeTeam_norm", "AwayTeam_norm"]:
        if c in df.columns:
            mask = mask | df[c].astype(str).str.contains(t, case=False, na=False)
    df = df[mask]

# filtro por jornada
if wk_sel != "(todas)":
    df = df[pd.to_numeric(df["Matchday"], errors="coerce").astype("Int64") == int(wk_sel)]

# orden final por EV
if order_ev_desc and "ev_pick" in df.columns:
    df = df.sort_values("ev_pick", ascending=False)

# ================== KPIs rápidos ==================
n_rows = int(len(df))
roi_pick = None
hit_rate = None
n_bets = None

if "profit" in df.columns and n_rows > 0:
    roi_pick = float(pd.to_numeric(df["profit"], errors="coerce").fillna(0).sum() / n_rows)

if "correct" in df.columns and n_rows > 0:
    hit_rate = float(pd.to_numeric(df["correct"], errors="coerce").where(lambda x: x.isin([0,1])).mean())

if "bet_placed" in df.columns:
    n_bets = int((df["bet_placed"] == True).sum())

k1, k2, k3 = st.columns(3)
k1.metric("Filas visibles", f"{n_rows}")
if roi_pick is not None:
    k2.metric("ROI (pick)", f"{roi_pick:.1%}")
if hit_rate is not None:
    k3.metric("Acierto", f"{hit_rate:.1%}")
if n_bets is not None:
    st.caption(f"Apuestas colocadas en vista: **{n_bets}**")

st.divider()

# ================== Selección de columnas para vista ==================
preferred_cols = [
    "Date","Matchday","Season","HomeTeam_norm","AwayTeam_norm",
    "y_true","y_pred",
    "proba_H","proba_D","proba_A",
    "B365H","B365D","B365A","overround",
    "odds_pick","p_pick","ev_pick","kelly_pick","bet_placed",
    "correct","profit","cum_profit_season",
]

# 🔒 Eliminamos columnas privadas/no necesarias
cols_to_drop = {"pred_key", "pred_key_match"}
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

ordered_cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
view = df[ordered_cols].copy()

# formateos suaves
if "Date" in view.columns:
    view["Date"] = pd.to_datetime(view["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
for c in ["proba_H","proba_D","proba_A","p_pick","kelly_pick","ev_pick"]:
    if c in view.columns:
        view[c] = pd.to_numeric(view[c], errors="coerce").round(4)
for c in ["profit","cum_profit_season","odds_pick","B365H","B365D","B365A","overround"]:
    if c in view.columns:
        view[c] = pd.to_numeric(view[c], errors="coerce").round(3)

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
    "overround": "Overround",
    "odds_pick": "Cuota pick",
    "p_pick": "p(pick)",
    "ev_pick": "EV pick",
    "kelly_pick": "Kelly",
    "bet_placed": "Apuesta colocada",
    "correct": "Acierto",
    "profit": "Beneficio",
    "cum_profit_season": "Beneficio acumulado (temp)",
}
view = view.rename(columns=rename_map)

st.dataframe(view, use_container_width=True, hide_index=True)

# ================== Descargas ==================
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        "Descargar CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name=f"matchlog_{sel}_{'modelo' if source=='Modelo' else 'bet365'}_SOLO_DISPUTADOS.csv",
        mime="text/csv",
    )
with col_dl2:
    st.download_button(
        "Descargar JSON",
        data=view.to_json(orient="records", force_ascii=False).encode("utf-8"),
        file_name=f"matchlog_{sel}_{'modelo' if source=='Modelo' else 'bet365'}_SOLO_DISPUTADOS.json",
        mime="application/json",
    )
