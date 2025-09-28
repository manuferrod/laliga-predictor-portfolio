# pages/03_Métricas.py
import sys, importlib.util
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

# =============== Import robusto de scripts/io =================
ROOT = Path(__file__).resolve().parents[1]   # raíz del repo
SCRIPTS_DIR = ROOT / "scripts"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import scripts.io as io  # intento normal
except Exception:
    spec = importlib.util.spec_from_file_location("io", SCRIPTS_DIR / "io.py")
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)  # type: ignore

# =================== Página ===================
st.set_page_config(page_title="Métricas", page_icon="📊")
st.header("Métricas y ROI por temporada")

# -------- helpers locales (no dependemos de io.load_roi) --------
def _pick_col(df: pd.DataFrame, *cands: str) -> str | None:
    low = {str(c).lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    # heurística de temporada (1990-2100)
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any() and (s.dropna().between(1990, 2100).mean() > 0.6):
            return c
    return None

def _load_roi_model(model: str) -> pd.DataFrame:
    """
    Lee outputs/roi_by_season_{model}.csv y normaliza a columnas:
      - Season (Int64)
      - ROI (float)
    Devuelve DF vacío si no existe.
    """
    try:
        df = io.load_csv(f"roi_by_season_{model}.csv")
    except FileNotFoundError:
        return pd.DataFrame()

    # Normaliza Season (acepta 'test_season', 'season', etc.)
    season_col = _pick_col(df, "test_season", "Season", "season", "Temporada")
    if season_col and season_col != "Season":
        df = df.rename(columns={season_col: "Season"})
    if "Season" in df.columns:
        df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")

    # Normaliza ROI (cualquier columna que empiece por 'roi')
    if "ROI" not in df.columns:
        cand = next((c for c in df.columns if str(c).lower().startswith("roi")), None)
        if cand:
            df = df.rename(columns={cand: "ROI"})
    if "ROI" in df.columns:
        df["ROI"] = pd.to_numeric(df["ROI"], errors="coerce")

    # Nos quedamos con lo relevante
    keep = [c for c in ["Season", "ROI"] if c in df.columns]
    return df[keep] if keep else pd.DataFrame()

# -------- carga de datos --------
roi_base = _load_roi_model("base")
roi_smote = _load_roi_model("smote")

if roi_base.empty and roi_smote.empty:
    st.info("Aún no hay ROI por temporada en outputs/. Asegúrate de generar:\n"
            "- outputs/roi_by_season_base.csv\n- outputs/roi_by_season_smote.csv")
    st.stop()

# Unificamos
blocks = []
for tag, df in [("BASE", roi_base), ("SMOTE", roi_smote)]:
    if not df.empty and {"Season", "ROI"}.issubset(df.columns):
        tmp = df[["Season", "ROI"]].copy()
        tmp["Modelo"] = tag
        blocks.append(tmp)

if not blocks:
    st.info("No hay columnas 'Season' y 'ROI' válidas en los ficheros ROI.")
    st.stop()

plot_df = pd.concat(blocks, ignore_index=True)

# Tabla
with st.expander("Ver tabla ROI por temporada", expanded=False):
    st.dataframe(plot_df.sort_values(["Season", "Modelo"]),
                 use_container_width=True, hide_index=True)

# Barras ROI por temporada
try:
    fig = px.bar(
        plot_df.sort_values("Season"),
        x="Season", y="ROI", color="Modelo", barmode="group",
        title="ROI por temporada"
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"No pude dibujar el gráfico de barras: {e}")

# Línea de tendencia (opcional)
try:
    line_df = plot_df.sort_values(["Modelo", "Season"])
    fig2 = px.line(line_df, x="Season", y="ROI", color="Modelo", markers=True,
                   title="Evolución del ROI por temporada")
    fig2.update_yaxes(tickformat=".0%")
    fig2.update_layout(legend_title_text="")
    st.plotly_chart(fig2, use_container_width=True)
except Exception:
    pass
