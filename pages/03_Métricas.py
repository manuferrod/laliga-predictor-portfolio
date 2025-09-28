# pages/03_Métricas.py
import sys, importlib.util
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# =============== Import robusto de scripts/io =================
# OJO: desde /pages necesitamos subir un nivel para llegar a la raíz.
ROOT = Path(__file__).resolve().parents[1]   # <- repo root
SCRIPTS_DIR = ROOT / "scripts"

# Asegura que la raíz está en sys.path antes del import normal
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import scripts.io as io  # intento normal (cuando 'scripts' está en sys.path)
except Exception:
    # Fallback: carga directa desde la ruta del archivo
    spec = importlib.util.spec_from_file_location("io", SCRIPTS_DIR / "io.py")
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)  # type: ignore

# =================== Página ===================
st.set_page_config(page_title="Métricas", page_icon="📊")
st.header("Métricas y ROI por temporada")

# Carga de ROI por temporada con normalización de columnas
roi_base = io.load_roi("base")   # devuelve DF (posiblemente vacío)
roi_smote = io.load_roi("smote")

if roi_base.empty and roi_smote.empty:
    st.info("Aún no hay ROI por temporada en outputs/.")
    st.stop()

# Unificamos los DataFrames que tengan Season y ROI válidos
blocks = []
for tag, df in [("BASE", roi_base), ("SMOTE", roi_smote)]:
    if df.empty:
        continue
    # Garantiza columnas esperadas
    if "Season" not in df.columns or "ROI" not in df.columns:
        continue
    tmp = df[["Season", "ROI"]].copy()
    tmp["Modelo"] = tag
    blocks.append(tmp)

if not blocks:
    st.info("No hay columnas 'Season' y 'ROI' válidas en los ficheros ROI.")
    st.stop()

plot_df = pd.concat(blocks, ignore_index=True)

# Tabla
with st.expander("Ver tabla ROI por temporada", expanded=False):
    st.dataframe(plot_df.sort_values(["Season","Modelo"]), use_container_width=True, hide_index=True)

# Gráfico de barras ROI por temporada
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
    st.error(f"No pude dibujar el gráfico: {e}")

# (Opcional) línea de tendencia por modelo
try:
    line_df = plot_df.sort_values(["Modelo", "Season"])
    fig2 = px.line(line_df, x="Season", y="ROI", color="Modelo", markers=True,
                   title="Evolución del ROI por temporada")
    fig2.update_yaxes(tickformat=".0%")
    fig2.update_layout(legend_title_text="")
    st.plotly_chart(fig2, use_container_width=True)
except Exception:
    pass
