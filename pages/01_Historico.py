# pages/01_Historico.py
# --- importar el módulo scripts/io de forma robusta ---
import sys, importlib.util
from pathlib import Path
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import scripts.io as io  # preferido
except Exception:
    spec = importlib.util.spec_from_file_location("io", SCRIPTS_DIR / "io.py")
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)  # type: ignore[attr-defined]

st.set_page_config(page_title="Curvas", page_icon="📈")
st.header("Curvas de beneficio acumulado")

# 1) temporadas
try:
    seas = io.seasons()
except Exception as e:
    st.error(f"No pude cargar temporadas desde outputs: {e}")
    st.stop()

if not seas:
    st.warning("No hay temporadas en outputs todavía.")
    st.stop()

sel = st.selectbox("Temporada", seas, index=len(seas)-1)

# 2) cargar curva
df = io.load_cumprofit(sel)

series_cols = [c for c in df.columns if c != "x"]
if not series_cols:
    st.warning(
        "No encontré series para graficar en "
        f"`outputs/cumprofit_curves/cumprofit_{sel}.json|csv`. "
        "Asegúrate de que el archivo contiene columnas/series (por ejemplo 'base', 'smote', 'bet365')."
    )
    st.dataframe(df.head())
    st.stop()

if df is None or df.empty:
    st.info(f"No encontré curvas para la temporada {sel}.")
    st.stop()

# 3) plot
long = df.melt(id_vars="x", var_name="Serie", value_name="Beneficio")
fig = px.line(long, x="x", y="Beneficio", color="Serie", title=f"Beneficio acumulado — Temporada {sel}")
fig.update_layout(legend_title_text="")
st.plotly_chart(fig, use_container_width=True)
