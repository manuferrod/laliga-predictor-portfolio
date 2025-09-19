# --- robust import of scripts/io.py even when running inside pages/ ---
import sys, importlib.util
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import scripts.io as io_mod   # preferido
except Exception as e:
    # fallback: carga directa por ruta (cuando el import normal falla)
    spec = importlib.util.spec_from_file_location("io", SCRIPTS_DIR / "io.py")
    io_mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(io_mod)  # type: ignore[attr-defined]
    except Exception as e2:
        st.stop()  # corta limpio con mensaje claro
        raise RuntimeError(f"No pude importar scripts/io.py: {e} | fallback: {e2}")

# usa io_mod.<función> en el resto del archivo:
seasons = io_mod.seasons
load_cumprofit = io_mod.load_cumprofit

seas = seasons()
df = load_cumprofit(sel)

st.set_page_config(page_title="Curvas", page_icon="📈")
st.header("Curvas de beneficio acumulado")

seas = seasons()
if not seas:
    st.warning("No hay temporadas en outputs todavía.")
    st.stop()

sel = st.selectbox("Temporada", seas, index=len(seas)-1)
df = load_cumprofit(sel)
if df.empty:
    st.error(f"No encontré curvas para la temporada {sel}.")
    st.stop()

fig = px.line(df.melt("x", var_name="Serie", value_name="Beneficio"), x="x", y="Beneficio", color="Serie")
fig.update_layout(legend_title_text="")
st.plotly_chart(fig, use_container_width=True)

st.caption("Nota: la serie *Bet365* es el benchmark con stake 1 por apuesta; el modelo usa el mismo stake y selección por EV.")
