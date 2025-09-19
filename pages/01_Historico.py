import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import plotly.express as px
from scripts.io import seasons, load_cumprofit

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
