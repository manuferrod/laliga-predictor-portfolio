import streamlit as st
import pandas as pd
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

long = df.melt(id_vars="x", var_name="Serie", value_name="Beneficio")
fig = px.line(long, x="x", y="Beneficio", color="Serie", markers=False, title=f"Beneficio acumulado — Temporada {sel}")
fig.update_layout(legend_title_text="")
st.plotly_chart(fig, use_container_width=True)

st.caption("Nota: la serie *Bet365* es el benchmark con stake 1 por apuesta; el modelo usa el mismo stake y selección por EV.")
