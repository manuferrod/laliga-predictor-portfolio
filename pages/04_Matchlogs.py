import streamlit as st
import pandas as pd
from scripts.io import seasons, load_matchlog

st.set_page_config(page_title="Matchlogs", page_icon="📋")
st.header("Matchlogs por temporada")

seas = seasons()
model = st.radio("Modelo", ["base","smote"], horizontal=True)

if not seas:
    st.warning("No hay temporadas en outputs.")
    st.stop()

sel = st.selectbox("Temporada", seas, index=len(seas)-1)
df = load_matchlog(model, sel)
if df.empty:
    st.info(f"No hay matchlog para {model} / {sel}.")
    st.stop()

# Filtros rápidos
team = st.text_input("Filtrar por equipo (contiene)")
if team:
    mask = df["HomeTeam_norm"].str.contains(team, case=False, na=False) | df["AwayTeam_norm"].str.contains(team, case=False, na=False)
    df = df[mask]

st.dataframe(df, use_container_width=True, hide_index=True)

# Descarga
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Descargar CSV", csv, file_name=f"matchlog_{sel}_{model}.csv", mime="text/csv")
