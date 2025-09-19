import streamlit as st
from scripts.io import seasons, load_matchlog

st.set_page_config(page_title="Matchlogs", page_icon="📋")
st.header("Matchlogs por temporada")

seas = seasons()
if not seas:
    st.warning("No hay temporadas en outputs.")
    st.stop()

model = st.radio("Modelo", ["base","smote"], horizontal=True)
sel = st.selectbox("Temporada", seas, index=len(seas)-1)

df = load_matchlog(model, sel)
if df.empty:
    st.info(f"No hay matchlog para {model} / {sel}.")
    st.stop()

team = st.text_input("Filtrar por equipo (contiene)")
if team:
    mask = df.get("HomeTeam_norm", "").str.contains(team, case=False, na=False) | \
           df.get("AwayTeam_norm", "").str.contains(team, case=False, na=False)
    df = df[mask]

st.dataframe(df, use_container_width=True, hide_index=True)
st.download_button("Descargar CSV", df.to_csv(index=False).encode("utf-8"),
                   file_name=f"matchlog_{sel}_{model}.csv", mime="text/csv")
