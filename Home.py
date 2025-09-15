
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predicción LaLiga – Portfolio", layout="wide")
st.title("Predicciones próxima jornada (demo pública)")

@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # Derivadas útiles
    df["confidence"] = df[["p_home","p_draw","p_away"]].max(axis=1)
    df["pred_label"] = df["predicted_result"].map({"H":"Local","D":"Empate","A":"Visitante"})
    return df

try:
    df = load_predictions("outputs/predictions_next.json")
except FileNotFoundError:
    st.error("No se encontró outputs/predictions_next.json. Sube tus predicciones y vuelve a cargar la app.")
    st.stop()

# Panel superior con info rápida
left, right = st.columns([3,2])
with left:
    st.subheader("Partidos próximos")
    teams = sorted(set(df["home_team"]).union(set(df["away_team"])))
    filtro_equipos = st.multiselect("Filtrar por equipo (opcional):", teams, default=[])
    if filtro_equipos:
        mask = df["home_team"].isin(filtro_equipos) | df["away_team"].isin(filtro_equipos)
        df_view = df[mask].copy()
    else:
        df_view = df.copy()

    # Orden por mayor confianza
    df_view = df_view.sort_values("confidence", ascending=False)

    cols = ["date","home_team","away_team","p_home","p_draw","p_away","pred_label","confidence","model_version"]
    st.dataframe(df_view[cols].rename(columns={
        "date":"Fecha","home_team":"Local","away_team":"Visitante",
        "p_home":"P(Local)","p_draw":"P(Empate)","p_away":"P(Visitante)",
        "pred_label":"Predicción","confidence":"Confianza","model_version":"Modelo"
    }), use_container_width=True, hide_index=True)

with right:
    st.subheader("Resumen")
    st.metric("Nº partidos", len(df))
    st.metric("Confianza media", f"{df['confidence'].mean():.0%}")
    try:
        as_of = pd.to_datetime(df["as_of"].iloc[0])
        st.caption(f"Generado: {as_of.strftime('%Y-%m-%d %H:%M %Z')}")
    except Exception:
        st.caption("Generado: N/D")

st.divider()
st.caption("Demo con fines de evaluación académica. No ofrece consejo de apuesta. No redistribuye datos de terceros.")
