
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
st.title("Histórico y métricas (agregadas por temporada)")

@st.cache_data
def load_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

try:
    metrics = load_metrics("outputs/historical_metrics.csv")
except FileNotFoundError:
    st.error("No se encontró outputs/historical_metrics.csv")
    st.stop()

c1, c2 = st.columns([1,3])
with c1:
    season = st.selectbox("Temporada", sorted(metrics["season"].unique()))
    row = metrics.loc[metrics["season"] == season].iloc[0]
    st.metric("Accuracy (test)", f"{row['acc']:.2%}")
    st.metric("Log loss (test)", f"{row['logloss']:.2f}")
    st.metric("ROI simulado", f"{row['roi']:+.1%}")
    st.metric("Cobertura", f"{row['coverage']:.0%}")
    st.caption(f"N partidos: {int(row['n_games'])}")
    st.caption(f"Actualizado: {row['updated_at']}")
with c2:
    st.subheader("Comparativa por temporada")
    # Pequeña tabla ordenada por temporada
    st.dataframe(
        metrics.sort_values("season")[["season","acc","logloss","roi","n_games"]]
        .rename(columns={"season":"Temporada","acc":"Accuracy","logloss":"Log loss","roi":"ROI","n_games":"Partidos"}),
        use_container_width=True, hide_index=True
    )

st.divider()
st.caption("Estas métricas son agregadas; no se publican datos partido a partido para proteger la propiedad intelectual.")
