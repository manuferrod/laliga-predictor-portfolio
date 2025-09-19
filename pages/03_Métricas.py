import streamlit as st
import pandas as pd
import plotly.express as px
from scripts.io import load_csv

st.set_page_config(page_title="Métricas", page_icon="📊")
st.header("Métricas y ROI por temporada")

def _load_roi_model(model):
    try:
        df = load_csv(f"roi_by_season_{model}.csv")
        # normaliza nombres
        if "Season" not in df.columns:
            cand = [c for c in df.columns if c.lower() == "season" or c.lower().endswith("season")]
            if cand: df = df.rename(columns={cand[0]:"Season"})
        if "ROI" not in df.columns:
            cand = [c for c in df.columns if c.lower().startswith("roi")]
            if cand: df = df.rename(columns={cand[0]:"ROI"})
        return df
    except FileNotFoundError:
        return pd.DataFrame()

roi_b = _load_roi_model("base")
roi_s = _load_roi_model("smote")

if roi_b.empty and roi_s.empty:
    st.info("Aún no hay ROI por temporada en outputs.")
else:
    data = []
    for tag, df in [("BASE", roi_b), ("SMOTE", roi_s)]:
        if not df.empty and "Season" in df.columns and "ROI" in df.columns:
            tmp = df[["Season","ROI"]].copy()
            tmp["Modelo"] = tag
            data.append(tmp)
    if data:
        plot_df = pd.concat(data, ignore_index=True)
        fig = px.bar(plot_df, x="Season", y="ROI", color="Modelo", barmode="group", title="ROI por temporada")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
