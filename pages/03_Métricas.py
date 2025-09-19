import streamlit as st
import pandas as pd
import plotly.express as px
from scripts.io import load_csv

st.set_page_config(page_title="Métricas", page_icon="📊")
st.header("Métricas y ROI por temporada")

# ROI por temporada
roi_b = load_csv("roi_by_season_base.csv")
roi_s = load_csv("roi_by_season_smote.csv") if (st.session_state.get("has_smote", True)) else pd.DataFrame()

if not roi_b.empty:
    rb = roi_b.rename(columns=lambda x: x.strip())
    if "ROI" not in rb.columns:
        roi_col = [c for c in rb.columns if c.lower().startswith("roi")][0]
        rb = rb.rename(columns={roi_col: "ROI"})
    rb["Modelo"] = "BASE"
    data = rb[["Season","ROI","Modelo"]]
    if not roi_s.empty:
        rs = roi_s.rename(columns=lambda x: x.strip())
        if "ROI" not in rs.columns:
            roi_col = [c for c in rs.columns if c.lower().startswith("roi")][0]
            rs = rs.rename(columns={roi_col: "ROI"})
        rs["Modelo"] = "SMOTE"
        data = pd.concat([data, rs[["Season","ROI","Modelo"]]], ignore_index=True)

    fig = px.bar(data, x="Season", y="ROI", color="Modelo", barmode="group", title="ROI por temporada")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Aún no hay ROI por temporada.")

st.divider()
# Clasificación por temporada (tabla)
try:
    clf_b = load_csv("classification_by_season_base.csv")
    st.subheader("Clasificación por temporada (BASE)")
    st.dataframe(clf_b, use_container_width=True, hide_index=True)
except Exception:
    st.info("No encontré classification_by_season_base.csv")
