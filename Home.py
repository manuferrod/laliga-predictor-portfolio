from pathlib import Path
import streamlit as st
import pandas as pd
from scripts.io import seasons, load_roi

st.set_page_config(page_title="LaLiga 1X2 · Portfolio", page_icon="⚽", layout="wide")
st.title("LaLiga 1X2 · Modelo vs Bet365")

# Detalles y resumen rápido
seas = seasons()
colA, colB, colC = st.columns(3)
colA.metric("Temporadas disponibles", len(seas) if seas else 0)
# KPI ROI último año (base/smote si están)
roi_base = load_roi("base")
roi_smote = load_roi("smote")
if not roi_base.empty:
    last_season = int(roi_base["Season"].max())
    kpi_base = roi_base.loc[roi_base["Season"] == last_season]
    kpi = float(kpi_base["ROI"].iloc[0]) if "ROI" in kpi_base.columns else float(kpi_base.filter(like="roi", axis=1).iloc[0,0])
    colB.metric(f"ROI modelo BASE {last_season}", f"{kpi:.1%}")
if not roi_smote.empty:
    last_season_s = int(roi_smote["Season"].max())
    kpi_sm = roi_smote.loc[roi_smote["Season"] == last_season_s]
    kpi_s = float(kpi_sm["ROI"].iloc[0]) if "ROI" in kpi_sm.columns else float(kpi_sm.filter(like="roi", axis=1).iloc[0,0])
    colC.metric(f"ROI modelo SMOTE {last_season_s}", f"{kpi_s:.1%}")

st.divider()
st.markdown(
"""
**Navegación rápida**
- **Curvas**: beneficio acumulado por temporada (modelo vs Bet365).
- **Métricas**: clasificación (accuracy, logloss, AUC) y ROIs por temporada.
- **Matchlogs**: tabla de partidos por temporada (filtrable).
"""
)

st.info("Los datos de `outputs/` se actualizan automáticamente desde el motor (repo A) tras cada ronda.")
