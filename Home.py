from pathlib import Path
import streamlit as st
import pandas as pd
from scripts.io import seasons, load_roi

st.set_page_config(page_title="LaLiga 1X2 · Portfolio", page_icon="⚽", layout="wide")
st.title("LaLiga 1X2 · Modelo vs Bet365")

# ← robusto ante ausencia de Season / CSV
try:
    seas = seasons()
except Exception as e:
    seas = []
    st.warning(f"No pude inferir temporadas desde outputs: {e}")

colA, colB, colC = st.columns(3)
colA.metric("Temporadas disponibles", len(seas) if seas else 0)

roi_base = load_roi("base")
roi_smote = load_roi("smote")

def _latest_roi(df):
    if df.empty:
        return None, None
    if "Season" not in df.columns or "ROI" not in df.columns:
        return None, None
    s = int(df["Season"].dropna().max())
    val = float(df.loc[df["Season"] == s, "ROI"].iloc[0])
    return s, val

s_b, v_b = _latest_roi(roi_base)
if s_b is not None:
    colB.metric(f"ROI modelo BASE {s_b}", f"{v_b:.1%}")

s_s, v_s = _latest_roi(roi_smote)
if s_s is not None:
    colC.metric(f"ROI modelo SMOTE {s_s}", f"{v_s:.1%}")

st.divider()
st.markdown("""
**Navegación rápida**
- **Curvas**: beneficio acumulado por temporada (modelo vs Bet365).
- **Métricas**: clasificación y ROIs por temporada.
- **Matchlogs**: partidos por temporada (filtrable).
""")

st.info("Los datos de `outputs/` se actualizan automáticamente desde el motor (repo A) tras cada ronda.")
