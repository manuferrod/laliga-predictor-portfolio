# pages/01_Historico.py
from __future__ import annotations

# --- importar el módulo scripts/io de forma robusta ---
import sys, importlib.util
from pathlib import Path
import streamlit as st
import plotly.express as px
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    # Preferido (cuando la app corre normal)
    from scripts.io import (
        ensure_outputs_dir,
        has_outputs,
        available_seasons,
        load_cumprofit,
    )
except Exception:
    # Fallback defensivo (por si falla el import relativo)
    spec = importlib.util.spec_from_file_location("io", SCRIPTS_DIR / "io.py")
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)  # type: ignore
    ensure_outputs_dir = io.ensure_outputs_dir
    has_outputs        = io.has_outputs
    available_seasons  = io.available_seasons
    load_cumprofit     = io.load_cumprofit

st.set_page_config(page_title="Curvas", page_icon="📈")
st.header("Curvas de beneficio acumulado")

# 0) Comprobaciones básicas de outputs/
ensure_outputs_dir()
if not has_outputs():
    st.warning("No se encontraron artefactos en `outputs/`. "
               "Sube los ficheros generados por el motor o espera a la sincronización.")
    st.stop()

# 1) Temporadas disponibles (desde outputs/)
try:
    seas = available_seasons()
except Exception as e:
    st.error(f"No pude inferir temporadas desde outputs: {e}")
    st.stop()

if not seas:
    st.warning("No hay temporadas detectadas en `outputs/` todavía.")
    st.stop()

sel = st.selectbox("Temporada", seas, index=len(seas) - 1)

# 2) Cargar curva (desde outputs/cumprofit_curves/cumprofit_<TEMP>.json|csv)
df = load_cumprofit(sel)
if df is None or df.empty:
    st.info(
        f"No encontré series para graficar en "
        f"`outputs/cumprofit_curves/cumprofit_{sel}.json|csv`."
    )
    st.stop()

# Asegura columna x (por si algún loader devolviera otro nombre)
df.columns = [str(c).strip() for c in df.columns]
if "x" not in df.columns:
    # fallback muy defensivo
    for cand in ("match_num", "index", "i", "step", "round", "n"):
        if cand in df.columns:
            df = df.rename(columns={cand: "x"})
            break
    else:
        df.insert(0, "x", range(1, len(df) + 1))

series_cols = [c for c in df.columns if c != "x"]
if not series_cols:
    st.warning(
        "El archivo de curvas no contiene ninguna serie ('Model (BASE)', "
        "'Model (SMOTE)', 'Bet365'). Revisa el JSON/CSV."
    )
    st.dataframe(df.head())
    st.stop()

# 3) Plot
long = df.melt(id_vars="x", var_name="Serie", value_name="Beneficio")
fig = px.line(long, x="x", y="Beneficio", color="Serie",
              title=f"Beneficio acumulado — Temporada {sel}")
fig.update_layout(legend_title_text="")
st.plotly_chart(fig, use_container_width=True)

# (opcional) muestra de datos
with st.expander("Ver datos"):
    st.dataframe(df, use_container_width=True, hide_index=True)
