# Metricas.py
from __future__ import annotations

import sys, importlib.util
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# --- import robusto del módulo scripts/io ---
ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    # Preferido
    from scripts.io import (
        ensure_outputs_dir,
        has_outputs,
        load_csv,
        load_roi,        # ya normaliza Season y ROI
        BASE as OUTPUTS_BASE,
    )
except Exception:
    # Fallback defensivo
    spec = importlib.util.spec_from_file_location("io", SCRIPTS_DIR / "io.py")
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)  # type: ignore
    ensure_outputs_dir = io.ensure_outputs_dir
    has_outputs        = io.has_outputs
    load_csv           = io.load_csv
    load_roi           = io.load_roi
    OUTPUTS_BASE       = io.BASE

st.set_page_config(page_title="Métricas", page_icon="📊", layout="wide")
st.header("Métricas y ROI por temporada")

# 0) Chequeos básicos
ensure_outputs_dir()
if not has_outputs():
    st.info("Aún no hay artefactos en `outputs/`.")
    st.stop()

# 1) Cargar ROI de los modelos (usa io.load_roi que ya normaliza)
roi_base  = load_roi("base")   # -> columnas normalizadas: Season, ROI (si existen)
roi_smote = load_roi("smote")  # idem

def _load_bet365_metrics() -> pd.DataFrame:
    """Lee outputs/bet365_metrics_by_season.csv y normaliza a Season/ROI si es posible."""
    p = OUTPUTS_BASE / "bet365_metrics_by_season.csv"
    if not p.exists():
        return pd.DataFrame()
    df = load_csv("bet365_metrics_by_season.csv")
    # Normaliza Season
    if "Season" not in df.columns:
        for c in df.columns:
            if str(c).lower() in {"test_season", "season"}:
                df = df.rename(columns={c: "Season"})
                break
    # Normaliza ROI
    if "ROI" not in df.columns:
        # suele venir como 'roi'
        for c in df.columns:
            if str(c).lower().startswith("roi"):
                df = df.rename(columns={c: "ROI"})
                break
    # Tipos
    if "Season" in df.columns:
        df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")
    if "ROI" in df.columns:
        df["ROI"] = pd.to_numeric(df["ROI"], errors="coerce")
    return df

roi_b365 = _load_bet365_metrics()

# 2) Unificar para gráfico (BASE / SMOTE / Bet365 si existe)
series = []
if not roi_base.empty and {"Season","ROI"}.issubset(roi_base.columns):
    tmp = roi_base[["Season","ROI"]].copy()
    tmp["Modelo"] = "BASE"
    series.append(tmp)
if not roi_smote.empty and {"Season","ROI"}.issubset(roi_smote.columns):
    tmp = roi_smote[["Season","ROI"]].copy()
    tmp["Modelo"] = "SMOTE"
    series.append(tmp)
if not roi_b365.empty and {"Season","ROI"}.issubset(roi_b365.columns):
    tmp = roi_b365[["Season","ROI"]].copy()
    tmp["Modelo"] = "Bet365"
    series.append(tmp)

if not series:
    st.info("No encontré ROI por temporada (ni BASE, ni SMOTE, ni Bet365).")
    st.stop()

plot_df = pd.concat(series, ignore_index=True).dropna(subset=["Season"]).sort_values(["Season","Modelo"])
plot_df["Season"] = plot_df["Season"].astype(int)

# 3) Gráfico principal (ROI por temporada)
c1, c2 = st.columns([3,2])
with c1:
    fig = px.bar(
        plot_df,
        x="Season",
        y="ROI",
        color="Modelo",
        barmode="group",
        title="ROI por temporada (Modelo vs Bet365)",
    )
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # Tabla rápida (pivot para lectura)
    pivot = (plot_df
             .pivot_table(index="Season", columns="Modelo", values="ROI", aggfunc="first")
             .sort_index())
    # Formateo amigable
    st.subheader("Tabla ROI (%)")
    st.dataframe(
        (pivot * 100).round(2).rename_axis(None, axis=1),
        use_container_width=True,
    )
    # Descarga CSV combinado
    st.download_button(
        "Descargar ROI combinado (CSV)",
        data=plot_df.to_csv(index=False).encode("utf-8"),
        file_name="roi_combinado.csv",
        mime="text/csv",
    )

st.divider()

# 4) (Opcional) Métricas de clasificación si existen
def _load_class_by_season(tag: str) -> pd.DataFrame:
    p = OUTPUTS_BASE / f"classification_by_season_{tag}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = load_csv(f"classification_by_season_{tag}.csv")
    # Normaliza Season
    if "Season" not in df.columns:
        for c in df.columns:
            if str(c).lower() in {"season", "test_season"}:
                df = df.rename(columns={c: "Season"})
                break
    for m in ["accuracy","acc","Accuracy"]:
        if m in df.columns:
            df = df.rename(columns={m: "Accuracy"})
            break
    for m in ["log_loss","logloss","LogLoss","Log_Loss"]:
        if m in df.columns:
            df = df.rename(columns={m: "LogLoss"})
            break
    for m in ["brier","Brier","brier_score"]:
        if m in df.columns:
            df = df.rename(columns={m: "Brier"})
            break
    # Tipos
    if "Season" in df.columns:
        df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")
    for c in ["Accuracy","LogLoss","Brier"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

cls_base  = _load_class_by_season("base")
cls_smote = _load_class_by_season("smote")

if not cls_base.empty or not cls_smote.empty:
    st.subheader("Métricas de clasificación por temporada")
    tabs = st.tabs(["Accuracy", "LogLoss", "Brier"])
    for metric, tab in zip(["Accuracy","LogLoss","Brier"], tabs):
        with tab:
            parts = []
            if metric in cls_base.columns:
                a = cls_base[["Season", metric]].copy(); a["Modelo"] = "BASE"; parts.append(a)
            if metric in cls_smote.columns:
                b = cls_smote[["Season", metric]].copy(); b["Modelo"] = "SMOTE"; parts.append(b)
            if parts:
                dd = pd.concat(parts, ignore_index=True).dropna(subset=["Season", metric])
                fig2 = px.line(dd.sort_values(["Modelo","Season"]), x="Season", y=metric, color="Modelo",
                               markers=True, title=metric)
                if metric == "Accuracy":
                    fig2.update_yaxes(tickformat=".1%")
                st.plotly_chart(fig2, use_container_width=True)
                with st.expander(f"Ver datos · {metric}"):
                    st.dataframe(dd.sort_values(["Season","Modelo"]), use_container_width=True, hide_index=True)
            else:
                st.info(f"No encontré '{metric}' en los CSV de clasificación.")
