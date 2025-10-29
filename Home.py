# Home.py
from __future__ import annotations
from pathlib import Path
from PIL import Image
import streamlit as st

ICON = Image.open("logo.png")
st.set_page_config(page_title="LaLiga 1X2", page_icon=ICON, layout="wide")

# ======= Hero =======
st.title("🏟️ LaLiga 1X2")
st.caption("Predicción y análisis de resultados 1X2 para LaLiga")

with st.container():
    st.markdown(
        """
        Bienvenido/a a **LaLiga 1X2**, una web app que combina *datos históricos, cuotas de mercado* y un
        **modelo de clasificación multinomial** para analizar jornadas pasadas y estimar resultados de las próximas.

        **¿Qué puedes hacer aquí?**
        - **Revisar jornadas completadas**: resultados reales, aciertos del modelo, cuotas y *profit/ROI* por partido.
        - **Explorar matchlogs** con filtros por equipo, jornada y *value* (si aplica).
        - **Ver métricas por temporada** (accuracy, logloss, brier, ROI) del **modelo** y del **mercado**.
        - **Comparar Modelo vs Bet365** con curvas de beneficio acumulado.
        - **(Privado)** Consultar **predicciones de la próxima jornada** con PIN.
        """,
        unsafe_allow_html=False
    )

st.divider()

# ======= Cómo funciona =======
st.header("Cómo funciona")
st.markdown(
    """
    - **Datos**: históricos de partidos y cuotas (por ejemplo, Bet365) + variables derivadas (*features*).
    - **Modelo**: **Logistic Regression (multinomial)** con ventana *walk-forward* (varias temporadas),
      que produce probabilidades **p(H), p(D), p(A)** por partido.
    - **Alineación robusta**: todas las tablas se indexan con una **clave estable** y por **(fecha + orden en el día)** para evitar desalineaciones.
    - **Métricas**: los CSV en `outputs/` (p. ej., `metrics_main_by_season.csv`, `matchlogs_<season>.csv`, etc.)
      alimentan cada página de la app.
    """
)

# ======= Por qué esta web =======
st.header("¿Por qué esta web y no otra?")
st.markdown(
    """
    - **Transparencia**: cada cifra visible (aciertos, ROI, beneficio) se **traza** a un fichero concreto de `outputs/`.
    - **Rigor**: solo se muestran **jornadas 100% completadas** en público. Las **predicciones futuras** quedan en el área privada.
    - **Reproducibilidad**: el flujo del *notebook* y de generación de artefactos está diseñado para dar **resultados estables**.
    - **Auditable**: las curvas de beneficio y las tablas de matchlogs permiten auditar pick por pick.
    """
)

# ======= Navegación sugerida =======
st.header("Navegación")
st.markdown(
    """
    - **🏠 Home**: resumen de temporada, KPIs y trayectoria de beneficio *(público)*.  
    - **📅 Jornadas**: detalle de cada jornada completada *(público)*.  
    - **📋 Matchlogs**: explorador con filtros y descargas *(público)*.  
    - **📊 Métricas**: ROI/accuracy por temporada (modelo y mercado) *(público)*.  
    - **🆚 Modelo vs Mercado**: comparación de curvas y KPIs *(público)*.  
    - **🧪 Análisis de Cuotas**: controles de calidad y desalineaciones *(público/privado, opcional)*.  
    - **🔒 Predicciones (Privado)**: próximas jornadas con **PIN**.
    """
)

# ======= Transparencia y uso responsable =======
with st.expander("Transparencia y uso responsable"):
    st.markdown(
        """
        - Este sitio **no es una recomendación financiera**; su objetivo es **analítico y educativo**.  
        - Los **ROI** y **beneficios** mostrados se calculan con **stake unitario** (configurable en la UI).  
        - Las **probabilidades** del modelo se muestran como **p(H), p(D), p(A)** junto con métricas de confianza
          (entropía, margen top-2), cuando están disponibles.
        """
    )

st.divider()
st.caption(
    "© LaLiga 1X2 — Área pública basada únicamente en jornadas completadas. "
    "Predicciones futuras disponibles en la pestaña privada con PIN."
)
