# Home.py
from __future__ import annotations
from pathlib import Path
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# --------- Metadatos del pie ---------
CONTACT_EMAIL = "manuelfernandezrodriguez1@gmail.com"
PAYPAL_URL    = "https://paypal.me/LaLiga1x2"
LAST_UPDATE   = "Octubre 29, 2025"
DATA_SOURCES = {
    "Football-Data.co.uk": "https://www.football-data.co.uk/",
    "Understat": "https://understat.com/",
    "ClubElo": "https://www.clubelo.com/",
    "Transfermarkt": "https://www.transfermarkt.com/",
    "FBref": "https://fbref.com/"
}
APP_VERSION   = "1.0.0"

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

# ======= Caja de soporte / contacto (HTML estable en iframe) =======
st.divider()

sources_html = " / ".join(
    f'<a href="{url}" target="_blank">{name}</a>'
    for name, url in DATA_SOURCES.items()
)

import streamlit.components.v1 as components

components.html(
    f"""
<div class="llx2-support">
  <div class="box">
    <p class="title">¿Te resulta útil LaLiga 1X2?</p>
    <p class="text">
      Esta app es <b>gratuita</b>. Si te ha ayudado o añadido valor a tu trabajo, puedes
      apoyar el proyecto con una donación. Tu contribución me ayuda a seguir mejorándola. 🙌
    </p>

    <div class="actions">
      <a class="btn" href="{PAYPAL_URL}" target="_blank">💙 Apoyar en PayPal</a>
      <button class="btn" onclick="
        navigator.clipboard.writeText('{CONTACT_EMAIL}');
        this.innerText='✔ Copiado';
        setTimeout(()=>this.innerText='✉️ Copiar email',1500);
      ">✉️ Copiar email</button>
    </div>

    <p class="text mt">
      <b>Predicciones futuras (zona privada):</b> si deseas acceso, copia mi correo y
      <b>contacta conmigo</b> para que te indique los pasos.
    </p>

    <div class="meta">
      <div>📅 Datos actualizados: <b>{LAST_UPDATE}</b> · Fuentes: {sources_html}</div>
      <div>💙 <a href="{PAYPAL_URL}" target="_blank">Apoyar en PayPal</a> · Versión <b>{APP_VERSION}</b></div>
    </div>
  </div>
</div>

<style>
  .llx2-support .box {{
    padding: 1.1rem 1.25rem;
    border-radius: 16px;
    border: 1px solid rgba(120,120,120,.25);
    background: rgba(30, 100, 160, .10);
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, "Helvetica Neue", Arial, "Apple Color Emoji","Segoe UI Emoji";
  }}
  .llx2-support .title {{ font-size: 1.05rem; font-weight: 600; margin: 0 0 .5rem 0; }}
  .llx2-support .text {{ margin: 0; line-height: 1.6; }}
  .llx2-support .mt {{ margin-top: .85rem; }}
  .llx2-support .actions {{ margin-top: .85rem; display: flex; gap: .9rem; flex-wrap: wrap; }}
  .llx2-support .btn {{
    text-decoration: none; padding: .55rem .9rem; border-radius: 999px;
    border: 1px solid rgba(120,120,120,.35); background: transparent; cursor: pointer;
  }}
  .llx2-support .btn:hover {{ filter: brightness(1.05); }}
  .llx2-support .meta {{
    display: flex; justify-content: space-between; align-items: center;
    gap: .75rem; margin-top: 1.1rem; padding-top: 1.1rem;
    border-top: 1px solid rgba(120,120,120,.25);
    font-size: .93rem; flex-wrap: wrap;
  }}
  .llx2-support .meta a {{ text-decoration: none; }}
  @media (max-width: 700px) {{
    .llx2-support .meta {{ flex-direction: column; align-items: flex-start; }}
  }}
</style>
    """,
    height=320,
)
# ======= /Caja soporte =======

st.divider()
