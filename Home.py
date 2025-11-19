# Home.py
from __future__ import annotations
from pathlib import Path
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

st.markdown(
    """
    <meta name="google-site-verification" content="Ik6p2ImS4kWpFkm-1kr5yectOjj0-z2ERD7pfJNtXZ8" />
    """,
    unsafe_allow_html=True
)

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

ICON = Image.open("logo.png")
st.set_page_config(page_title="LaLiga 1X2", page_icon=ICON, layout="wide")

st.markdown(
    '<meta name="google-site-verification" content="Ik6p2ImS4kWpFkm-1kr5yectOjj0-z2ERD7pfJNtXZ8"/>',
    unsafe_allow_html=True
)

# --- Emojis en el sidebar, robusto en todas las páginas ---
def add_sidebar_icons(mapping: dict[str, str]):
    # mapping = {"Home":"🏠", "Temporada":"📅", ...}
    items_js = ",".join([f'["{k}","{v}"]' for k, v in mapping.items()])
    js_code = """
    <script>
    const mapping = new Map([{items_js}]);
    let tries = 0;
    const iv = setInterval(() => {{
      const nav = window.parent.document.querySelector('[data-testid="stSidebarNav"] ul');
      if (!nav) {{ if (++tries>20) clearInterval(iv); return; }}
      const spans = nav.querySelectorAll('li a span');
      spans.forEach(span => {{
        const label = span.textContent.trim();
        const ico = mapping.get(label);
        if (ico && !span.dataset.iconApplied) {{
          span.dataset.iconApplied = "1";
          // evita duplicar si ya tenía emoji manual
          if (!label.startsWith(ico)) {{
            span.textContent = `${{ico}} ${{label}}`;
          }}
        }}
      }});
      if (++tries>20) clearInterval(iv);
    }}, 300);
    </script>
    """.format(items_js=items_js)

    components.html(js_code, height=0)

# 👉 Define aquí tus iconos (los textos deben coincidir EXACTO con los nombres del sidebar)
SIDEBAR_ICONS = {
    "Home": "🏠",
    "Temporada Actual": "📅",
    "Histórico": "📈",
    "Métricas": "📊",
    "Matchlogs": "🧾",
}

add_sidebar_icons(SIDEBAR_ICONS)

# ======= Hero =======
st.title("🏟️ LaLiga 1X2")
st.caption("Predicción y análisis de resultados 1X2 para LaLiga")

with st.container():
    st.markdown(
        """
        Bienvenido/a a **LaLiga 1X2**, una web app que combina *datos históricos, cuotas de mercado* y un
        **modelo de clasificación multinomial** para analizar jornadas pasadas y estimar resultados de las próximas.
        """,
        unsafe_allow_html=False
    )

st.divider()

# ======= Qué puedes hacer aquí =======
st.header("¿Qué puedes hacer aquí?")
st.markdown(
    """
    - **Revisar en detalle** la **temporada actual** y **todas las temporadas desde 2010**.  
    - Consultar **métricas clave**: nivel de acierto (accuracy), log loss, Brier, **ROI** y **beneficio acumulado** por jornada/temporada.  
    - **Comparar** el rendimiento del **modelo** con el **benchmark de mercado**: apostar siempre a lo más probable según **Bet365**, con curvas de beneficio lado a lado.  
    - Explorar **matchlogs** con filtros por equipo, jornada y “value”.  
    - Obtener **predicciones para la próxima jornada** además de **análisis pre-partido** (tendencias recientes, forma, etc.) *(zona privada con PIN; solicita acceso)*.
    """
)

st.divider()

# ======= Cómo funciona =======
st.header("Cómo funciona")
st.markdown(
    """
    El proyecto **LaLiga 1X2** nace con el objetivo de combinar el análisis de datos y la modelización estadística
    para entender mejor cómo se comportan los resultados del fútbol y las cuotas de las casas de apuestas.

    **1️⃣ Fuentes de datos**
    Los datos se obtienen de varias fuentes complementarias: [Football-Data.co.uk](https://www.football-data.co.uk/), [Understat](https://understat.com/), [ClubElo](https://www.clubelo.com/), [Transfermarkt](https://www.transfermarkt.com/) y [FBref](https://fbref.com/). 

    **2️⃣ Preparación de los datos**
    Toda esta información pasa por un proceso de **limpieza, integración y normalización**, en el que se unifican
    nombres de equipos, se alinean temporadas, se eliminan valores ausentes y se crean decenas de **variables derivadas**.

    **3️⃣ El modelo**
    Una vez preparado el dataset, se alimenta a un **modelo de regresión logística multinomial**, entrenado
    con una ventana móvil (*walk-forward*) que utiliza varias temporadas anteriores para estimar las probabilidades
    de cada posible resultado: **p(H)** = victoria local,  **p(D)** = empate,  **p(A)** = victoria visitante.  

    El modelo aprende a partir de la relación entre el rendimiento de los equipos, sus métricas contextuales
    y el histórico de cuotas, lo que permite **detectar discrepancias entre la estimación estadística y la valoración del mercado**.

    **4️⃣ Resultados y evaluación**
    Cada jornada se evalúa mediante métricas de clasificación (**accuracy, log loss, Brier score**) y métricas económicas
    (**ROI y beneficio acumulado**).  
    El rendimiento del modelo se compara con un **benchmark de mercado** basado en apostar siempre a la opción
    más probable según Bet365.  
    Los resultados se almacenan en ficheros reproducibles y se visualizan dinámicamente en esta app.

    En definitiva, **LaLiga 1X2** pretende ofrecer una visión transparente, analítica y evolutiva de la competición,
    combinando la potencia de los datos con el rigor del modelado estadístico para entender —y medir— el valor en el fútbol.
    """
)

st.divider()

# ======= Navegación sugerida =======
st.header("Navegación")
st.markdown(
    """
    - **📅 Temporada Actual**: detalle de cada jornada completada de la temporada actual*(público)*, y sección privada con las predicciones y analisis estadísticos para la jornada siguiente*(público)*.  
    - **📈 Historico**: curvas de beneficio acumulado para todo el histórico de temporadas (modelo y mercado) *(público)*.
    - **📊 Métricas**: ROI/accuracy por temporada (modelo y mercado) *(público)*.
    - **🧾 Matchlogs**: explorador de cada uno de los partidos de todo el histórico con filtros y descargas *(público)*.     
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
      <div>Versión <b>{APP_VERSION}</b></div>
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
