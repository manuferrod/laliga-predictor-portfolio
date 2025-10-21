
import streamlit as st

st.set_page_config(layout="wide")
st.title("Metodología (resumen)")

st.markdown("""
**Objetivo.** Predecir resultados 1X2 por jornada en LaLiga y comparar con el mercado (probabilidades implícitas) para evaluar si hay valor esperado positivo.

**Validación.**
- División **temporal** (train: temporadas pasadas → test: temporada futura).
- Controles de **data leakage**: p.ej., posición previa vs final, medias móviles calculadas sólo con información disponible hasta el partido, xG sujeto a disponibilidad histórica.

**Modelos probados.**
- Regresión logística, Random Forest, XGBoost.
- Mejor desempeño observado (test): *XGBoost* (aprox. acc≈0.56; logloss≈0.94).

**Evaluación económica (simulada).**
- Comparación contra probabilidades implícitas del mercado.
- ROI **sólo informativo**. *No es consejo de apuesta.*

**Privacidad y alcance.**
- Esta demo **no** entrena modelos ni expone scrapers/datos de terceros.
- Se muestran predicciones **precomputadas** y métricas **agregadas**.
- Uso permitido: evaluación académica/profesional. Uso comercial prohibido sin autorización.

**Contacto.**
- Autor: Manuel Fernández · LinkedIn/Email según README del repositorio.
""")
