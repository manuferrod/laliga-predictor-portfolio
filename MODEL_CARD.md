
# Model Card — Predicción 1X2 LaLiga (Demo pública)

**Propósito.** Demostración académica y profesional de un sistema de predicción 1X2 por jornada con validación temporal y evaluación económica (ROI simulado).

**Entradas.** No se exponen datos brutos; la app consume predicciones precomputadas (match-level) y métricas agregadas por temporada.

**Validación.** Split temporal (train en temporadas pasadas, test en temporada futura). Controles de *data leakage* (p. ej., posición previa vs final, medias móviles sólo con información pasada, disponibilidad histórica de xG).

**Modelos.** LR, RF, XGBoost; mejor desempeño observado en test con XGBoost (acc≈0.56; logloss≈0.94).

**Limitaciones.** Sesgo por cambios de mercado, disponibilidad de xG, y posibles desequilibrios de clase. El ROI es simulado y no implica consejo de apuesta.

**Uso permitido.** Evaluación académica/profesional.

**Uso prohibido.** Uso comercial, redistribución de datos o artefactos de entrenamiento, obras derivadas sin autorización.

**Contacto.** Manuel Fernández — LinkedIn/Email en README.
