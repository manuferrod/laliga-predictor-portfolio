
# Predicción 1X2 en LaLiga — Demo pública (Portfolio)

**Qué es:** una app de Streamlit que muestra **predicciones precomputadas** para la próxima jornada y **métricas agregadas** por temporada. No incluye scraping, entrenamiento ni datos de terceros.

**Por qué importa:** demuestra un flujo ML aplicado a negocio (predicción por partido, validación temporal y comparación con mercado/ROI) **sin** exponer propiedad intelectual.

## Estructura del repo
```
.
├─ Home.py                 # archivo principal Streamlit (multi-página)
├─ pages/
│   ├─ 01_Historico.py
│   └─ 02_Metodología.py
├─ outputs/
│   ├─ predictions_next.json      # predicciones próximas (match-level)
│   └─ historical_metrics.csv     # métricas agregadas por temporada
├─ scripts/
│   └─ make_predictions_json.py   # util para generar el JSON de la app
├─ .streamlit/config.toml         # tema de la app
├─ MODEL_CARD.md
├─ DATA_CARD.md
├─ LICENSE
└─ requirements.txt
```

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run Home.py
```

## Despliegue (Streamlit Community Cloud)
1. Sube este repo a GitHub (público).
2. Crea una app en Streamlit Cloud y selecciona `Home.py` como archivo principal.
3. Comprueba que la app carga `outputs/predictions_next.json` y `outputs/historical_metrics.csv`.

## Actualizar predicciones por jornada (manual, simple)
1. Exporta un CSV/parquet desde tus notebooks con columnas:  
   `date, season, match_id, home_team, away_team, p_home, p_draw, p_away, predicted_result`
2. Genera el JSON para la app:
   ```bash
   python scripts/make_predictions_json.py --in path/a/tus_preds.csv --out outputs/predictions_next.json --version xgb-YYYY.MM.DD
   ```
3. Sube el commit a GitHub → la app se actualiza.

## Privacidad y licencia
- Este repositorio **no** redistribuye datos de terceros ni artefactos de entrenamiento.
- Se publica con licencia restrictiva (ver `LICENSE`): uso **no comercial** y sin obras derivadas salvo permiso expreso.
- El PDF completo del TFM puede compartirse con marca de agua y licencia **CC BY-NC-ND 4.0** (recomendado).

## Métricas de referencia (test)
Accuracy ≈ **0.56**, Log loss ≈ **0.94**. Las cifras exactas y metodología se describen en la página *Metodología*.

---
© Manuel Fernández, 2025 — Demo con fines de evaluación académica/profesional.
