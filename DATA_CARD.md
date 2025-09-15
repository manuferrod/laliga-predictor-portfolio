
# Data Card — (Demo)

**Origen de datos.** No se redistribuyen datos de terceros. La app muestra únicamente predicciones precomputadas y métricas agregadas.

**Cobertura.** Temporadas históricas de LaLiga (2014–2025 aprox.) para entrenamiento/validación en el proyecto original (no expuesto aquí).

**Campos en `outputs/predictions_next.json`.**
- `date` (YYYY-MM-DD), `season` (p. ej., `2025_26`), `match_id` (ID estable),
- `home_team`, `away_team`,
- `p_home`, `p_draw`, `p_away` (probabilidades),
- `predicted_result` (`H`/`D`/`A`),
- `model_version` (etiqueta del modelo),
- `as_of` (timestamp de generación).

**Campos en `outputs/historical_metrics.csv`.**
- `season`, `acc`, `logloss`, `roi`, `coverage`, `n_games`, `updated_at`.

**Licencias/Restricciones.** Uso no comercial. No se incluye ni se habilita descarga de datos de terceros.
