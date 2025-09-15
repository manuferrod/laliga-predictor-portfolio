
"""
Script de utilidad para transformar un CSV/Parquet de predicciones a JSON para la app.
Espera columnas: date, season, match_id, home_team, away_team, p_home, p_draw, p_away, predicted_result.
"""
import pandas as pd
import argparse, json
from datetime import datetime, timezone

def main(in_path: str, out_path: str, model_version: str):
    if in_path.endswith(".parquet"):
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    required = ["date","season","match_id","home_team","away_team","p_home","p_draw","p_away","predicted_result"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df = df[required].copy()
    df["as_of"] = datetime.now(timezone.utc).isoformat()
    df["model_version"] = model_version

    records = df.to_dict(orient="records")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="CSV/Parquet con predicciones")
    parser.add_argument("--out", dest="out_path", required=True, help="Ruta de salida JSON")
    parser.add_argument("--version", dest="model_version", required=True, help="Etiqueta del modelo, p.ej. xgb-2025.09.10")
    args = parser.parse_args()
    main(args.in_path, args.out_path, args.model_version)
