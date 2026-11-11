# -*- coding: utf-8 -*-
"""Carga el modelo K-means guardado y prueba costos nuevos."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
ARCHIVO_PIPELINE = BASE / "modelos" / "pipeline_kmeans.joblib"
ARCHIVO_MAPA = BASE / "modelos" / "mapa_niveles_kmeans.joblib"
ARCHIVO_SALIDA = BASE / "resultados" / "prueba_kmeans.csv"


def ejecutar_prueba() -> None:
    if not ARCHIVO_PIPELINE.is_file() or not ARCHIVO_MAPA.is_file():
        raise FileNotFoundError(
            "No se encontraron los activos de K-means. Ejecuta primero 01_kmeans.py."
        )

    pipeline = joblib.load(ARCHIVO_PIPELINE)
    mapa_niveles = joblib.load(ARCHIVO_MAPA)

    casos = pd.DataFrame(
        {"charges": [2500.0, 8500.0, 16000.0, 26000.0, 42000.0]}
    )
    clusters = pipeline.predict(casos)
    distancias = pipeline.transform(casos)

    resultados = casos.copy()
    resultados["cluster_predicho"] = clusters
    resultados["nivel_predicho"] = [mapa_niveles[int(valor)] for valor in clusters]
    resultados["distancia_al_centroide"] = np.min(distancias, axis=1)

    ARCHIVO_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8-sig")

    print("=== PRUEBA DEL MODELO K-MEANS GUARDADO ===")
    print(resultados.round(4).to_string(index=False))
    print(f"\nResultados guardados en: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    ejecutar_prueba()
