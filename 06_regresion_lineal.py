# -*- coding: utf-8 -*-
"""Regresion lineal multiple con validacion cruzada."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(__file__).resolve().parent
ARCHIVO_DATOS = BASE / "data" / "insurance.csv"
CARPETA_MODELOS = BASE / "modelos"
CARPETA_RESULTADOS = BASE / "resultados"

NUMERICAS = ["age", "bmi", "children"]
CATEGORICAS = ["sex", "smoker", "region"]
VARIABLES = NUMERICAS + CATEGORICAS
OBJETIVO = "charges"
SEMILLA = 42
PARTICIONES = 5


def cargar_datos() -> tuple[pd.DataFrame, pd.Series]:
    if not ARCHIVO_DATOS.is_file():
        raise FileNotFoundError(f"No se encontro el dataset: {ARCHIVO_DATOS}")

    datos = pd.read_csv(ARCHIVO_DATOS)
    requeridas = VARIABLES + [OBJETIVO]
    faltantes = sorted(set(requeridas) - set(datos.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {faltantes}")

    datos = datos[requeridas].dropna().drop_duplicates().copy()
    for columna in NUMERICAS + [OBJETIVO]:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")
    datos = datos.dropna()

    print("=== DATOS PARA REGRESION LINEAL ===")
    print(f"Registros utilizados: {len(datos)}")
    print(f"Variables predictoras: {VARIABLES}")
    print(f"Variable objetivo: {OBJETIVO}")
    return datos[VARIABLES], datos[OBJETIVO]


def crear_pipeline() -> Pipeline:
    preprocesamiento = ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), NUMERICAS),
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAS,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento),
            ("regresor", LinearRegression()),
        ]
    )


def ejecutar() -> None:
    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    x, y = cargar_datos()
    x_entrena, x_prueba, y_entrena, y_prueba = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=SEMILLA,
    )

    print("\n=== DIVISION DEL DATASET ===")
    print(f"Entrenamiento: {len(x_entrena)} registros (80 %)")
    print(f"Prueba: {len(x_prueba)} registros (20 %)")

    pipeline = crear_pipeline()
    validacion = KFold(
        n_splits=PARTICIONES,
        shuffle=True,
        random_state=SEMILLA,
    )
    puntuaciones = cross_validate(
        pipeline,
        x_entrena,
        y_entrena,
        cv=validacion,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
        },
        return_train_score=False,
        n_jobs=1,
    )

    resultados_cv = pd.DataFrame(
        {
            "fold": np.arange(1, PARTICIONES + 1),
            "r2": puntuaciones["test_r2"],
            "mae": -puntuaciones["test_mae"],
            "mse": -puntuaciones["test_mse"],
        }
    )
    resultados_cv["rmse"] = np.sqrt(resultados_cv["mse"])

    print("\n=== VALIDACION CRUZADA KFOLD ===")
    print(resultados_cv.round(4).to_string(index=False))
    for metrica in ["r2", "mae", "rmse"]:
        print(
            f"{metrica}: {resultados_cv[metrica].mean():.4f} "
            f"+/- {resultados_cv[metrica].std(ddof=1):.4f}"
        )

    pipeline.fit(x_entrena, y_entrena)
    predicciones = pipeline.predict(x_prueba)
    mse = mean_squared_error(y_prueba, predicciones)
    metricas_prueba = {
        "r2": float(r2_score(y_prueba, predicciones)),
        "mae": float(mean_absolute_error(y_prueba, predicciones)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
    }

    print("\n=== PRUEBA INDEPENDIENTE ===")
    for nombre, valor in metricas_prueba.items():
        if nombre == "r2":
            print(f"{nombre}: {valor:.4f} ({valor * 100:.2f} % de variacion explicada)")
        else:
            print(f"{nombre}: ${valor:,.2f}")

    tabla_predicciones = x_prueba.reset_index(drop=True).copy()
    tabla_predicciones["charges_real"] = y_prueba.reset_index(drop=True)
    tabla_predicciones["charges_predicho"] = predicciones
    tabla_predicciones["error"] = (
        tabla_predicciones["charges_real"] - tabla_predicciones["charges_predicho"]
    )

    fig, ejes = plt.subplots(1, 2, figsize=(12, 5))
    ejes[0].scatter(y_prueba, predicciones, alpha=0.65, color="#1f77b4")
    limite_min = min(float(y_prueba.min()), float(predicciones.min()))
    limite_max = max(float(y_prueba.max()), float(predicciones.max()))
    ejes[0].plot([limite_min, limite_max], [limite_min, limite_max], "r--")
    ejes[0].set_title("Valores reales contra predichos")
    ejes[0].set_xlabel("Charges reales")
    ejes[0].set_ylabel("Charges predichos")

    residuos = y_prueba.to_numpy() - predicciones
    ejes[1].scatter(predicciones, residuos, alpha=0.65, color="#2ca02c")
    ejes[1].axhline(0, color="red", linestyle="--")
    ejes[1].set_title("Grafica de residuos")
    ejes[1].set_xlabel("Charges predichos")
    ejes[1].set_ylabel("Error real - predicho")
    fig.suptitle("Evaluacion de la regresion lineal")
    fig.tight_layout()
    fig.savefig(CARPETA_RESULTADOS / "evaluacion_regresion_lineal.png", dpi=170)
    plt.close(fig)

    resumen = {
        "modelo": "Regresion lineal multiple",
        "objetivo": OBJETIVO,
        "variables": VARIABLES,
        "registros_entrenamiento": int(len(x_entrena)),
        "registros_prueba": int(len(x_prueba)),
        "cv_particiones": PARTICIONES,
        "cv_r2_promedio": float(resultados_cv["r2"].mean()),
        "cv_r2_desviacion": float(resultados_cv["r2"].std(ddof=1)),
        "cv_mae_promedio": float(resultados_cv["mae"].mean()),
        "cv_rmse_promedio": float(resultados_cv["rmse"].mean()),
        **{f"test_{clave}": valor for clave, valor in metricas_prueba.items()},
    }

    joblib.dump(pipeline, CARPETA_MODELOS / "pipeline_regresion_lineal.joblib")
    resultados_cv.to_csv(
        CARPETA_RESULTADOS / "cv_regresion_lineal.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tabla_predicciones.to_csv(
        CARPETA_RESULTADOS / "predicciones_regresion_lineal.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (CARPETA_RESULTADOS / "metricas_regresion_lineal.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nArchivos generados:")
    print("- modelos/pipeline_regresion_lineal.joblib")
    print("- resultados/cv_regresion_lineal.csv")
    print("- resultados/predicciones_regresion_lineal.csv")
    print("- resultados/metricas_regresion_lineal.json")
    print("- resultados/evaluacion_regresion_lineal.png")


if __name__ == "__main__":
    ejecutar()
