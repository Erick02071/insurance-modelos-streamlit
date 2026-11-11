# -*- coding: utf-8 -*-
"""Arbol de decision para regresion con validacion cruzada."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree


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

    print("=== DATOS PARA ARBOL DE DECISION ===")
    print(f"Registros utilizados: {len(datos)}")
    print(f"Variables predictoras: {VARIABLES}")
    print(f"Variable objetivo: {OBJETIVO}")
    return datos[VARIABLES], datos[OBJETIVO]


def crear_pipeline() -> Pipeline:
    preprocesamiento = ColumnTransformer(
        transformers=[
            ("numericas", "passthrough", NUMERICAS),
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAS,
            ),
        ]
    )
    regresor = DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=5,
        random_state=SEMILLA,
    )
    return Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento),
            ("regresor", regresor),
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
    print(
        f"r2: {metricas_prueba['r2']:.4f} "
        f"({metricas_prueba['r2'] * 100:.2f} % de variacion explicada)"
    )
    for nombre in ["mae", "mse", "rmse"]:
        print(f"{nombre}: ${metricas_prueba[nombre]:,.2f}")

    tabla_predicciones = x_prueba.reset_index(drop=True).copy()
    tabla_predicciones["charges_real"] = y_prueba.reset_index(drop=True)
    tabla_predicciones["charges_predicho"] = predicciones
    tabla_predicciones["error"] = (
        tabla_predicciones["charges_real"] - tabla_predicciones["charges_predicho"]
    )

    preprocesamiento = pipeline.named_steps["preprocesamiento"]
    arbol = pipeline.named_steps["regresor"]
    nombres_variables = preprocesamiento.get_feature_names_out()
    importancias = pd.DataFrame(
        {
            "variable": nombres_variables,
            "importancia": arbol.feature_importances_,
        }
    ).sort_values("importancia", ascending=False)

    fig, ejes = plt.subplots(1, 2, figsize=(12, 5))
    ejes[0].scatter(y_prueba, predicciones, alpha=0.65, color="#ff7f0e")
    limite_min = min(float(y_prueba.min()), float(predicciones.min()))
    limite_max = max(float(y_prueba.max()), float(predicciones.max()))
    ejes[0].plot([limite_min, limite_max], [limite_min, limite_max], "r--")
    ejes[0].set_title("Valores reales contra predichos")
    ejes[0].set_xlabel("Charges reales")
    ejes[0].set_ylabel("Charges predichos")

    principales = importancias.head(10).sort_values("importancia")
    ejes[1].barh(principales["variable"], principales["importancia"], color="#2ca02c")
    ejes[1].set_title("Importancia de variables")
    ejes[1].set_xlabel("Importancia")
    fig.suptitle("Evaluacion del arbol de decision")
    fig.tight_layout()
    fig.savefig(CARPETA_RESULTADOS / "evaluacion_arbol_decision.png", dpi=170)
    plt.close(fig)

    plt.figure(figsize=(18, 9))
    plot_tree(
        arbol,
        feature_names=nombres_variables,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=7,
    )
    plt.title("Arbol de decision (primeros cuatro niveles)")
    plt.tight_layout()
    plt.savefig(CARPETA_RESULTADOS / "estructura_arbol_decision.png", dpi=170)
    plt.close()

    resumen = {
        "modelo": "Arbol de decision para regresion",
        "objetivo": OBJETIVO,
        "variables": VARIABLES,
        "parametros": {"max_depth": 5, "min_samples_leaf": 5},
        "registros_entrenamiento": int(len(x_entrena)),
        "registros_prueba": int(len(x_prueba)),
        "cv_particiones": PARTICIONES,
        "cv_r2_promedio": float(resultados_cv["r2"].mean()),
        "cv_r2_desviacion": float(resultados_cv["r2"].std(ddof=1)),
        "cv_mae_promedio": float(resultados_cv["mae"].mean()),
        "cv_rmse_promedio": float(resultados_cv["rmse"].mean()),
        **{f"test_{clave}": valor for clave, valor in metricas_prueba.items()},
    }

    joblib.dump(pipeline, CARPETA_MODELOS / "pipeline_arbol_decision.joblib")
    resultados_cv.to_csv(
        CARPETA_RESULTADOS / "cv_arbol_decision.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tabla_predicciones.to_csv(
        CARPETA_RESULTADOS / "predicciones_arbol_decision.csv",
        index=False,
        encoding="utf-8-sig",
    )
    importancias.to_csv(
        CARPETA_RESULTADOS / "importancias_arbol_decision.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (CARPETA_RESULTADOS / "metricas_arbol_decision.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nArchivos generados:")
    print("- modelos/pipeline_arbol_decision.joblib")
    print("- resultados/cv_arbol_decision.csv")
    print("- resultados/predicciones_arbol_decision.csv")
    print("- resultados/importancias_arbol_decision.csv")
    print("- resultados/metricas_arbol_decision.json")
    print("- resultados/evaluacion_arbol_decision.png")
    print("- resultados/estructura_arbol_decision.png")


if __name__ == "__main__":
    ejecutar()
