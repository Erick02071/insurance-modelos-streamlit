# -*- coding: utf-8 -*-
"""Naive Bayes multiclase con validacion cruzada para predecir level."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(__file__).resolve().parent
ARCHIVO_DATOS = BASE / "resultados" / "insurance_with_levels.csv"
CARPETA_MODELOS = BASE / "modelos"
CARPETA_RESULTADOS = BASE / "resultados"
ARCHIVO_CODIFICADOR = CARPETA_MODELOS / "label_encoder_clasificacion.joblib"

NUMERICAS = ["age", "bmi", "children"]
CATEGORICAS = ["sex", "smoker", "region"]
VARIABLES = NUMERICAS + CATEGORICAS
OBJETIVO = "level"
SEMILLA = 42
PARTICIONES = 5


def cargar_datos() -> tuple[pd.DataFrame, pd.Series]:
    if not ARCHIVO_DATOS.is_file():
        raise FileNotFoundError(
            f"No se encontro {ARCHIVO_DATOS}. Ejecuta primero 01_kmeans.py."
        )
    if not ARCHIVO_CODIFICADOR.is_file():
        raise FileNotFoundError(
            "No se encontro el LabelEncoder. Ejecuta primero 02_regresion_logistica.py."
        )

    datos = pd.read_csv(ARCHIVO_DATOS)
    requeridas = VARIABLES + [OBJETIVO]
    faltantes = sorted(set(requeridas) - set(datos.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {faltantes}")

    datos = datos[requeridas].dropna().drop_duplicates().copy()
    datos[OBJETIVO] = datos[OBJETIVO].astype(str).str.strip().str.lower()
    codificador = joblib.load(ARCHIVO_CODIFICADOR)
    if set(codificador.classes_) != set(datos[OBJETIVO].unique()):
        raise ValueError("El LabelEncoder no coincide con las clases del dataset.")

    print("=== DATOS PARA NAIVE BAYES ===")
    print(f"Registros utilizados: {len(datos)}")
    print(f"Variables predictoras: {VARIABLES}")
    print("Variable objetivo: level")
    print("Se excluye charges para evitar fuga de informacion.")
    print("\nDistribucion de clases:")
    print(datos[OBJETIVO].value_counts().to_string())
    return datos[VARIABLES], datos[OBJETIVO]


def crear_pipeline() -> Pipeline:
    preprocesamiento = ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), NUMERICAS),
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAS,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento),
            ("clasificador", GaussianNB()),
        ]
    )


def ejecutar() -> None:
    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    x, y_texto = cargar_datos()
    codificador = joblib.load(ARCHIVO_CODIFICADOR)
    y = codificador.transform(y_texto)

    x_entrena, x_prueba, y_entrena, y_prueba = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=SEMILLA,
        stratify=y,
    )
    print("\n=== DIVISION DEL DATASET ===")
    print(f"Entrenamiento: {len(x_entrena)} registros (80 %)")
    print(f"Prueba: {len(x_prueba)} registros (20 %)")
    print("Modelo: Gaussian Naive Bayes")

    pipeline = crear_pipeline()
    validacion = StratifiedKFold(
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
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
        },
        return_train_score=False,
        n_jobs=1,
    )
    resultados_cv = pd.DataFrame(
        {
            "fold": np.arange(1, PARTICIONES + 1),
            "accuracy": puntuaciones["test_accuracy"],
            "balanced_accuracy": puntuaciones["test_balanced_accuracy"],
            "f1_macro": puntuaciones["test_f1_macro"],
        }
    )

    print("\n=== VALIDACION CRUZADA ESTRATIFICADA ===")
    print(resultados_cv.round(4).to_string(index=False))
    for metrica in ["accuracy", "balanced_accuracy", "f1_macro"]:
        print(
            f"{metrica}: {resultados_cv[metrica].mean():.4f} "
            f"+/- {resultados_cv[metrica].std(ddof=1):.4f}"
        )

    pipeline.fit(x_entrena, y_entrena)
    predicciones = pipeline.predict(x_prueba)
    metricas_prueba = {
        "accuracy": float(accuracy_score(y_prueba, predicciones)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_prueba, predicciones)
        ),
        "precision_macro": float(
            precision_score(y_prueba, predicciones, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_prueba, predicciones, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_prueba, predicciones, average="macro", zero_division=0)
        ),
    }

    print("\n=== PRUEBA INDEPENDIENTE ===")
    for nombre, valor in metricas_prueba.items():
        print(f"{nombre}: {valor:.4f} ({valor * 100:.2f} %)")

    etiquetas = np.arange(len(codificador.classes_))
    reporte = classification_report(
        y_prueba,
        predicciones,
        labels=etiquetas,
        target_names=codificador.classes_,
        output_dict=True,
        zero_division=0,
    )
    reporte_tabla = pd.DataFrame(reporte).transpose()
    print("\nReporte de clasificacion:")
    print(reporte_tabla.round(4).to_string())

    matriz = confusion_matrix(y_prueba, predicciones, labels=etiquetas)
    visualizacion = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=codificador.classes_,
    )
    visualizacion.plot(cmap="Oranges", values_format="d")
    plt.title("Matriz de confusion - Gaussian Naive Bayes")
    plt.tight_layout()
    plt.savefig(CARPETA_RESULTADOS / "matriz_confusion_naive_bayes.png", dpi=170)
    plt.close()

    resumen = {
        "modelo": "Gaussian Naive Bayes",
        "objetivo": OBJETIVO,
        "variables": VARIABLES,
        "clases": codificador.classes_.tolist(),
        "registros_entrenamiento": int(len(x_entrena)),
        "registros_prueba": int(len(x_prueba)),
        "cv_particiones": PARTICIONES,
        "cv_accuracy_promedio": float(resultados_cv["accuracy"].mean()),
        "cv_accuracy_desviacion": float(resultados_cv["accuracy"].std(ddof=1)),
        "cv_balanced_accuracy_promedio": float(
            resultados_cv["balanced_accuracy"].mean()
        ),
        "cv_f1_macro_promedio": float(resultados_cv["f1_macro"].mean()),
        **{f"test_{clave}": valor for clave, valor in metricas_prueba.items()},
    }

    joblib.dump(pipeline, CARPETA_MODELOS / "pipeline_naive_bayes.joblib")
    resultados_cv.to_csv(
        CARPETA_RESULTADOS / "cv_naive_bayes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reporte_tabla.to_csv(
        CARPETA_RESULTADOS / "reporte_naive_bayes.csv",
        encoding="utf-8-sig",
    )
    (CARPETA_RESULTADOS / "metricas_naive_bayes.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nArchivos generados:")
    print("- modelos/pipeline_naive_bayes.joblib")
    print("- resultados/cv_naive_bayes.csv")
    print("- resultados/reporte_naive_bayes.csv")
    print("- resultados/metricas_naive_bayes.json")
    print("- resultados/matriz_confusion_naive_bayes.png")


if __name__ == "__main__":
    ejecutar()
