# -*- coding: utf-8 -*-
"""K-means con validacion cruzada para crear niveles de costo del seguro."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parent
ARCHIVO_DATOS = BASE / "data" / "insurance.csv"
CARPETA_MODELOS = BASE / "modelos"
CARPETA_RESULTADOS = BASE / "resultados"

VARIABLE = "charges"
NUMERO_CLUSTERS = 3
SEMILLA = 42
PARTICIONES = 5
PARTICIONES_MINIMAS = 2


def cargar_datos() -> pd.DataFrame:
    if not ARCHIVO_DATOS.is_file():
        raise FileNotFoundError(
            f"No se encontro el dataset: {ARCHIVO_DATOS}\n"
            "Copia insurance.csv dentro de la carpeta data."
        )

    datos = pd.read_csv(ARCHIVO_DATOS)
    if VARIABLE not in datos.columns:
        raise ValueError(f"El dataset no contiene la columna {VARIABLE}.")

    cantidad_original = len(datos)
    datos[VARIABLE] = pd.to_numeric(datos[VARIABLE], errors="coerce")
    datos = datos.dropna(subset=[VARIABLE]).drop_duplicates().reset_index(drop=True)

    if len(datos) < NUMERO_CLUSTERS * PARTICIONES_MINIMAS:
        raise ValueError("No hay suficientes registros para validar K-means.")

    print("=== REVISION DEL DATASET ===")
    print(f"Registros originales: {cantidad_original}")
    print(f"Registros utilizados: {len(datos)}")
    print(f"Filas eliminadas: {cantidad_original - len(datos)}")
    print(f"Columnas: {list(datos.columns)}")
    print("\nEstadisticas de charges:")
    print(datos[VARIABLE].describe().round(2).to_string())
    return datos


def validar_con_kfold(x: pd.DataFrame) -> pd.DataFrame:
    """Evalua K-means en datos no vistos mediante KFold de cinco particiones."""
    kfold = KFold(n_splits=PARTICIONES, shuffle=True, random_state=SEMILLA)
    resultados: list[dict[str, float | int]] = []

    for numero_fold, (indices_entrena, indices_valida) in enumerate(
        kfold.split(x), start=1
    ):
        x_entrena = x.iloc[indices_entrena]
        x_valida = x.iloc[indices_valida]

        escalador = StandardScaler()
        x_entrena_escalado = escalador.fit_transform(x_entrena)
        x_valida_escalado = escalador.transform(x_valida)

        modelo = KMeans(
            n_clusters=NUMERO_CLUSTERS,
            n_init=20,
            random_state=SEMILLA,
        )
        modelo.fit(x_entrena_escalado)
        etiquetas = modelo.predict(x_valida_escalado)
        distancias = modelo.transform(x_valida_escalado)
        distancia_cuadrada_media = float(
            np.mean(np.min(distancias, axis=1) ** 2)
        )

        silueta = float("nan")
        if 1 < len(np.unique(etiquetas)) < len(x_valida_escalado):
            silueta = float(silhouette_score(x_valida_escalado, etiquetas))

        resultados.append(
            {
                "fold": numero_fold,
                "registros_entrenamiento": len(x_entrena),
                "registros_validacion": len(x_valida),
                "silueta": silueta,
                "distancia_cuadrada_media": distancia_cuadrada_media,
            }
        )

    return pd.DataFrame(resultados)


def calcular_curva_codo(x_escalado: np.ndarray) -> pd.DataFrame:
    resultados = []
    for cantidad in range(2, 7):
        modelo = KMeans(
            n_clusters=cantidad,
            n_init=20,
            random_state=SEMILLA,
        )
        etiquetas = modelo.fit_predict(x_escalado)
        resultados.append(
            {
                "clusters": cantidad,
                "inercia": float(modelo.inertia_),
                "silueta": float(silhouette_score(x_escalado, etiquetas)),
            }
        )
    return pd.DataFrame(resultados)


def entrenar_modelo_final(
    datos: pd.DataFrame,
) -> tuple[Pipeline, dict[int, str], pd.DataFrame, np.ndarray]:
    x = datos[[VARIABLE]]
    pipeline = Pipeline(
        steps=[
            ("escalador", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=NUMERO_CLUSTERS,
                    n_init=20,
                    random_state=SEMILLA,
                ),
            ),
        ]
    )
    etiquetas = pipeline.fit_predict(x)

    escalador = pipeline.named_steps["escalador"]
    modelo = pipeline.named_steps["kmeans"]
    centroides_originales = escalador.inverse_transform(
        modelo.cluster_centers_
    ).ravel()
    orden = np.argsort(centroides_originales)
    mapa_niveles = {
        int(orden[0]): "bajo",
        int(orden[1]): "medio",
        int(orden[2]): "alto",
    }

    salida = datos.copy()
    salida["cluster"] = etiquetas
    salida["level"] = salida["cluster"].map(mapa_niveles)
    return pipeline, mapa_niveles, salida, centroides_originales


def guardar_grafica(
    datos_niveles: pd.DataFrame,
    curva_codo: pd.DataFrame,
) -> None:
    colores = {"bajo": "#2ca02c", "medio": "#ff9f1c", "alto": "#d62728"}
    orden = ["bajo", "medio", "alto"]

    figura, ejes = plt.subplots(1, 3, figsize=(16, 5))

    ejes[0].plot(
        curva_codo["clusters"],
        curva_codo["inercia"],
        marker="o",
        color="#1f77b4",
    )
    ejes[0].axvline(NUMERO_CLUSTERS, linestyle="--", color="#555555")
    ejes[0].set_title("Metodo del codo")
    ejes[0].set_xlabel("Numero de clusters")
    ejes[0].set_ylabel("Inercia")
    ejes[0].grid(alpha=0.25)

    for nivel in orden:
        valores = datos_niveles.loc[datos_niveles["level"] == nivel, VARIABLE]
        ejes[1].hist(
            valores,
            bins=25,
            alpha=0.65,
            label=f"{nivel} (n={len(valores)})",
            color=colores[nivel],
        )
    ejes[1].set_title("Distribucion de costos por nivel")
    ejes[1].set_xlabel("Costo del seguro")
    ejes[1].set_ylabel("Frecuencia")
    ejes[1].legend()
    ejes[1].grid(alpha=0.25)

    valores_boxplot = [
        datos_niveles.loc[datos_niveles["level"] == nivel, VARIABLE]
        for nivel in orden
    ]
    caja = ejes[2].boxplot(valores_boxplot, tick_labels=orden, patch_artist=True)
    for parche, nivel in zip(caja["boxes"], orden):
        parche.set_facecolor(colores[nivel])
        parche.set_alpha(0.65)
    ejes[2].set_title("Costos por cluster")
    ejes[2].set_xlabel("Nivel")
    ejes[2].set_ylabel("Costo del seguro")
    ejes[2].grid(alpha=0.25)

    figura.tight_layout()
    figura.savefig(CARPETA_RESULTADOS / "kmeans_resultados.png", dpi=170)
    plt.close(figura)


def ejecutar() -> None:
    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    datos = cargar_datos()
    x = datos[[VARIABLE]]

    print("\n=== VALIDACION CRUZADA K-MEANS ===")
    validacion = validar_con_kfold(x)
    print(validacion.round(4).to_string(index=False))
    print(
        f"\nSilueta CV: {validacion['silueta'].mean():.4f} "
        f"+/- {validacion['silueta'].std(ddof=1):.4f}"
    )
    print(
        "Distancia cuadrada media CV: "
        f"{validacion['distancia_cuadrada_media'].mean():.4f} "
        f"+/- {validacion['distancia_cuadrada_media'].std(ddof=1):.4f}"
    )

    escalador_temporal = StandardScaler()
    x_escalado = escalador_temporal.fit_transform(x)
    curva_codo = calcular_curva_codo(x_escalado)

    pipeline, mapa_niveles, datos_niveles, centroides = entrenar_modelo_final(datos)
    etiquetas = datos_niveles["cluster"].to_numpy()
    x_final_escalado = pipeline.named_steps["escalador"].transform(x)

    silueta_final = silhouette_score(x_final_escalado, etiquetas)
    calinski = calinski_harabasz_score(x_final_escalado, etiquetas)
    davies = davies_bouldin_score(x_final_escalado, etiquetas)

    print("\n=== MODELO FINAL CON TRES CLUSTERS ===")
    print(f"Silueta: {silueta_final:.4f}")
    print(f"Calinski-Harabasz: {calinski:.4f}")
    print(f"Davies-Bouldin: {davies:.4f}")

    resumen = (
        datos_niveles.groupby(["cluster", "level"], as_index=False)
        .agg(
            registros=(VARIABLE, "size"),
            costo_minimo=(VARIABLE, "min"),
            costo_promedio=(VARIABLE, "mean"),
            costo_maximo=(VARIABLE, "max"),
        )
        .sort_values("costo_promedio")
    )
    resumen["centroide"] = resumen["cluster"].map(
        {indice: valor for indice, valor in enumerate(centroides)}
    )
    print("\nResumen de clusters:")
    print(resumen.round(2).to_string(index=False))

    joblib.dump(pipeline, CARPETA_MODELOS / "pipeline_kmeans.joblib")
    joblib.dump(mapa_niveles, CARPETA_MODELOS / "mapa_niveles_kmeans.joblib")
    validacion.to_csv(
        CARPETA_RESULTADOS / "validacion_cruzada_kmeans.csv",
        index=False,
        encoding="utf-8-sig",
    )
    curva_codo.to_csv(
        CARPETA_RESULTADOS / "metodo_codo_kmeans.csv",
        index=False,
        encoding="utf-8-sig",
    )
    resumen.to_csv(
        CARPETA_RESULTADOS / "resumen_clusters_kmeans.csv",
        index=False,
        encoding="utf-8-sig",
    )
    datos_niveles.drop(columns="cluster").to_csv(
        CARPETA_RESULTADOS / "insurance_with_levels.csv",
        index=False,
        encoding="utf-8-sig",
    )
    guardar_grafica(datos_niveles, curva_codo)

    print("\nArchivos generados:")
    print("- modelos/pipeline_kmeans.joblib")
    print("- modelos/mapa_niveles_kmeans.joblib")
    print("- resultados/validacion_cruzada_kmeans.csv")
    print("- resultados/metodo_codo_kmeans.csv")
    print("- resultados/resumen_clusters_kmeans.csv")
    print("- resultados/insurance_with_levels.csv")
    print("- resultados/kmeans_resultados.png")


if __name__ == "__main__":
    ejecutar()
