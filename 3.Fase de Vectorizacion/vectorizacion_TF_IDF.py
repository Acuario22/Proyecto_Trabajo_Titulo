# OBJETIVO
# Fase 3 del pipeline de Ingeniería de Datos: transformar el texto lematizado
# (Texto_Lematizado) en una representación numérica mediante la técnica
# TF-IDF (Term Frequency - Inverse Document Frequency).
#
# El resultado es una matriz dispersa (sparse) que sirve como insumo directo
# para el entrenamiento de los modelos de clasificación (Capítulo 4).
#
# Este script:
#   1) Carga el dataset con el texto ya lematizado.
#   2) Ajusta (fit) un vectorizador TF-IDF sobre el corpus completo.
#   3) Transforma el corpus a su representación vectorial (matriz TF-IDF).
#   4) Guarda la matriz TF-IDF y el vectorizador entrenado para su reutilización
#      posterior en el entrenamiento y en la inferencia (Capítulo 4).
#   5) Reporta estadísticas descriptivas de la matriz resultante (dimensionalidad,
#      dispersión, términos con mayor peso global) para su documentación.

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path


# ---------------------
# CONFIGURACIÓN
# ---------------------
BASE_DIR = Path(__file__).resolve().parent.parent

archivo_entrada = BASE_DIR / "2.Fase de prepocesamiento" / "dataset_reportes_procesado_final.csv"
archivo_matriz_salida = BASE_DIR / "3.Fase de Vectorizacion" / "matriz_tf_idf.joblib"
archivo_vectorizador_salida = BASE_DIR / "3.Fase de Vectorizacion" / "vectorizador_tf_idf.joblib"

COLUMNA_TEXTO = "Texto_Lematizado"

# Parámetros del vectorizador TF-IDF
MAX_FEATURES = 5000       # tamaño máximo del vocabulario (limita dimensionalidad)
NGRAM_RANGE = (1, 2)      # unigramas y bigramas de palabras (ej. "bache" y "bache calle")
MIN_DF = 2                # ignora términos que aparecen en menos de 2 reportes (reduce ruido)


# ---------------------
# FUNCIÓN: VECTORIZAR EL CORPUS
# ---------------------
def vectorizar_corpus(textos: pd.Series):
    vectorizador = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF
    )
    matriz_tfidf = vectorizador.fit_transform(textos.fillna(""))
    return matriz_tfidf, vectorizador


# ---------------------
# MAIN
# ---------------------
def main():
    print("=" * 60)
    print("VECTORIZACIÓN TF-IDF DEL CORPUS")
    print("=" * 60)

    print("\nCargando dataset preprocesado (texto lematizado)...")
    df = pd.read_csv(archivo_entrada, encoding="utf-8-sig")
    print(f"Total de reportes: {len(df)}")

    if COLUMNA_TEXTO not in df.columns:
        raise ValueError(
            f"No se encontró la columna '{COLUMNA_TEXTO}'. "
            f"Verifica que el archivo provenga de la etapa de lematización."
        )

    print(f"\nAjustando y transformando vectorizador TF-IDF sobre '{COLUMNA_TEXTO}'...")
    matriz_tfidf, vectorizador = vectorizar_corpus(df[COLUMNA_TEXTO])

    # ---------------------
    # ESTADÍSTICAS DE LA MATRIZ RESULTANTE
    # ---------------------
    n_documentos, n_terminos = matriz_tfidf.shape
    total_celdas = n_documentos * n_terminos
    celdas_no_cero = matriz_tfidf.nnz
    dispersión = 1 - (celdas_no_cero / total_celdas)

    print("\n" + "-" * 60)
    print("ESTADÍSTICAS DE LA MATRIZ TF-IDF")
    print("-" * 60)
    print(f"Dimensiones de la matriz: {n_documentos} documentos x {n_terminos} términos")
    print(f"Celdas distintas de cero: {celdas_no_cero:,} de {total_celdas:,}")
    print(f"Dispersión (sparsity):    {dispersión*100:.2f}%")

    # Términos con mayor peso TF-IDF promedio en el corpus (más "distintivos")
    pesos_promedio = np.asarray(matriz_tfidf.mean(axis=0)).flatten()
    vocabulario = np.array(vectorizador.get_feature_names_out())
    top_indices = pesos_promedio.argsort()[::-1][:15]

    print("\nTop 15 términos con mayor peso TF-IDF promedio en el corpus:")
    for idx in top_indices:
        print(f"  {vocabulario[idx]:20s}  {pesos_promedio[idx]:.4f}")

    # ---------------------
    # GUARDAR MATRIZ Y VECTORIZADOR
    # ---------------------
    archivo_matriz_salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(matriz_tfidf, archivo_matriz_salida)
    joblib.dump(vectorizador, archivo_vectorizador_salida)

    print(f"\nMatriz TF-IDF guardada en:      {archivo_matriz_salida}")
    print(f"Vectorizador guardado en:       {archivo_vectorizador_salida}")
    print("\n(El vectorizador se guarda para poder transformar nuevos reportes")
    print(" con el mismo vocabulario durante la etapa de inferencia del sistema.)")


if __name__ == "__main__":
    main()