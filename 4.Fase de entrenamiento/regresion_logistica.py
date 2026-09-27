# Script para entrenar y evaluar un modelo de 
# Regresión Logística para las variables objetivo del dataset de reportes.
#
# COMO FUNCIONA ESTE SCRIPT:
# - Este script se ejecuta DESPUÉS de vectorizacion_TF_IDF.py, tomando como
#   entrada la matriz TF-IDF generada y el dataset con las etiquetas.  
# - Para cada variable objetivo (Categoria, Urgencia, Sentimiento), 
#   se entrena un modelo de Regresión Logística y se evalúa su desempeño.
# - Se generan métricas de evaluación (Precision, Recall, F1-Score: Macro y Weighted ), 
#   matriz de confusión y se guardan los modelos entrenados en formato .joblib.
# - Se genera un resumen comparativo de las 3 tareas de clasificación.

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

#Importación de librerías de Scikit-learn para el entrenamiento y evaluación del modelo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------
# CONFIGURACIÓN
# ---------------------
BASE_DIR = Path(__file__).resolve().parent.parent

#Reportes procesados: dataset con las tres variables objetivo
archivo_dataset = BASE_DIR / "2.Fase de prepocesamiento" / "dataset_reportes_procesado_final.csv"

#Matriz TF-IDF generada a partir de la columna "Descripcion" del dataset 
archivo_matriz_tfidf = BASE_DIR / "3.Fase de Vectorizacion" / "matriz_tf_idf.joblib"

carpeta_salida = BASE_DIR / "4.Fase de Entrenamiento" / "regresion_logistica"

VARIABLES_OBJETIVO = ["Categoria", "Urgencia", "Sentimiento"]

#Semilla para garantizar reproducibilidad
RANDOM_STATE = 42

#Proporción de datos para el conjunto de prueba (20% para prueba, 80% para entrenamiento)
TEST_SIZE = 0.2

# ------------------------------------------------------------------
# FUNCIÓN 1: ENTRENAR Y EVALUAR UN MODELO PARA UNA VARIABLE OBJETIVO
# ------------------------------------------------------------------
def entrenar_y_evaluar(X, y, nombre_variable: str) -> dict:
    print("\n" + "=" * 60)
    print(f"VARIABLE OBJETIVO: {nombre_variable}")
    print("=" * 60)

    # División estratificada para preservar la proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Entrenamiento: {X_train.shape[0]} muestras | Prueba: {X_test.shape[0]} muestras")

    # Entrenamiento del modelo
    modelo = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    modelo.fit(X_train, y_train)

    # Predicciones sobre el conjunto de prueba
    y_pred = modelo.predict(X_test)

    # -----------------------
    # MÉTRICAS DE EVALUACIÓN
    # -----------------------
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))

    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"F1-Score Macro:    {f1_macro:.4f}")
    print(f"F1-Score Weighted: {f1_weighted:.4f}")
    print(f"Precisión Macro:   {precision_macro:.4f}")
    print(f"Recall Macro:      {recall_macro:.4f}")

    # ---------------------
    # MATRIZ DE CONFUSIÓN
    # ---------------------
    clases_ordenadas = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=clases_ordenadas)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=clases_ordenadas, yticklabels=clases_ordenadas
    )
    plt.title(f"Matriz de Confusión - Regresión Logística - {nombre_variable}")
    plt.ylabel("Valor Real")
    plt.xlabel("Valor Predicho")
    plt.tight_layout()

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_figura = carpeta_salida / f"matriz_confusion_{nombre_variable.lower()}.png"
    plt.savefig(ruta_figura, dpi=150)
    plt.close()
    print(f"Matriz de confusión guardada en: {ruta_figura}")

    # ------------------------
    # GUARDAR EL MODELO ENTRENADO
    # ------------------------
    ruta_modelo = carpeta_salida / f"modelo_regresion_logistica_{nombre_variable.lower()}.joblib"
    joblib.dump(modelo, ruta_modelo)
    print(f"Modelo guardado en: {ruta_modelo}")

    return {
        "Variable": nombre_variable,
        "F1_Macro": f1_macro,
        "F1_Weighted": f1_weighted,
        "Precision_Macro": precision_macro,
        "Recall_Macro": recall_macro
    }

# ---------------------
# MAIN
# ---------------------
def main():
    print("=" * 60)
    print("ENTRENAMIENTO - REGRESIÓN LOGÍSTICA")
    print("=" * 60)

    print("\nCargando dataset y matriz TF-IDF...")
    df = pd.read_csv(archivo_dataset, encoding="utf-8-sig")
    X = joblib.load(archivo_matriz_tfidf)
    print(f"Dataset: {len(df)} reportes | Matriz TF-IDF: {X.shape}")

    resultados = []
    for variable in VARIABLES_OBJETIVO:
        if variable not in df.columns:
            print(f"Columna '{variable}' no encontrada en el dataset, se omite.")
            continue
        y = df[variable]
        resultado = entrenar_y_evaluar(X, y, variable)
        resultados.append(resultado)

    # ---------------------
    # RESUMEN COMPARATIVO
    # ---------------------
    print("\n" + "=" * 60)
    print("RESUMEN COMPARATIVO - REGRESIÓN LOGÍSTICA")
    print("=" * 60)
    df_resumen = pd.DataFrame(resultados)
    print(df_resumen.to_string(index=False))

    ruta_resumen = carpeta_salida / "resumen_regresion_logistica.csv"
    df_resumen.to_csv(ruta_resumen, index=False, encoding="utf-8-sig")
    print(f"\nResumen guardado en: {ruta_resumen}")

if __name__ == "__main__":
    main()