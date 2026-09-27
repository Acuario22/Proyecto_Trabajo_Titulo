# Script para entrenar y evaluar un modelo de 
# Naive Bayes Multinomial para cada variable objetivo del dataset de reportes.
#
# COMO FUNCIONA ESTE SCRIPT:
# - Este script se ejecuta DESPUÉS de vectorizacion_TF_IDF.py, tomando como
#   entrada la matriz TF-IDF generada y el dataset con las etiquetas.
# - Para cada variable objetivo (Categoria, Urgencia, Sentimiento), 
#   se entrena un modelo de Naive Bayes Multinomial y se evalúa su desempeño.
# - Mismo flujo que los scripts de Regresión Logística y SVM Lineal.

# MEJORAS:
# - Se implementa búsqueda de hiperparámetros mediante GridSearchCV para optimizar
#  el parámetro de suavizado "alpha" del modelo de Naive Bayes Multinomial.
# - Se utiliza StratifiedKFold para asegurar que cada pliegue de validación
#   cruzada mantenga la proporción de clases.

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
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

archivo_dataset = BASE_DIR / "2.Fase de prepocesamiento" / "dataset_reportes_procesado_final.csv"
archivo_matriz_tfidf = BASE_DIR / "3.Fase de Vectorizacion" / "matriz_tf_idf.joblib"

carpeta_salida = BASE_DIR / "4.Fase de Entrenamiento" / "naive_bayes"

VARIABLES_OBJETIVO = ["Categoria", "Urgencia", "Sentimiento"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5

# ---------------------
# GRILLA DE HIPERPARÁMETROS A EXPLORAR
# ---------------------
# alpha: parámetro de suavizado de Laplace (Laplace/Lidstone smoothing).
#    Valores bajos (cercanos a 0) = menos suavizado, el modelo confía más
#    fuertemente en las frecuencias observadas en el entrenamiento (mayor
#    riesgo de asignar probabilidad cero a términos poco frecuentes).
#    Valores altos = más suavizado, el modelo es más conservador y menos
#    sensible a términos raros.
PARAM_GRID = {
    "alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
}


# -------------------------------------------------------------------
# FUNCIÓN 1: ENTRENAR Y EVALUAR UN MODELO PARA UNA VARIABLE OBJETIVO
# -------------------------------------------------------------------
def optimizar_y_evaluar(X, y, nombre_variable: str) -> dict:
    print("\n" + "=" * 60)
    print(f"VARIABLE OBJETIVO: {nombre_variable}")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Entrenamiento: {X_train.shape[0]} muestras | Prueba: {X_test.shape[0]} muestras")

    cv_estratificado = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    modelo_base = MultinomialNB()

    busqueda_grid = GridSearchCV(
        estimator=modelo_base,
        param_grid=PARAM_GRID,
        scoring="f1_macro",
        cv=cv_estratificado,
        n_jobs=-1,  
        verbose=1
    )

    print("\nIniciando búsqueda de hiperparámetros con GridSearchCV...")
    busqueda_grid.fit(X_train, y_train)

    print(f"\nMejores hiperparámetros encontrados: {busqueda_grid.best_params_}")
    mejor_modelo = busqueda_grid.best_estimator_

    y_pred = mejor_modelo.predict(X_test)



    # ---------------------
    # MÉTRICAS DE EVALUACIÓN
    # ---------------------
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
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=clases_ordenadas, yticklabels=clases_ordenadas
    )
    plt.title(f"Matriz de Confusión - Naive Bayes Multinomial - {nombre_variable}")
    plt.ylabel("Valor Real")
    plt.xlabel("Valor Predicho")
    plt.tight_layout()

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_figura = carpeta_salida / f"matriz_confusion_{nombre_variable.lower()}.png"
    plt.savefig(ruta_figura, dpi=150)
    plt.close()
    print(f"Matriz de confusión guardada en: {ruta_figura}")

    # ---------------------------
    # GUARDAR EL MODELO ENTRENADO
    # ----------------------------
    ruta_modelo = carpeta_salida / f"modelo_naive_bayes_{nombre_variable.lower()}.joblib"
    joblib.dump(mejor_modelo, ruta_modelo)
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
    print("ENTRENAMIENTO - NAIVE BAYES MULTINOMIAL")
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
        resultado = optimizar_y_evaluar(X, y, variable)
        resultados.append(resultado)

    # ---------------------
    # RESUMEN COMPARATIVO
    # ---------------------
    print("\n" + "=" * 60)
    print("RESUMEN COMPARATIVO - NAIVE BAYES MULTINOMIAL")
    print("=" * 60)
    df_resumen = pd.DataFrame(resultados)
    print(df_resumen.to_string(index=False))

    ruta_resumen = carpeta_salida / "resumen_naive_bayes.csv"
    df_resumen.to_csv(ruta_resumen, index=False, encoding="utf-8-sig")
    print(f"\nResumen guardado en: {ruta_resumen}")

if __name__ == "__main__":
    main()