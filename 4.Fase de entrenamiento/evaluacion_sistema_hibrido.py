# 14_evaluacion_sistema_hibrido.py
#
# OBJETIVO
# Capítulo 4 - Sección 4.2: Evaluación cuantitativa del sistema híbrido de
# priorización, combinando la predicción del mejor modelo de Machine Learning
# para Urgencia (SVM Lineal optimizado) con el motor de priorización basado
# en reglas (13_motor_priorizacion_reglas.py).
#
# Este script:
#   1) Recrea EXACTAMENTE el mismo split de entrenamiento/prueba utilizado
#      durante el entrenamiento y la optimización (mismo random_state,
#      mismo test_size, misma estratificación), para que el conjunto de
#      prueba sea idéntico y la comparación sea válida.
#   2) Carga el mejor modelo de Urgencia ya entrenado y optimizado (.joblib).
#   3) Genera la urgencia predicha por el modelo de ML para el conjunto de
#      prueba (baseline).
#   4) Aplica el motor de reglas sobre cada reporte del conjunto de prueba,
#      usando el texto, la categoría real y la urgencia predicha por ML,
#      obteniendo la urgencia FINAL del sistema híbrido.
#   5) Calcula F1-Score Macro para ambos escenarios (ML solo vs. ML + reglas)
#      y los compara, generando también la matriz de confusión del sistema
#      híbrido.

import sys
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
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
# IMPORTAR EL MOTOR DE REGLAS
# ---------------------
# Se asume que 13_motor_priorizacion_reglas.py está en la misma carpeta,
# o en una carpeta accesible desde el PYTHONPATH. Ajustar la ruta si es
# necesario.
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(Path(__file__).resolve().parent))

from motor_priorizacion_reglas import aplicar_motor_priorizacion  # noqa: E402


# ---------------------
# CONFIGURACIÓN
# ---------------------
archivo_dataset = BASE_DIR / "2.Fase de prepocesamiento" / "dataset_reportes_procesado_final.csv"
archivo_matriz_tfidf = BASE_DIR / "3.Fase de Vectorizacion" / "matriz_tf_idf.joblib"

# Ajustar a la ruta real donde quedó guardado el mejor modelo de Urgencia
# (SVM Lineal optimizado, F1-Macro = 0.4021 según sección 4.2.4)
archivo_modelo_urgencia = BASE_DIR / "4.Fase de entrenamiento" / "svm_lineal" / "modelo_svm_lineal_urgencia.joblib"

carpeta_salida = BASE_DIR / "4.Fase de entrenamiento" / "sistema_hibrido"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---------------------
# MAIN
# ---------------------
def main():
    print("=" * 60)
    print("EVALUACIÓN DEL SISTEMA HÍBRIDO (ML + MOTOR DE REGLAS)")
    print("=" * 60)

    print("\nCargando dataset, matriz TF-IDF y modelo de Urgencia optimizado...")
    df = pd.read_csv(archivo_dataset, encoding="utf-8-sig")
    X = joblib.load(archivo_matriz_tfidf)
    modelo_urgencia = joblib.load(archivo_modelo_urgencia)

    y_urgencia = df["Urgencia"]

    # ---------------------
    # RECREAR EXACTAMENTE EL MISMO SPLIT USADO EN ENTRENAMIENTO
    # ---------------------
    # Se usa np.arange para obtener también los ÍNDICES del conjunto de
    # prueba, y así poder recuperar el Texto_Reporte y la Categoria
    # correspondientes a cada fila de X_test / y_test.
    indices = df.index.values
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_urgencia, indices,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_urgencia
    )

    print(f"Conjunto de prueba recreado: {X_test.shape[0]} muestras")

    # Textos y categorías reales correspondientes al conjunto de prueba
    textos_test = df.loc[idx_test, "Texto_Reporte"].values
    categorias_test = df.loc[idx_test, "Categoria"].values

    # ---------------------
    # PASO 1: PREDICCIÓN BASELINE DEL MODELO DE ML (SIN MOTOR DE REGLAS)
    # ---------------------
    y_pred_ml = modelo_urgencia.predict(X_test)

    f1_macro_ml = f1_score(y_test, y_pred_ml, average="macro", zero_division=0)
    print(f"\nF1-Score Macro (solo ML, baseline):        {f1_macro_ml:.4f}")

    # ---------------------
    # PASO 2: APLICAR EL MOTOR DE REGLAS SOBRE CADA PREDICCIÓN
    # ---------------------
    y_pred_hibrido = []
    total_ajustes = 0

    for texto, categoria, urgencia_ml in zip(textos_test, categorias_test, y_pred_ml):
        resultado = aplicar_motor_priorizacion(urgencia_ml, texto, categoria)
        y_pred_hibrido.append(resultado["Urgencia_Final"])
        if resultado["Hubo_Ajuste"]:
            total_ajustes += 1

    print(f"Reportes ajustados por el motor de reglas: {total_ajustes}/{len(y_pred_hibrido)} "
          f"({total_ajustes/len(y_pred_hibrido)*100:.1f}%)")

    # ---------------------
    # PASO 3: EVALUAR EL SISTEMA HÍBRIDO (ML + REGLAS)
    # ---------------------
    f1_macro_hibrido = f1_score(y_test, y_pred_hibrido, average="macro", zero_division=0)
    f1_weighted_hibrido = f1_score(y_test, y_pred_hibrido, average="weighted", zero_division=0)
    precision_macro_hibrido = precision_score(y_test, y_pred_hibrido, average="macro", zero_division=0)
    recall_macro_hibrido = recall_score(y_test, y_pred_hibrido, average="macro", zero_division=0)

    print(f"\nF1-Score Macro (ML + motor de reglas):     {f1_macro_hibrido:.4f}")
    print(f"F1-Score Weighted (ML + motor de reglas):  {f1_weighted_hibrido:.4f}")
    print(f"Precisión Macro (ML + motor de reglas):    {precision_macro_hibrido:.4f}")
    print(f"Recall Macro (ML + motor de reglas):       {recall_macro_hibrido:.4f}")

    diferencia = f1_macro_hibrido - f1_macro_ml
    print(f"\nDiferencia (Híbrido - ML solo): {diferencia:+.4f}")

    print("\nReporte de clasificación (sistema híbrido):")
    print(classification_report(y_test, y_pred_hibrido, zero_division=0))

    # ---------------------
    # MATRIZ DE CONFUSIÓN DEL SISTEMA HÍBRIDO
    # ---------------------
    clases_ordenadas = sorted(y_urgencia.unique())
    cm = confusion_matrix(y_test, y_pred_hibrido, labels=clases_ordenadas)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=clases_ordenadas, yticklabels=clases_ordenadas
    )
    plt.title("Matriz de Confusión - Sistema Híbrido (ML + Motor de Reglas) - Urgencia")
    plt.ylabel("Valor Real")
    plt.xlabel("Valor Predicho (Final)")
    plt.tight_layout()

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_figura = carpeta_salida / "matriz_confusion_sistema_hibrido_urgencia.png"
    plt.savefig(ruta_figura, dpi=150)
    plt.close()
    print(f"\nMatriz de confusión guardada en: {ruta_figura}")

    # ---------------------
    # RESUMEN COMPARATIVO (ML SOLO vs. SISTEMA HÍBRIDO)
    # ---------------------
    resumen = pd.DataFrame([
        {
            "Escenario": "Solo Modelo ML (SVM optimizado)",
            "F1_Macro": f1_macro_ml,
            "Reportes_Ajustados": 0
        },
        {
            "Escenario": "Sistema Híbrido (ML + Motor de Reglas)",
            "F1_Macro": f1_macro_hibrido,
            "Reportes_Ajustados": total_ajustes
        }
    ])

    print("\n" + "=" * 60)
    print("RESUMEN COMPARATIVO")
    print("=" * 60)
    print(resumen.to_string(index=False))

    ruta_resumen = carpeta_salida / "resumen_sistema_hibrido_urgencia.csv"
    resumen.to_csv(ruta_resumen, index=False, encoding="utf-8-sig")
    print(f"\nResumen guardado en: {ruta_resumen}")


if __name__ == "__main__":
    main()