# 15_analisis_errores_urgencia.py
#
# OBJETIVO
# Análisis cualitativo de los falsos negativos de la clase Urgencia=Alta
# que persisten en el sistema híbrido (ML + motor de reglas), con el fin
# de diagnosticar su causa raíz y fundamentar las hipótesis de mejora
# propuestas como trabajo futuro (sección de Discusión/Limitaciones).
#
# Este script:
#   1) Recrea el mismo conjunto de prueba utilizado en la evaluación del
#      sistema híbrido (14_evaluacion_sistema_hibrido.py).
#   2) Identifica los reportes donde Urgencia_Real = "Alta" pero la
#      predicción FINAL del sistema híbrido fue "Media" o "Baja"
#      (falsos negativos de la clase crítica).
#   3) Para cada caso, verifica si el motor de reglas detectó o no alguna
#      señal de riesgo en el texto, clasificando los errores en dos grupos:
#      a) "Keyword ausente": el texto no contiene ninguna palabra clave del
#         diccionario de riesgo -> el diccionario es insuficiente/incompleto.
#      b) "Keyword presente pero no disparó": el texto contiene alguna
#         palabra del diccionario, pero no fue suficiente para escalar a
#         Alta (ej. detectó riesgo ALTO pero la urgencia de ML ya era Baja,
#         y el escalamiento de un solo nivel no alcanzó a llegar a Alta).
#   4) Exporta ambos grupos a CSV para revisión manual y cita en la tesis.

import sys
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(Path(__file__).resolve().parent))

from motor_priorizacion_reglas import (  # noqa: E402
    aplicar_motor_priorizacion,
    detectar_nivel_riesgo
)


# ---------------------
# CONFIGURACIÓN
# ---------------------
archivo_dataset = BASE_DIR / "2.Fase de prepocesamiento" / "dataset_reportes_procesado_final.csv"
archivo_matriz_tfidf = BASE_DIR / "3.Fase de Vectorizacion" / "matriz_tf_idf.joblib"
archivo_modelo_urgencia = BASE_DIR / "4.Fase de entrenamiento" / "svm_lineal" / "modelo_svm_lineal_urgencia.joblib"

carpeta_salida = BASE_DIR / "4.Fase de entrenamiento" / "sistema_hibrido"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---------------------
# MAIN
# ---------------------
def main():
    print("=" * 60)
    print("ANÁLISIS DE ERRORES - FALSOS NEGATIVOS DE URGENCIA ALTA")
    print("=" * 60)

    df = pd.read_csv(archivo_dataset, encoding="utf-8-sig")
    X = joblib.load(archivo_matriz_tfidf)
    modelo_urgencia = joblib.load(archivo_modelo_urgencia)

    y_urgencia = df["Urgencia"]
    indices = df.index.values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_urgencia, indices,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_urgencia
    )

    textos_test = df.loc[idx_test, "Texto_Reporte"].values
    categorias_test = df.loc[idx_test, "Categoria"].values
    y_test_arr = y_test.values

    y_pred_ml = modelo_urgencia.predict(X_test)

    # ---------------------
    # RECONSTRUIR LA PREDICCIÓN DEL SISTEMA HÍBRIDO, CASO POR CASO
    # ---------------------
    casos_falsos_negativos = []

    for id_rep, texto, categoria, urgencia_real, urgencia_ml in zip(
        idx_test, textos_test, categorias_test, y_test_arr, y_pred_ml
    ):
        resultado = aplicar_motor_priorizacion(urgencia_ml, texto, categoria)
        urgencia_final = resultado["Urgencia_Final"]
        nivel_riesgo = resultado["Nivel_Riesgo_Detectado"]

        # Nos interesan solo los casos donde la urgencia REAL era Alta,
        # pero el sistema híbrido NO terminó prediciendo Alta.
        if urgencia_real == "Alta" and urgencia_final != "Alta":
            if nivel_riesgo == "NINGUNO":
                causa = "Keyword ausente (diccionario insuficiente)"
            else:
                # Detectó algo (ALTO), pero el escalamiento de un solo nivel
                # no alcanzó a llegar a "Alta" desde la urgencia base del ML
                causa = "Keyword presente pero escalamiento insuficiente"

            casos_falsos_negativos.append({
                "ID_Reporte": df.loc[id_rep, "ID_Reporte"] if "ID_Reporte" in df.columns else id_rep,
                "Texto_Reporte": texto,
                "Categoria": categoria,
                "Urgencia_Real": urgencia_real,
                "Urgencia_ML": urgencia_ml,
                "Nivel_Riesgo_Detectado": nivel_riesgo,
                "Urgencia_Final_Sistema": urgencia_final,
                "Causa_Probable": causa
            })

    df_errores = pd.DataFrame(casos_falsos_negativos)

    print(f"\nTotal de falsos negativos de Urgencia Alta: {len(df_errores)}")
    print("\nDistribución por causa probable:")
    print(df_errores["Causa_Probable"].value_counts())

    # ---------------------
    # GUARDAR RESULTADOS SEPARADOS POR CAUSA
    # ---------------------
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    ruta_completo = carpeta_salida / "falsos_negativos_urgencia_alta.csv"
    df_errores.to_csv(ruta_completo, index=False, encoding="utf-8-sig")
    print(f"\nListado completo guardado en: {ruta_completo}")

    df_sin_keyword = df_errores[df_errores["Causa_Probable"] == "Keyword ausente (diccionario insuficiente)"]
    ruta_sin_keyword = carpeta_salida / "falsos_negativos_sin_keyword.csv"
    df_sin_keyword.to_csv(ruta_sin_keyword, index=False, encoding="utf-8-sig")
    print(f"Casos sin keyword detectada guardados en: {ruta_sin_keyword}")

    # ---------------------
    # MUESTRA EN CONSOLA PARA REVISIÓN RÁPIDA
    # ---------------------
    print("\n" + "=" * 60)
    print("MUESTRA DE CASOS (primeros 10)")
    print("=" * 60)
    for _, fila in df_errores.head(10).iterrows():
        print(f"\n[{fila['Categoria']}] Urgencia real: {fila['Urgencia_Real']} | "
              f"ML predijo: {fila['Urgencia_ML']} | Final sistema: {fila['Urgencia_Final_Sistema']}")
        print(f"Causa: {fila['Causa_Probable']}")
        print(f"Texto: {fila['Texto_Reporte']}")


if __name__ == "__main__":
    main()