# 05_muestreo_auditoria_manual.py
#
# OBJETIVO
# Segundo filtro de la estrategia de auditoría y control de calidad del
# dataset (2.7.2 - B): extraer una muestra aleatoria representativa del 5%
# del dataset (n=100) para su inspección manual cualitativa, bajo tres
# criterios de aceptación:
#   1) Consistencia de Categoría
#   2) Coherencia de Sentimiento
#   3) Exactitud de Palabras Clave
#
# Este script NO evalúa los criterios automáticamente (son de juicio
# cualitativo humano). Su función es:
#   a) Generar la muestra aleatoria reproducible (random_state=42).
#   b) Exportar una plantilla en CSV con columnas vacías para que el
#      desarrollador complete la evaluación de cada criterio manualmente.
#   c) Una vez completada la plantilla a mano, un segundo bloque de este
#      script (ver función resumir_auditoria) calcula las estadísticas
#      finales de aprobación/rechazo para incluir en el informe.

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------
# CONFIGURACIÓN
# ---------------------
archivo_entrada = BASE_DIR / "Fase de generación" / "dataset_reportes_sinteticos.csv"
archivo_plantilla = "auditoria_manual_muestra.csv"
archivo_completado = "auditoria_manual_muestra_completada.csv"  # se usa luego, tras completar a mano

TAMANO_MUESTRA = 100
SEMILLA = 42  # random_state fijo, para que la muestra sea reproducible


# ---------------------
# PASO 1: GENERAR LA MUESTRA Y LA PLANTILLA DE AUDITORÍA
# ---------------------
def generar_plantilla():
    print("=" * 60)
    print("GENERACIÓN DE MUESTRA PARA AUDITORÍA MANUAL")
    print("=" * 60)

    df = pd.read_csv(archivo_entrada, encoding="utf-8-sig")
    print(f"Total de reportes en el dataset: {len(df)}")

    # Muestra aleatoria reproducible (5% del dataset, n=100)
    df_muestra = df.sample(n=TAMANO_MUESTRA, random_state=SEMILLA).copy()
    print(f"Muestra extraída: {len(df_muestra)} reportes ({len(df_muestra)/len(df)*100:.1f}% del total)")

    # Columnas vacías para completar manualmente durante la auditoría.
    # Se usa "Sí"/"No" como valores esperados en cada criterio, y una
    # columna final "Aprobado" que resume el resultado global del reporte.
    df_muestra["Criterio_1_Consistencia_Categoria"] = ""   # Sí / No
    df_muestra["Criterio_2_Coherencia_Sentimiento"] = ""   # Sí / No
    df_muestra["Criterio_3_Exactitud_Palabras_Clave"] = "" # Sí / No
    df_muestra["Aprobado"] = ""                            # Aprobado / Rechazado
    df_muestra["Observaciones"] = ""                       # comentario libre del auditor

    df_muestra.to_csv(archivo_plantilla, index=False, encoding="utf-8-sig")
    print(f"\nPlantilla guardada en: {archivo_plantilla}")
    print("\nInstrucciones:")
    print("  1. Abre el archivo generado (Excel, LibreOffice o similar).")
    print("  2. Para cada fila, revisa el Texto_Reporte contra los 3 criterios:")
    print("     - Consistencia de Categoría: ¿el texto corresponde a la Categoria asignada?")
    print("     - Coherencia de Sentimiento: ¿el tono del texto coincide con el Sentimiento asignado?")
    print("     - Exactitud de Palabras Clave: ¿las Palabras_Clave_Sugeridas pertenecen al texto?")
    print("  3. Completa cada columna de criterio con 'Sí' o 'No'.")
    print("  4. Completa 'Aprobado' con 'Aprobado' si los 3 criterios son 'Sí', o 'Rechazado' si falla alguno.")
    print("  5. Agrega observaciones si lo consideras relevante.")
    print(f"  6. Guarda el archivo completado como: {archivo_completado}")
    print("  7. Vuelve a correr este script con la función resumir_auditoria() para obtener las estadísticas finales.")


# ---------------------
# PASO 2: RESUMIR LOS RESULTADOS DE LA AUDITORÍA YA COMPLETADA A MANO
# ---------------------
def resumir_auditoria():
    print("=" * 60)
    print("RESUMEN DE AUDITORÍA MANUAL")
    print("=" * 60)

    df_aud = pd.read_csv(archivo_completado, encoding="utf-8-sig")
    total = len(df_aud)

    if df_aud["Aprobado"].isnull().any() or (df_aud["Aprobado"] == "").any():
        print("⚠ Advertencia: hay filas sin completar en la columna 'Aprobado'. "
              "Termina la auditoría manual antes de generar el resumen final.")

    conteo = df_aud["Aprobado"].value_counts()
    print("\nResultado global:")
    print(conteo)

    n_aprobados = conteo.get("Aprobado", 0)
    n_rechazados = conteo.get("Rechazado", 0)
    print(f"\nTasa de aprobación: {n_aprobados}/{total} ({n_aprobados/total*100:.1f}%)")
    print(f"Tasa de rechazo:    {n_rechazados}/{total} ({n_rechazados/total*100:.1f}%)")

    print("\nDetalle por criterio (proporción de 'No'):")
    for criterio in [
        "Criterio_1_Consistencia_Categoria",
        "Criterio_2_Coherencia_Sentimiento",
        "Criterio_3_Exactitud_Palabras_Clave"
    ]:
        n_no = (df_aud[criterio] == "No").sum()
        print(f"  {criterio}: {n_no}/{total} ({n_no/total*100:.1f}% de fallas)")

    if n_rechazados > 0:
        df_rechazados = df_aud[df_aud["Aprobado"] == "Rechazado"]
        df_rechazados.to_csv("auditoria_manual_rechazados.csv", index=False, encoding="utf-8-sig")
        print(f"\nSe guardaron los {n_rechazados} casos rechazados en 'auditoria_manual_rechazados.csv'")


# ---------------------
# MAIN
# ---------------------
if __name__ == "__main__":
    # Ejecuta primero generar_plantilla(). Una vez completada la auditoría
    # manual y guardado el archivo como "auditoria_manual_muestra_completada.csv",
    # comenta la línea de abajo y descomenta resumir_auditoria().

    generar_plantilla()
    # resumir_auditoria()