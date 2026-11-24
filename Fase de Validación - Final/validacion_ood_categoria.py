#COMO FUNCIONA ESTE SCRIPT:
#   1) Carga un archivo con los reportes reales recopilados (Texto, Direccion,
#      Categoria_Esperada, Fuente), en formato .csv o .xlsx.
#   2) Aplica sobre cada texto EXACTAMENTE el mismo pipeline de preprocesamiento
#      utilizado en el entrenamiento (limpieza/normalización + lematización con
#      spaCy es_core_news_md + stopwords), para evitar training-serving skew.
#   3) Vectoriza con el vectorizador TF-IDF ya ajustado (sin reajustarlo) y
#      predice la categoría con el modelo serializado.
#   4) Agrega las columnas Categoria_Predicha, Confianza y Acierto, y compara
#      contra Categoria_Esperada.
#   5) Calcula Accuracy, F1-Score Macro (métrica principal del proyecto),
#      Precision y Recall Macro, reporte por clase y matriz de confusión.
#   6) Contrasta el F1-Score Macro obtenido contra el del conjunto de prueba
#      sintético, cuantificando la desviación (criterio de la sección 2.7.3.C).
#   7) Exporta el detalle completo y los casos erróneos para su análisis.

import re
import math
import unicodedata
from pathlib import Path

import emoji
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------
# CONFIGURACIÓN
# ---------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# Archivo exportado desde Google Sheets (Archivo > Descargar > .xlsx o .csv)
archivo_entrada = BASE_DIR / "Fase de Validación - Final" / "Reportes_para_validación_OOD.csv"

# IMPORTANTE: usar los MISMOS artefactos que utiliza el microservicio
# servicio_ml, para que el resultado corresponda al sistema integrado.
archivo_vectorizador = BASE_DIR / "3.Fase de Vectorizacion" / "vectorizador_tf_idf.joblib"
archivo_modelo_categoria = (
    BASE_DIR / "4.Fase de entrenamiento" / "naive_bayes" / "modelo_naive_bayes_categoria.joblib"
)

carpeta_salida = Path(__file__).resolve().parent / "resultados"

MODELO_SPACY = "es_core_news_md"

# F1-Score Macro del modelo sobre el conjunto de prueba sintético (Tabla 25)
F1_MACRO_SINTETICO = 0.9727

# Columnas esperadas en el archivo de entrada
COL_TEXTO = "Texto"
COL_ESPERADA = "Categoria_Esperada"

# Stopwords adicionales de dominio (idénticas a las del entrenamiento)
STOPWORDS_EXTRA = {
    "favor", "municipio", "señor", "señora",
    "él", "ella", "ello", "los", "las",
    "q", "pq", "xq",
}


# ---------------------
# PREPROCESAMIENTO (réplica del pipeline de entrenamiento, secciones 3.2.1 a 3.2.3)
# ---------------------
def limpiar_texto(texto) -> str:
    if pd.isna(texto):
        return ""
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFC", texto)
    texto = emoji.replace_emoji(texto, replace="")
    texto = texto.replace("\n", " ")
    texto = texto.replace("´", "").replace("`", "").replace("¨", "")
    texto = re.sub(r"\d+", "", texto)
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def lematizar(texto_limpio: str, nlp, stopwords: set) -> str:
    if not texto_limpio:
        return ""
    doc = nlp(texto_limpio)
    lemas = []
    for token in doc:
        if token.text.strip() == "" or token.text in stopwords:
            continue
        lema = token.text if token.pos_ == "PROPN" else token.lemma_
        # Los lemas compuestos (pronombres clíticos) se separan antes de filtrar
        sub_lemas = [l for l in lema.split() if l not in stopwords]
        lemas.extend(sub_lemas)
    return " ".join(lemas)


# ---------------------
# UTILIDADES
# ---------------------
def normalizar_etiqueta(etiqueta) -> str:
    """Minúsculas y sin tildes, para comparar etiquetas sin depender del formato."""
    if pd.isna(etiqueta):
        return ""
    nfkd = unicodedata.normalize("NFKD", str(etiqueta))
    sin_tildes = "".join(c for c in nfkd if not unicodedata.combining(c))
    return sin_tildes.strip().lower()


NOMBRES_PRESENTACION = {
    "pavimentacion": "Pavimentación",
    "alcantarillado": "Alcantarillado",
    "iluminacion": "Iluminación",
}


def intervalo_wilson(aciertos: int, n: int, z: float = 1.96):
    """Intervalo de confianza de Wilson (95%) para una proporción."""
    if n == 0:
        return (0.0, 0.0)
    p = aciertos / n
    denom = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / denom
    margen = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centro - margen), min(1.0, centro + margen))


def cargar_reportes(ruta: Path) -> pd.DataFrame:
    if ruta.suffix.lower() == ".csv":
        return pd.read_csv(ruta, encoding="utf-8-sig")
    return pd.read_excel(ruta)  # requiere openpyxl para .xlsx


# ---------------------
# MAIN
# ---------------------
def main():
    print("=" * 60)
    print("VALIDACIÓN OOD - CLASIFICADOR DE CATEGORÍA")
    print("=" * 60)

    df = cargar_reportes(archivo_entrada)
    print(f"\nFilas leídas: {len(df)}")

    for col in (COL_TEXTO, COL_ESPERADA):
        if col not in df.columns:
            raise ValueError(f"No se encontró la columna obligatoria '{col}'. "
                             f"Columnas disponibles: {list(df.columns)}")

    # Se descartan filas sin texto o sin categoría esperada
    n_inicial = len(df)
    df = df.dropna(subset=[COL_TEXTO, COL_ESPERADA]).copy()
    df = df[df[COL_TEXTO].astype(str).str.strip() != ""]
    if len(df) < n_inicial:
        print(f"Se descartaron {n_inicial - len(df)} filas sin texto o sin categoría esperada.")
    df = df.reset_index(drop=True)

    df["Categoria_Esperada_Norm"] = df[COL_ESPERADA].apply(normalizar_etiqueta)
    categorias_validas = set(NOMBRES_PRESENTACION)
    invalidas = df[~df["Categoria_Esperada_Norm"].isin(categorias_validas)]
    if len(invalidas) > 0:
        print("\n⚠ Filas con una categoría esperada fuera de las tres clases definidas:")
        print(invalidas[[COL_TEXTO, COL_ESPERADA]].to_string())
        raise ValueError("Corrige las categorías esperadas antes de continuar.")

    print("\nDistribución de la muestra (categoría esperada):")
    print(df["Categoria_Esperada_Norm"].map(NOMBRES_PRESENTACION).value_counts())

    # --- Carga de artefactos ---
    print("\nCargando spaCy, vectorizador y modelo...")
    nlp = spacy.load(MODELO_SPACY, disable=["parser", "ner"])
    stopwords = nlp.Defaults.stop_words.union(STOPWORDS_EXTRA)
    vectorizador = joblib.load(archivo_vectorizador)
    modelo = joblib.load(archivo_modelo_categoria)

    # --- Preprocesamiento + inferencia ---
    df["Texto_Limpio"] = df[COL_TEXTO].apply(limpiar_texto)
    df["Texto_Lematizado"] = df["Texto_Limpio"].apply(lambda t: lematizar(t, nlp, stopwords))

    X = vectorizador.transform(df["Texto_Lematizado"])
    predicciones = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)

    df["Categoria_Predicha"] = [NOMBRES_PRESENTACION[normalizar_etiqueta(p)] for p in predicciones]
    df["Confianza"] = probabilidades.max(axis=1).round(4)
    df["Categoria_Predicha_Norm"] = df["Categoria_Predicha"].apply(normalizar_etiqueta)
    df["Acierto"] = df["Categoria_Predicha_Norm"] == df["Categoria_Esperada_Norm"]

    # --- Métricas ---
    y_true = df["Categoria_Esperada_Norm"].map(NOMBRES_PRESENTACION)
    y_pred = df["Categoria_Predicha"]
    clases = sorted(NOMBRES_PRESENTACION.values())

    n = len(df)
    aciertos = int(df["Acierto"].sum())
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=clases, zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", labels=clases, zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", labels=clases, zero_division=0)
    ic_inf, ic_sup = intervalo_wilson(aciertos, n)

    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print(f"Reportes evaluados:  {n}")
    print(f"Aciertos:            {aciertos}/{n}")
    print(f"Accuracy:            {accuracy:.4f}  (IC 95% Wilson: {ic_inf:.4f} - {ic_sup:.4f})")
    print(f"F1-Score Macro:      {f1_macro:.4f}")
    print(f"Precisión Macro:     {precision_macro:.4f}")
    print(f"Recall Macro:        {recall_macro:.4f}")

    print("\nReporte de clasificación por clase:")
    print(classification_report(y_true, y_pred, labels=clases, zero_division=0))

    # --- Comparación contra el conjunto de prueba sintético ---
    desviacion_abs = F1_MACRO_SINTETICO - f1_macro
    desviacion_rel = desviacion_abs / F1_MACRO_SINTETICO * 100
    print("=" * 60)
    print("COMPARACIÓN CONTRA EL CONJUNTO DE PRUEBA SINTÉTICO")
    print("=" * 60)
    print(f"F1-Score Macro sintético (Tabla 25): {F1_MACRO_SINTETICO:.4f}")
    print(f"F1-Score Macro OOD (reportes reales): {f1_macro:.4f}")
    print(f"Desviación absoluta: {desviacion_abs:+.4f} | relativa: {desviacion_rel:+.1f}%")

    print(f"\nConfianza promedio en aciertos: {df.loc[df['Acierto'], 'Confianza'].mean():.4f}")
    if (~df["Acierto"]).any():
        print(f"Confianza promedio en errores:  {df.loc[~df['Acierto'], 'Confianza'].mean():.4f}")

    # --- Matriz de confusión ---
    cm = confusion_matrix(y_true, y_pred, labels=clases)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clases, yticklabels=clases)
    plt.title("Matriz de Confusión - Validación OOD (reportes reales) - Categoría")
    plt.ylabel("Categoría Esperada")
    plt.xlabel("Categoría Predicha")
    plt.tight_layout()

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_fig = carpeta_salida / "matriz_confusion_ood_categoria.png"
    plt.savefig(ruta_fig, dpi=150)
    plt.close()

    # --- Exportaciones ---
    columnas_salida = [c for c in df.columns if c not in ("Categoria_Esperada_Norm", "Categoria_Predicha_Norm")]
    df[columnas_salida].to_csv(carpeta_salida / "resultados_ood_categoria.csv",
                               index=False, encoding="utf-8-sig")
    errores = df.loc[~df["Acierto"], columnas_salida]
    errores.to_csv(carpeta_salida / "errores_ood_categoria.csv", index=False, encoding="utf-8-sig")

    resumen = pd.DataFrame([{
        "N_reportes": n,
        "Aciertos": aciertos,
        "Accuracy": accuracy,
        "IC95_inf": ic_inf,
        "IC95_sup": ic_sup,
        "F1_Macro_OOD": f1_macro,
        "Precision_Macro": precision_macro,
        "Recall_Macro": recall_macro,
        "F1_Macro_Sintetico": F1_MACRO_SINTETICO,
        "Desviacion_Absoluta": desviacion_abs,
        "Desviacion_Relativa_%": desviacion_rel,
    }])
    resumen.to_csv(carpeta_salida / "resumen_ood_categoria.csv", index=False, encoding="utf-8-sig")

    print(f"\nErrores de clasificación: {len(errores)}/{n}")
    print(f"Archivos guardados en: {carpeta_salida}")


if __name__ == "__main__":
    main()