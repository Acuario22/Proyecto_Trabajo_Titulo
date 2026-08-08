# 02b_tokenizacion_lematizacion.py
#
# OBJETIVO
# Continuar la Fase 2 (Ingeniería y Preparación de Datos) sobre el texto ya
# normalizado y limpio (Texto_Limpio), aplicando:
#   1) Tokenización       -> separar el texto en palabras individuales
#   2) Eliminación de stopwords -> quitar palabras sin valor semántico (el, la, de, que, etc.)
#   3) Lematización        -> reducir cada palabra a su forma base/raíz
#                              (ej. "reportado", "reportando" -> "reportar")
#
# Este script se ejecuta DESPUÉS de 02_preprocesamiento_texto.py, tomando como
# entrada el archivo "dataset_reportes_preprocesado.csv" (que ya tiene la
# columna "Texto_Limpio").
#
# REQUISITOS (instalar antes de correr):
#   pip install spacy
#   python -m spacy download es_core_news_md
#
# Usamos spaCy porque su modelo en español ya trae un lematizador entrenado
# y una lista de stopwords en español, evitando tener que armar todo a mano.
# Usamos el modelo "md" (mediano) en vez de "sm" porque tiene mejor precisión
# de lematización, especialmente en verbos conjugados.

import pandas as pd
import spacy
import json


# ---------------------
# CONFIGURACIÓN
# ---------------------
archivo_entrada = "dataset_reportes_preprocesado.csv"
archivo_salida = "dataset_reportes_procesado_final.csv"

MODELO_SPACY = "es_core_news_md"


# ---------------------
# CARGA DEL MODELO DE SPACY
# ---------------------
# Desactivamos componentes que no necesitamos (parser, ner) para que el
# procesamiento sea más rápido, ya que solo necesitamos tokenización y lematización.
print("Cargando modelo de lenguaje en español (spaCy)...")
nlp = spacy.load(MODELO_SPACY, disable=["parser", "ner"])
print("Modelo cargado correctamente.\n")

# Lista de stopwords en español que trae spaCy por defecto
stopwords_spacy = nlp.Defaults.stop_words

# ---------------------
# STOPWORDS ADICIONALES ESPECÍFICAS DEL DOMINIO (opcional)
# ---------------------
# Agrega aquí palabras propias de tus reportes que no aportan valor semántico
# para clasificar categoría/urgencia/sentimiento. Ajusta según lo que veas
# en tus resultados.
stopwords_extra = {
    "favor", "municipio", "señor", "señora",
    "él", "ella", "ello", "los", "las",
    "q"
}

stopwords_totales = stopwords_spacy.union(stopwords_extra)


# ---------------------
# FUNCIÓN: TOKENIZAR + LEMATIZAR + ELIMINAR STOPWORDS
# ---------------------
def procesar_texto(texto: str) -> dict:
    if not texto or pd.isna(texto):
        return {
            "tokens": [],
            "tokens_sin_stopwords": [],
            "lemas": [],
            "texto_lematizado": ""
        }

    doc = nlp(str(texto))
    tokens = [token.text for token in doc if token.text.strip() != ""]

    # Generamos el lema de cada token (respetando nombres propios, que NO
    # deben lematizarse, ej. "ercilla" no debe pasar a "ercillo").
    #
    # IMPORTANTE: para palabras con pronombre pegado (ej. "dejarlo"),
    # spaCy devuelve un lema COMPUESTO de varias palabras en un solo string
    # (ej. "dejar él"). Por eso separamos el lema en sub-palabras antes
    # de filtrar stopwords, en vez de comparar el string completo -- si no,
    # palabras como "él" quedan "escondidas" dentro del lema compuesto y
    # nunca se filtran.
    pares_token_lema = [
        (token.text, token.lemma_ if token.pos_ != "PROPN" else token.text)
        for token in doc
        if token.text.strip() != ""
    ]

    tokens_sin_stopwords = []
    lemas = []
    for t, lema in pares_token_lema:
        # Si el token original ya es stopword, lo saltamos directo
        if t in stopwords_totales:
            continue

        # Separamos el lema por si viene compuesto de varias palabras
        # y filtramos cada sub-palabra contra la lista de stopwords
        sub_lemas = [l for l in lema.split() if l not in stopwords_totales]

        if not sub_lemas:
            continue  # todo el lema resultó ser stopword (ej. "lo" -> "él")

        tokens_sin_stopwords.append(t)
        lemas.extend(sub_lemas)

    texto_lematizado = " ".join(lemas)

    return {
        "tokens": tokens,
        "tokens_sin_stopwords": tokens_sin_stopwords,
        "lemas": lemas,
        "texto_lematizado": texto_lematizado
    }


# ---------------------
# MAIN
# ---------------------
def main():
    print("=" * 60)
    print("TOKENIZACIÓN, STOPWORDS Y LEMATIZACIÓN")
    print("=" * 60)

    print("\nCargando dataset preprocesado...")
    df = pd.read_csv(archivo_entrada, encoding="utf-8-sig")
    print(f"Total de reportes encontrados: {len(df)}")

    if "Texto_Limpio" not in df.columns:
        raise ValueError(
            "No se encontró la columna 'Texto_Limpio'. "
            "Asegúrate de correr primero 02_preprocesamiento_texto.py."
        )

    print("\nProcesando tokenización y lematización (puede tardar unos segundos)...")
    resultados = df["Texto_Limpio"].apply(procesar_texto)

    df["Tokens"] = resultados.apply(lambda r: json.dumps(r["tokens"], ensure_ascii=False))
    df["Tokens_Sin_Stopwords"] = resultados.apply(lambda r: json.dumps(r["tokens_sin_stopwords"], ensure_ascii=False))
    df["Texto_Lematizado"] = resultados.apply(lambda r: r["texto_lematizado"])

    df.to_csv(archivo_salida, index=False, encoding="utf-8-sig")
    print(f"\nDataset final guardado en: {archivo_salida}")

    print("\n" + "=" * 60)
    print("EJEMPLOS")
    print("=" * 60)
    for i in range(min(5, len(df))):
        print(f"\nEjemplo {i+1}")
        print("\nTEXTO LIMPIO:")
        print(df.iloc[i]["Texto_Limpio"])
        print("\nTOKENS (sin stopwords):")
        print(df.iloc[i]["Tokens_Sin_Stopwords"])
        print("\nTEXTO LEMATIZADO:")
        print(df.iloc[i]["Texto_Lematizado"])
        print("\n" + "-" * 60)

    print("\n" + "=" * 60)
    print("ESTADÍSTICAS")
    print("=" * 60)
    promedio_tokens_antes = resultados.apply(lambda r: len(r["tokens"])).mean()
    promedio_tokens_despues = resultados.apply(lambda r: len(r["tokens_sin_stopwords"])).mean()
    print(f"Promedio de tokens antes de quitar stopwords:  {promedio_tokens_antes:.1f}")
    print(f"Promedio de tokens después de quitar stopwords: {promedio_tokens_despues:.1f}")
    print(f"Reducción promedio: {(1 - promedio_tokens_despues/promedio_tokens_antes) * 100:.1f}%")


if __name__ == "__main__":
    main()