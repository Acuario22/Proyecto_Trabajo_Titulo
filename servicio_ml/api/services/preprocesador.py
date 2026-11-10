import re
import unicodedata
import emoji
from typing import Set

STOPWORDS_EXTRA: Set[str] = {
    "favor", "municipio", "señor", "señora",
    "él", "ella", "ello", "los", "las",
    "q", "pq", "xq"
}


def eliminar_emojis(texto: str) -> str:
    """Elimina emojis utilizando la librería emoji."""
    return emoji.replace_emoji(texto, replace="")


def limpiar_texto(texto: str) -> str:
    """
    Replica exactamente la limpieza definida en 2.Fase de prepocesamiento/preprocesamiento_texto.py:
    1. Minúsculas
    2. Normalización Unicode NFC
    3. Eliminación de emojis
    4. Eliminación de saltos de línea
    5. Eliminación de acentos diacríticos sueltos
    6. Eliminación de números
    7. Retención exclusiva de letras en español (incluye tildes áéíóúüñ) y espacios
    8. Normalización de espacios múltiples y strip
    """
    if not texto:
        return ""

    texto = str(texto).lower()
    texto = unicodedata.normalize("NFC", texto)
    texto = eliminar_emojis(texto)
    texto = texto.replace("\n", " ")
    texto = texto.replace("´", "")
    texto = texto.replace("`", "")
    texto = texto.replace("¨", "")
    texto = re.sub(r"\d+", "", texto)
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


class PreprocesadorTexto:
    """
    Gestiona la lematización y el filtrado de stopwords sobre el texto limpio,
    replicando exactamente tocken_stepwo_lematizacion.py.
    """

    def __init__(self, nlp):
        self.nlp = nlp
        self.stopwords_totales = set(nlp.Defaults.stop_words).union(STOPWORDS_EXTRA)

    def lematizar_y_filtrar_stopwords(self, texto_limpio: str) -> str:
        """
        Tokeniza, lematiza respetando PROPN, descompone lemas compuestos (enclíticos)
        y filtra stopwords tanto en token original como en cada sub-lema.
        """
        if not texto_limpio:
            return ""

        doc = self.nlp(str(texto_limpio))

        pares_token_lema = [
            (token.text, token.lemma_ if token.pos_ != "PROPN" else token.text)
            for token in doc
            if token.text.strip() != ""
        ]

        lemas = []
        for t, lema in pares_token_lema:
            if t in self.stopwords_totales:
                continue

            sub_lemas = [l for l in lema.split() if l not in self.stopwords_totales]
            if not sub_lemas:
                continue

            lemas.extend(sub_lemas)

        return " ".join(lemas)

    def preprocesar(self, texto_crudo: str) -> str:
        """Pipeline completo: Limpieza -> Lematización + Stopwords."""
        limpio = limpiar_texto(texto_crudo)
        return self.lematizar_y_filtrar_stopwords(limpio)
