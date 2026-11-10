import time
import unicodedata
from typing import Dict, Any
from servicio_ml.api.services.preprocesador import PreprocesadorTexto

# Mapeo estricto contra el catálogo de la tabla TBLSENTIMIENTO en MySQL
MAPEO_SENTIMIENTOS_MYSQL = {
    "Enojado": "ENOJADO",
    "Neutral": "NEUTRAL",
    "Frustrado": "FRUSTRADO"
}


def normalizar_sentimiento_db(etiqueta: str) -> str:
    """
    Normaliza la etiqueta predicha para que coincida exactamente con
    la columna NOMBRE de TBLSENTIMIENTO (mayúsculas y sin tildes).
    """
    if etiqueta in MAPEO_SENTIMIENTOS_MYSQL:
        return MAPEO_SENTIMIENTOS_MYSQL[etiqueta]

    # Normalización dinámica por descomposición Unicode (fallback)
    nfkd = unicodedata.normalize("NFKD", etiqueta)
    sin_tildes = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return sin_tildes.upper().strip()


class ClasificadorSentimientoService:
    """
    Servicio de inferencia para la tarea de Sentimiento utilizando el modelo ganador
    de Regresión Logística y el vectorizador TF-IDF compartido.
    """

    def __init__(self, vectorizador, modelo, preprocesador: PreprocesadorTexto):
        self.vectorizador = vectorizador
        self.modelo = modelo
        self.preprocesador = preprocesador
        self.clases = list(modelo.classes_)

    def predecir(self, texto_crudo: str) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de inferencia para Sentimiento:
        1. Preprocesamiento y lematización idéntica al entrenamiento.
        2. Vectorización TF-IDF con vocabulario ajustado (3.333 términos).
        3. Predicción probabilística con Regresión Logística.
        4. Normalización de salida para compatibilidad directa con MySQL / Spring Boot.
        """
        inicio = time.perf_counter()

        # 1. Preprocesamiento de texto
        texto_lematizado = self.preprocesador.preprocesar(texto_crudo)

        # Fallback de seguridad si el reporte contiene solo stopwords
        if not texto_lematizado.strip():
            texto_lematizado = texto_crudo.strip().lower()

        # 2. Vectorización TF-IDF compartida
        X_vec = self.vectorizador.transform([texto_lematizado])

        # 3. Inferencia del clasificador (Regresión Logística)
        clase_predicha = str(self.modelo.predict(X_vec)[0])

        # 4. Cálculo de confianza (probabilidad asignada a la clase predicha)
        probabilidades = self.modelo.predict_proba(X_vec)[0]
        idx_clase = self.clases.index(clase_predicha)
        confianza = float(probabilidades[idx_clase])

        # 5. Normalización para MySQL
        sentimiento_db = normalizar_sentimiento_db(clase_predicha)

        fin = time.perf_counter()
        tiempo_ms = round((fin - inicio) * 1000.0, 2)

        return {
            "sentimiento": sentimiento_db,
            "confianza": round(confianza, 4),
            "tiempo_ms": tiempo_ms,
            "sentimiento_original": clase_predicha
        }
