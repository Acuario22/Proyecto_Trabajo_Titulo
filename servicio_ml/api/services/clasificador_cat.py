import time
import unicodedata
from typing import Dict, Any
from servicio_ml.api.services.preprocesador import PreprocesadorTexto

# Mapeo estricto contra el catálogo de la tabla TBLCATEGORIA en MySQL
MAPEO_CATEGORIAS_MYSQL = {
    "Pavimentación": "PAVIMENTACION",
    "Alcantarillado": "ALCANTARILLADO",
    "Iluminación": "ILUMINACION"
}


def normalizar_etiqueta_db(etiqueta: str) -> str:
    """
    Normaliza la etiqueta predicha para que coincida exactamente con
    la columna NOMBRE de TBLCATEGORIA (mayúsculas y sin tildes).
    """
    if etiqueta in MAPEO_CATEGORIAS_MYSQL:
        return MAPEO_CATEGORIAS_MYSQL[etiqueta]

    # Normalización dinámica por descomposición Unicode (fallback)
    nfkd = unicodedata.normalize("NFKD", etiqueta)
    sin_tildes = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return sin_tildes.upper().strip()


class ClasificadorCategoriaService:
    """
    Servicio de inferencia para la tarea de Categoría utilizando el modelo ganador
    Naive Bayes Multinomial y el vectorizador TF-IDF.
    """

    def __init__(self, vectorizador, modelo, preprocesador: PreprocesadorTexto):
        self.vectorizador = vectorizador
        self.modelo = modelo
        self.preprocesador = preprocesador
        self.clases = list(modelo.classes_)

    def predecir(self, texto_crudo: str) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de inferencia:
        1. Preprocesamiento y lematización idéntica al entrenamiento.
        2. Vectorización TF-IDF con vocabulario ajustado (5000 términos).
        3. Predicción probabilística con Naive Bayes.
        4. Normalización de salida para compatibilidad directa con MySQL / Spring Boot.
        """
        inicio = time.perf_counter()

        # 1. Preprocesamiento de texto
        texto_lematizado = self.preprocesador.preprocesar(texto_crudo)

        # Fallback de seguridad si el reporte contiene solo stopwords
        if not texto_lematizado.strip():
            texto_lematizado = texto_crudo.strip().lower()

        # 2. Vectorización
        X_vec = self.vectorizador.transform([texto_lematizado])

        # 3. Inferencia del clasificador
        clase_predicha = str(self.modelo.predict(X_vec)[0])

        # 4. Cálculo de confianza (probabilidad máxima)
        probabilidades = self.modelo.predict_proba(X_vec)[0]
        idx_clase = self.clases.index(clase_predicha)
        confianza = float(probabilidades[idx_clase])

        # 5. Normalización para MySQL
        categoria_db = normalizar_etiqueta_db(clase_predicha)

        fin = time.perf_counter()
        tiempo_ms = round((fin - inicio) * 1000.0, 2)

        return {
            "categoria": categoria_db,
            "confianza": round(confianza, 4),
            "tiempo_ms": tiempo_ms,
            "categoria_original": clase_predicha
        }
