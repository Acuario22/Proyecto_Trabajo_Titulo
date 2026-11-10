from pydantic import BaseModel, Field
from typing import Optional, Dict

class ClasificacionRequest(BaseModel):
    texto: str = Field(
        ...,
        min_length=3,
        description="Texto descriptivo del reporte ciudadano",
        json_schema_extra={"example": "Hay un hoyo gigante en la calzada de San Martín esquina Ercilla que puede causar un accidente"}
    )

class CategoriaResponse(BaseModel):
    categoria: str = Field(
        ...,
        description="Categoría predicha normalizada para MySQL (PAVIMENTACION, ALCANTARILLADO, ILUMINACION)",
        json_schema_extra={"example": "PAVIMENTACION"}
    )
    confianza: float = Field(
        ...,
        description="Probabilidad asignada por el clasificador (valor entre 0.0 y 1.0)",
        json_schema_extra={"example": 0.9872}
    )
    tiempo_ms: float = Field(
        ...,
        description="Tiempo de inferencia y preprocesamiento en milisegundos",
        json_schema_extra={"example": 12.4}
    )
    categoria_original: Optional[str] = Field(
        None,
        description="Etiqueta original con tildes producida por el modelo ('Pavimentación')",
        json_schema_extra={"example": "Pavimentación"}
    )

class HealthResponse(BaseModel):
    status: str
    componentes: Dict[str, bool]
    version: str


class SentimientoPredictRequest(BaseModel):
    texto: str = Field(
        ...,
        min_length=3,
        description="Texto descriptivo del reporte ciudadano para análisis de sentimiento",
        json_schema_extra={"example": "Llevamos meses reportando esto y nadie hace nada, es una verdadera vergüenza."}
    )


class SentimientoPredictResponse(BaseModel):
    sentimiento: str = Field(
        ...,
        description="Sentimiento predicho normalizado para MySQL (ENOJADO, NEUTRAL, FRUSTRADO)",
        json_schema_extra={"example": "ENOJADO"}
    )
    confianza: float = Field(
        ...,
        description="Probabilidad asignada por la regresión logística (entre 0.0 y 1.0)",
        json_schema_extra={"example": 0.8452}
    )
    tiempo_ms: float = Field(
        ...,
        description="Tiempo de inferencia y preprocesamiento en milisegundos",
        json_schema_extra={"example": 10.5}
    )
    sentimiento_original: Optional[str] = Field(
        None,
        description="Etiqueta original con tildes producida por el modelo ('Enojado', 'Neutral', 'Frustrado')",
        json_schema_extra={"example": "Enojado"}
    )


class UrgenciaPredictRequest(BaseModel):
    texto: str = Field(
        ...,
        min_length=3,
        description="Texto descriptivo del reporte ciudadano para clasificar urgencia y evaluar riesgo",
        json_schema_extra={"example": "Hay un poste de luz a punto de caer sobre la vereda con cable pelado energizado."}
    )
    categoria: Optional[str] = Field(
        None,
        description="Categoría opcional del reporte para evaluar reglas específicas (PAVIMENTACION, ALCANTARILLADO, ILUMINACION)",
        json_schema_extra={"example": "ILUMINACION"}
    )


class UrgenciaPredictResponse(BaseModel):
    urgencia_ml: str = Field(
        ...,
        description="Urgencia predicha por el modelo SVM Lineal normalizada para MySQL (BAJA, MEDIA, ALTA)",
        json_schema_extra={"example": "MEDIA"}
    )
    nivel_riesgo: str = Field(
        ...,
        description="Nivel de riesgo detectado por el motor de reglas (CRITICO, ALTO, NINGUNO)",
        json_schema_extra={"example": "CRITICO"}
    )
    urgencia_final: str = Field(
        ...,
        description="Urgencia final ajustada tras aplicar motor de priorización (BAJA, MEDIA, ALTA)",
        json_schema_extra={"example": "ALTA"}
    )
    hubo_ajuste: bool = Field(
        ...,
        description="Indica si el motor de reglas escaló la urgencia sobre la predicción de ML",
        json_schema_extra={"example": True}
    )
    confianza: float = Field(
        ...,
        description="Confianza normalizada de la predicción de Machine Learning (0.0 a 1.0)",
        json_schema_extra={"example": 0.8124}
    )
    tiempo_ms: float = Field(
        ...,
        description="Tiempo de inferencia y evaluación de reglas en milisegundos",
        json_schema_extra={"example": 12.3}
    )
    urgencia_ml_original: Optional[str] = Field(
        None,
        description="Etiqueta original del estimador SVM ('Baja', 'Media', 'Alta')"
    )
    urgencia_final_original: Optional[str] = Field(
        None,
        description="Etiqueta final original del motor de reglas ('Baja', 'Media', 'Alta')"
    )
