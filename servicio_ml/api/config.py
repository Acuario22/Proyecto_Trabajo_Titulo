from pathlib import Path
import os

class Settings:
    PROJECT_NAME: str = "Servicio de Inferencia ML - Reportes Urbanos"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"

    # Raíz del repositorio
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # Modelo de spaCy
    SPACY_MODEL: str = os.getenv("SPACY_MODEL", "es_core_news_md")

    # Rutas a artefactos .joblib
    VECTORIZADOR_PATH: Path = BASE_DIR / "3.Fase de Vectorizacion" / "vectorizador_tf_idf.joblib"
    MODELO_CATEGORIA_PATH: Path = BASE_DIR / "4.Fase de entrenamiento" / "naive_bayes" / "modelo_naive_bayes_categoria.joblib"
    MODELO_SENTIMIENTO_PATH: Path = BASE_DIR / "4.Fase de entrenamiento" / "regresion_logistica" / "modelo_regresion_logistica_sentimiento.joblib"
    MODELO_URGENCIA_PATH: Path = BASE_DIR / "4.Fase de entrenamiento" / "svm_lineal" / "modelo_svm_lineal_urgencia.joblib"
    MOTOR_REGLAS_PATH: Path = BASE_DIR / "4.Fase de entrenamiento" / "motor_priorizacion_reglas.py"

    # Configuración de servidor
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

settings = Settings()
