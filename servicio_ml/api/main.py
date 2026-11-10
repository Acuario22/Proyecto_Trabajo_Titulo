import logging
import joblib
import spacy
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from servicio_ml.api.config import settings
from servicio_ml.api.schemas import HealthResponse
from servicio_ml.api.services.preprocesador import PreprocesadorTexto
from servicio_ml.api.services.clasificador_cat import ClasificadorCategoriaService
from servicio_ml.api.services.clasificador_sentimiento import ClasificadorSentimientoService
from servicio_ml.api.services.motor_priorizacion import MotorPriorizacion
from servicio_ml.api.services.clasificador_urgencia import ClasificadorUrgenciaService
from servicio_ml.api.routers.clasificacion import router as clasificacion_router

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("servicio_ml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Ciclo de vida de FastAPI:
    Carga todos los artefactos de Machine Learning, NLP y el Motor de Reglas en memoria RAM
    durante el inicio del servidor, manteniéndolos calientes para inferencias ultra rápidas.
    """
    logger.info("=" * 60)
    logger.info("INICIANDO MICROSERVICIO DE INFERENCIA ML - FASE 1, FASE 2 & FASE 3")
    logger.info("=" * 60)

    # 1. Cargar modelo spaCy
    logger.info("1/5 Cargando modelo spaCy '%s' (sin parser ni ner)...", settings.SPACY_MODEL)
    try:
        nlp = spacy.load(settings.SPACY_MODEL, disable=["parser", "ner"])
        logger.info("Modelo spaCy cargado exitosamente.")
    except Exception as exc:
        logger.error("Error al cargar spaCy: %s", exc)
        raise exc

    # 2. Cargar vectorizador TF-IDF
    logger.info("2/5 Cargando Vectorizador TF-IDF desde '%s'...", settings.VECTORIZADOR_PATH)
    if not settings.VECTORIZADOR_PATH.exists():
        raise FileNotFoundError(f"No se encontró el vectorizador en: {settings.VECTORIZADOR_PATH}")
    vectorizador = joblib.load(settings.VECTORIZADOR_PATH)
    logger.info("Vectorizador TF-IDF cargado. Vocabulario: %d términos.", len(vectorizador.vocabulary_))

    # 3. Cargar modelo de Categoría (Naive Bayes)
    logger.info("3/5 Cargando Modelo Naive Bayes Categoría desde '%s'...", settings.MODELO_CATEGORIA_PATH)
    if not settings.MODELO_CATEGORIA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo de categoría en: {settings.MODELO_CATEGORIA_PATH}")
    modelo_cat = joblib.load(settings.MODELO_CATEGORIA_PATH)
    logger.info("Clasificador de Categoría cargado. Clases: %s", modelo_cat.classes_)

    # 4. Cargar modelo de Sentimiento (Regresión Logística)
    logger.info("4/5 Cargando Modelo Regresión Logística Sentimiento desde '%s'...", settings.MODELO_SENTIMIENTO_PATH)
    if not settings.MODELO_SENTIMIENTO_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo de sentimiento en: {settings.MODELO_SENTIMIENTO_PATH}")
    modelo_sentimiento = joblib.load(settings.MODELO_SENTIMIENTO_PATH)
    logger.info("Clasificador de Sentimiento cargado. Clases: %s", modelo_sentimiento.classes_)

    # 5. Cargar modelo de Urgencia (SVM Lineal) y Motor de Reglas
    logger.info("5/5 Cargando Modelo SVM Lineal Urgencia desde '%s'...", settings.MODELO_URGENCIA_PATH)
    if not settings.MODELO_URGENCIA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo de urgencia en: {settings.MODELO_URGENCIA_PATH}")
    modelo_urgencia = joblib.load(settings.MODELO_URGENCIA_PATH)
    logger.info("Clasificador de Urgencia cargado. Clases: %s", modelo_urgencia.classes_)

    motor_priorizacion = MotorPriorizacion()
    logger.info("Motor de priorización por reglas de riesgo inicializado.")

    # 6. Instanciar servicios en memoria
    preprocesador = PreprocesadorTexto(nlp)
    clasificador_service = ClasificadorCategoriaService(
        vectorizador=vectorizador,
        modelo=modelo_cat,
        preprocesador=preprocesador
    )
    clasificador_sentimiento_service = ClasificadorSentimientoService(
        vectorizador=vectorizador,
        modelo=modelo_sentimiento,
        preprocesador=preprocesador
    )
    clasificador_urgencia_service = ClasificadorUrgenciaService(
        vectorizador=vectorizador,
        modelo=modelo_urgencia,
        preprocesador=preprocesador,
        motor_priorizacion=motor_priorizacion
    )

    # Inyección en app.state para desacoplar handlers
    app.state.nlp = nlp
    app.state.vectorizador = vectorizador
    app.state.modelo_cat = modelo_cat
    app.state.modelo_sentimiento = modelo_sentimiento
    app.state.modelo_urgencia = modelo_urgencia
    app.state.motor_priorizacion = motor_priorizacion
    app.state.preprocesador = preprocesador
    app.state.clasificador_service = clasificador_service
    app.state.clasificador_sentimiento_service = clasificador_sentimiento_service
    app.state.clasificador_urgencia_service = clasificador_urgencia_service

    logger.info("=" * 60)
    logger.info("MICROSERVICIO LISTO PARA RECIBIR PETICIONES EN EL PUERTO %s", settings.PORT)
    logger.info("=" * 60)

    yield

    logger.info("Apagando microservicio y liberando recursos...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Microservicio de inferencia de Machine Learning para la clasificación automática "
        "de reportes ciudadanos de infraestructura urbana (Categoría, Sentimiento, Urgencia). "
    ),
    lifespan=lifespan
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar routers versionados
app.include_router(clasificacion_router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["General"])
async def root():
    return {
        "servicio": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "clasificar_categoria": f"{settings.API_V1_PREFIX}/clasificar/categoria",
            "clasificar_sentimiento": f"{settings.API_V1_PREFIX}/clasificar/sentimiento",
            "clasificar_urgencia": f"{settings.API_V1_PREFIX}/clasificar/urgencia"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoreo"])
async def health_check():
    """
    Verifica que todos los componentes y artefactos en memoria estén operativos.
    """
    spacy_ok = hasattr(app.state, "nlp") and app.state.nlp is not None
    vec_ok = hasattr(app.state, "vectorizador") and app.state.vectorizador is not None
    model_cat_ok = hasattr(app.state, "modelo_cat") and app.state.modelo_cat is not None
    model_sent_ok = hasattr(app.state, "modelo_sentimiento") and app.state.modelo_sentimiento is not None
    model_urg_ok = hasattr(app.state, "modelo_urgencia") and app.state.modelo_urgencia is not None
    motor_ok = hasattr(app.state, "motor_priorizacion") and app.state.motor_priorizacion is not None

    status_general = (
        "healthy"
        if (spacy_ok and vec_ok and model_cat_ok and model_sent_ok and model_urg_ok and motor_ok)
        else "unhealthy"
    )

    return HealthResponse(
        status=status_general,
        componentes={
            "spacy_nlp": spacy_ok,
            "vectorizador_tfidf": vec_ok,
            "modelo_naive_bayes_categoria": model_cat_ok,
            "modelo_regresion_logistica_sentimiento": model_sent_ok,
            "modelo_svm_lineal_urgencia": model_urg_ok,
            "motor_priorizacion_reglas": motor_ok
        },
        version=settings.VERSION
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "servicio_ml.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
