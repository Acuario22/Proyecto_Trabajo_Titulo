from fastapi import APIRouter, Request, HTTPException, status
from servicio_ml.api.schemas import (
    ClasificacionRequest,
    CategoriaResponse,
    SentimientoPredictRequest,
    SentimientoPredictResponse,
    UrgenciaPredictRequest,
    UrgenciaPredictResponse
)
from servicio_ml.api.services.clasificador_cat import ClasificadorCategoriaService
from servicio_ml.api.services.clasificador_sentimiento import ClasificadorSentimientoService
from servicio_ml.api.services.clasificador_urgencia import ClasificadorUrgenciaService

router = APIRouter(prefix="/clasificar", tags=["Clasificación"])


def get_clasificador_service(request: Request) -> ClasificadorCategoriaService:
    service = getattr(request.app.state, "clasificador_service", None)
    if not service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El servicio de clasificación de categoría no está inicializado en memoria."
        )
    return service


def get_clasificador_sentimiento_service(request: Request) -> ClasificadorSentimientoService:
    service = getattr(request.app.state, "clasificador_sentimiento_service", None)
    if not service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El servicio de clasificación de sentimiento no está inicializado en memoria."
        )
    return service


def get_clasificador_urgencia_service(request: Request) -> ClasificadorUrgenciaService:
    service = getattr(request.app.state, "clasificador_urgencia_service", None)
    if not service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El servicio de clasificación de urgencia no está inicializado en memoria."
        )
    return service


@router.post(
    "/categoria",
    response_model=CategoriaResponse,
    status_code=status.HTTP_200_OK,
    summary="Clasificar Categoría de Reporte Ciudadano",
    description="Recibe el texto descriptivo del ciudadano, ejecuta el pipeline de preprocesamiento, vectorización TF-IDF y clasificación Naive Bayes, retornando la categoría normalizada para la base de datos MySQL."
)
async def clasificar_categoria(
    payload: ClasificacionRequest,
    request: Request
) -> CategoriaResponse:
    try:
        service = get_clasificador_service(request)
        resultado = service.predecir(payload.texto)
        return CategoriaResponse(**resultado)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el proceso de inferencia de categoría: {str(exc)}"
        )


@router.post(
    "/sentimiento",
    response_model=SentimientoPredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Clasificar Sentimiento de Reporte Ciudadano",
    description="Recibe el texto descriptivo del ciudadano, ejecuta el preprocesamiento, vectorización TF-IDF y clasificación de sentimiento con Regresión Logística, retornando el sentimiento normalizado para MySQL (ENOJADO, NEUTRAL, FRUSTRADO)."
)
async def clasificar_sentimiento(
    payload: SentimientoPredictRequest,
    request: Request
) -> SentimientoPredictResponse:
    try:
        service = get_clasificador_sentimiento_service(request)
        resultado = service.predecir(payload.texto)
        return SentimientoPredictResponse(**resultado)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el proceso de inferencia de sentimiento: {str(exc)}"
        )


@router.post(
    "/urgencia",
    response_model=UrgenciaPredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Clasificar Urgencia de Reporte Ciudadano (SVM + Motor de Reglas)",
    description="Recibe el texto del reporte ciudadano (y opcionalmente su categoría), ejecuta la inferencia con SVM Lineal acoplada al motor de priorización por reglas de riesgo (CRITICO / ALTO / NINGUNO) y retorna la urgencia ML y la urgencia final normalizada para MySQL (BAJA, MEDIA, ALTA)."
)
async def clasificar_urgencia(
    payload: UrgenciaPredictRequest,
    request: Request
) -> UrgenciaPredictResponse:
    try:
        service = get_clasificador_urgencia_service(request)
        resultado = service.predecir(payload.texto, payload.categoria)
        return UrgenciaPredictResponse(**resultado)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el proceso de inferencia de urgencia: {str(exc)}"
        )
