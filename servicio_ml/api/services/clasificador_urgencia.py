import time
import numpy as np
from typing import Dict, Any, Optional
from servicio_ml.api.services.preprocesador import PreprocesadorTexto
from servicio_ml.api.services.motor_priorizacion import MotorPriorizacion

MAPEO_URGENCIAS_MYSQL = {
    "Alta": "ALTA",
    "Media": "MEDIA",
    "Baja": "BAJA",
    "ALTA": "ALTA",
    "MEDIA": "MEDIA",
    "BAJA": "BAJA"
}


class ClasificadorUrgenciaService:
    """
    Servicio de inferencia para la tarea de Urgencia utilizando el modelo ganador
    SVM Lineal (LinearSVC) acoplado al Motor de Priorización basado en reglas de riesgo.
    """

    def __init__(
        self,
        vectorizador,
        modelo,
        preprocesador: PreprocesadorTexto,
        motor_priorizacion: Optional[MotorPriorizacion] = None
    ):
        self.vectorizador = vectorizador
        self.modelo = modelo
        self.preprocesador = preprocesador
        self.motor = motor_priorizacion or MotorPriorizacion()
        self.clases = list(modelo.classes_)

    def predecir(self, texto_crudo: str, categoria: Optional[str] = None) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de Urgencia (Fase 3):
        1. Preprocesamiento y lematización idéntica al entrenamiento.
        2. Vectorización TF-IDF con vocabulario ajustado (3.333 términos).
        3. Predicción con SVM Lineal y cálculo de confianza mediante Softmax sobre la función de decisión.
        4. Evaluación en el Motor de Priorización por reglas de riesgo (CRITICO / ALTO / NINGUNO).
        5. Normalización estricta para MySQL (TBLURGENCIA: BAJA, MEDIA, ALTA).
        """
        inicio = time.perf_counter()

        # 1. Preprocesamiento de texto
        texto_lematizado = self.preprocesador.preprocesar(texto_crudo)

        if not texto_lematizado.strip():
            texto_lematizado = texto_crudo.strip().lower()

        # 2. Vectorización TF-IDF compartida
        X_vec = self.vectorizador.transform([texto_lematizado])

        # 3. Inferencia del clasificador (LinearSVC)
        clase_predicha = str(self.modelo.predict(X_vec)[0])

        # 4. Cálculo de confianza con Softmax sobre decision_function
        decision_scores = self.modelo.decision_function(X_vec)[0]
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probabilidades = exp_scores / np.sum(exp_scores)
        idx_clase = self.clases.index(clase_predicha)
        confianza = float(probabilidades[idx_clase])

        # 5. Normalizar etiqueta predicha por ML
        urgencia_ml_db = MAPEO_URGENCIAS_MYSQL.get(clase_predicha, clase_predicha.upper())

        # 6. Refinamiento determinístico mediante Motor de Reglas
        resultado_reglas = self.motor.aplicar_reglas(
            urgencia_predicha_ml=urgencia_ml_db,
            texto=texto_crudo,
            categoria=categoria
        )

        fin = time.perf_counter()
        tiempo_ms = round((fin - inicio) * 1000.0, 2)

        return {
            "urgencia_ml": urgencia_ml_db,
            "nivel_riesgo": resultado_reglas["nivel_riesgo"],
            "urgencia_final": resultado_reglas["urgencia_final"],
            "hubo_ajuste": resultado_reglas["hubo_ajuste"],
            "confianza": round(confianza, 4),
            "tiempo_ms": tiempo_ms,
            "urgencia_ml_original": clase_predicha,
            "urgencia_final_original": resultado_reglas["urgencia_final"].capitalize()
        }
