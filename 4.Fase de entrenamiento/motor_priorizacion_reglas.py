# Script para implementar un motor de priorización basado en reglas, 
# que ajusta la urgencia predicha por el modelo de Machine Learning 
# según señales textuales de riesgo detectadas en la descripción del reporte.
#
# Este módulo actúa como un mecanismo de REFINAMIENTO sobre la predicción
# de Urgencia entregada por el modelo de Machine Learning,NUNCA como reemplazo completo. 
#
# COMO FUNCIONA ESTE SCRIPT:
# - Se define un conjunto de palabras clave que indican riesgo crítico o alto.
# - Se implementa una función que analiza el texto del reporte y la categoría
#   para detectar señales de riesgo.

# NIVELES DE URGENCIA (orden de severidad): Baja < Media < Alta

import re
import pandas as pd
from pathlib import Path

# -------------------------------
# NIVELES DE URGENCIA Y SU ORDEN
# --------------------------------
ORDEN_URGENCIA = {"Baja": 0, "Media": 1, "Alta": 2}
NIVELES_INVERSOS = {v: k for k, v in ORDEN_URGENCIA.items()}

# -------------------------------------------------
# PALABRAS CLAVE DE RIESGO, POR NIVEL DE SEVERIDAD
# --------------------- ---------------------------
PALABRAS_RIESGO_CRITICO = [
    "cable energizado", "cable pelado", "electrocutar", "electrocución",
    "explosión", "atropellado", "atropello", "murió", "falleció",
    "grave riesgo de vida", "puede morir", "puede matar"
]

# Riesgo ALTO: escala la urgencia en UN nivel (Baja->Media, Media->Alta).
PALABRAS_RIESGO_ALTO = [
    "accidente", "peligro", "peligroso", "grave", "colapso",
    "niño cayó", "anciano cayó", "persona cayó", "se cayó",
    "riesgo de accidente", "podría causar un accidente"
]

# -----------------------------------------
# PALABRAS CLAVE ESPECÍFICAS POR CATEGORÍA
# -----------------------------------------
PALABRAS_RIESGO_CRITICO_POR_CATEGORIA = {
    "Iluminación": ["poste caído", "cable suelto", "cortocircuito", "poste a punto de caer"],
    "Alcantarillado": ["tapa abierta", "hoyo profundo sin tapa", "alguien podría caer"],
    "Pavimentación": ["socavón", "hoyo muy profundo", "colapso de la vía"]
}

# -----------------------------------------
# FUNCIÓN 1: DETECTAR NIVEL DE RIESGO
# ------------------------------------------
def detectar_nivel_riesgo(texto: str, categoria: str) -> str:
    """
    Analiza el texto de un reporte y determina si contiene señales de
    riesgo CRÍTICO, ALTO, o ninguna señal relevante (NINGUNO).
    """
    texto_lower = str(texto).lower()

    # 1) Revisar riesgo crítico general
    if any(palabra in texto_lower for palabra in PALABRAS_RIESGO_CRITICO):
        return "CRITICO"

    # 2) Revisar riesgo crítico específico de la categoría
    palabras_categoria = PALABRAS_RIESGO_CRITICO_POR_CATEGORIA.get(categoria, [])
    if any(palabra in texto_lower for palabra in palabras_categoria):
        return "CRITICO"

    # 3) Revisar riesgo alto general
    if any(palabra in texto_lower for palabra in PALABRAS_RIESGO_ALTO):
        return "ALTO"

    return "NINGUNO"


# -----------------------------------------------
# FUNCIÓN 2: APLICAR EL ESCALAMIENTO DE URGENCIA
# ------------------------------------------------
def aplicar_motor_priorizacion(urgencia_predicha_ml: str, texto: str, categoria: str) -> dict:
    """
    Recibe la urgencia predicha por el modelo de Machine Learning y aplica
    el escalamiento correspondiente según las señales de riesgo detectadas
    en el texto. Devuelve la urgencia FINAL del sistema y el detalle de si
    hubo o no un ajuste, para fines de trazabilidad y auditoría.
    """
    nivel_riesgo = detectar_nivel_riesgo(texto, categoria)
    nivel_ml = ORDEN_URGENCIA.get(urgencia_predicha_ml, 0)

    if nivel_riesgo == "CRITICO":
        # Riesgo crítico: se fuerza directamente a "Alta"
        nivel_final = ORDEN_URGENCIA["Alta"]
    elif nivel_riesgo == "ALTO":
        # Riesgo alto: se escala en un nivel, sin superar "Alta"
        nivel_final = min(nivel_ml + 1, ORDEN_URGENCIA["Alta"])
    else:
        # Sin señales de riesgo: se mantiene la predicción original del modelo
        nivel_final = nivel_ml

    urgencia_final = NIVELES_INVERSOS[nivel_final]
    hubo_ajuste = nivel_final > nivel_ml

    return {
        "Urgencia_ML": urgencia_predicha_ml,
        "Nivel_Riesgo_Detectado": nivel_riesgo,
        "Urgencia_Final": urgencia_final,
        "Hubo_Ajuste": hubo_ajuste
    }

# -------------------------------
# EJEMPLO DE USO / PRUEBA RÁPIDA
# ---------------------------------
if __name__ == "__main__":
    casos_prueba = [
        {
            "texto": "Hay un poste caído en la esquina con cable energizado, muy peligroso",
            "categoria": "Iluminación",
            "urgencia_ml": "Baja"
        },
        {
            "texto": "Hay un bache pequeño en la vereda, nada grave",
            "categoria": "Pavimentación",
            "urgencia_ml": "Baja"
        },
        {
            "texto": "La tapa de alcantarillado está abierta, alguien podría caer",
            "categoria": "Alcantarillado",
            "urgencia_ml": "Media"
        },
    ]

    print("=" * 70)
    print("PRUEBA DEL MOTOR DE PRIORIZACIÓN")
    print("=" * 70)
    for caso in casos_prueba:
        resultado = aplicar_motor_priorizacion(
            caso["urgencia_ml"], caso["texto"], caso["categoria"]
        )
        print(f"\nTexto: {caso['texto']}")
        print(f"Categoría: {caso['categoria']}")
        print(f"Urgencia ML original: {resultado['Urgencia_ML']}")
        print(f"Riesgo detectado: {resultado['Nivel_Riesgo_Detectado']}")
        print(f"Urgencia FINAL: {resultado['Urgencia_Final']}")
        print(f"¿Hubo ajuste?: {resultado['Hubo_Ajuste']}")