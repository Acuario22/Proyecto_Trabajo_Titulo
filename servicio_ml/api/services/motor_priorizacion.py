"""
Motor de priorización basado en reglas para refinamiento de urgencia.
Replica y amplía la lógica validada en 4.Fase de entrenamiento/motor_priorizacion_reglas.py
"""
import unicodedata
from typing import Optional, Dict, List

# Orden de severidad de urgencia
ORDEN_URGENCIA: Dict[str, int] = {"BAJA": 0, "MEDIA": 1, "ALTA": 2}
NIVELES_INVERSOS: Dict[int, str] = {0: "BAJA", 1: "MEDIA", 2: "ALTA"}

# Palabras clave de riesgo crítico (fuerza directamente a ALTA)
PALABRAS_RIESGO_CRITICO: List[str] = [
    "cable energizado", "cable pelado", "electrocutar", "electrocución",
    "explosión", "atropellado", "atropello", "murió", "falleció",
    "grave riesgo de vida", "puede morir", "puede matar"
]

# Palabras clave de riesgo alto (escala 1 nivel: BAJA -> MEDIA, MEDIA -> ALTA)
PALABRAS_RIESGO_ALTO: List[str] = [
    "accidente", "peligro", "peligroso", "grave", "colapso",
    "niño cayó", "anciano cayó", "persona cayó", "se cayó",
    "riesgo de accidente", "podría causar un accidente"
]

# Palabras clave críticas por categoría
PALABRAS_RIESGO_CRITICO_POR_CATEGORIA: Dict[str, List[str]] = {
    "ILUMINACION": ["poste caído", "poste caido", "cable suelto", "cortocircuito", "poste a punto de caer"],
    "ALCANTARILLADO": ["tapa abierta", "hoyo profundo sin tapa", "alguien podría caer", "alguien podria caer"],
    "PAVIMENTACION": ["socavón", "socavon", "hoyo muy profundo", "colapso de la vía", "colapso de la via"]
}


def normalizar_texto_reglas(texto: str) -> str:
    """Normaliza texto a minúsculas y remueve acentos sueltos para robustez en coincidencia."""
    return str(texto).lower()


def normalizar_categoria_clave(categoria: Optional[str]) -> Optional[str]:
    """Normaliza cualquier variante de categoría ('Pavimentación', 'PAVIMENTACION') a clave canónica."""
    if not categoria:
        return None
    nfkd = unicodedata.normalize("NFKD", str(categoria))
    sin_tildes = "".join([c for c in nfkd if not unicodedata.combining(c)])
    cat_upper = sin_tildes.upper().strip()
    if cat_upper in PALABRAS_RIESGO_CRITICO_POR_CATEGORIA:
        return cat_upper
    return None


class MotorPriorizacion:
    """
    Motor determinístico de priorización basado en señales textuales de riesgo.
    Actúa como un mecanismo de refinamiento sobre la predicción del modelo de ML.
    """

    @staticmethod
    def detectar_nivel_riesgo(texto: str, categoria: Optional[str] = None) -> str:
        """
        Analiza el texto del reporte y la categoría para detectar si contiene
        señales de riesgo CRITICO, ALTO, o NINGUNO.
        """
        texto_lower = normalizar_texto_reglas(texto)

        # 1. Riesgo crítico general
        if any(palabra in texto_lower for palabra in PALABRAS_RIESGO_CRITICO):
            return "CRITICO"

        # 2. Riesgo crítico por categoría
        cat_clave = normalizar_categoria_clave(categoria)
        if cat_clave and cat_clave in PALABRAS_RIESGO_CRITICO_POR_CATEGORIA:
            palabras_cat = PALABRAS_RIESGO_CRITICO_POR_CATEGORIA[cat_clave]
            if any(palabra in texto_lower for palabra in palabras_cat):
                return "CRITICO"
        elif not cat_clave:
            # Si no se especifica categoría, se evalúan todas las listas críticas para no omitir riesgos
            for palabras_cat in PALABRAS_RIESGO_CRITICO_POR_CATEGORIA.values():
                if any(palabra in texto_lower for palabra in palabras_cat):
                    return "CRITICO"

        # 3. Riesgo alto general
        if any(palabra in texto_lower for palabra in PALABRAS_RIESGO_ALTO):
            return "ALTO"

        return "NINGUNO"

    @classmethod
    def aplicar_reglas(
        cls,
        urgencia_predicha_ml: str,
        texto: str,
        categoria: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Aplica el escalamiento correspondiente de acuerdo con el nivel de riesgo detectado:
        - CRITICO: Escala directamente a ALTA.
        - ALTO: Escala en un nivel (+1 nivel de severidad, máx ALTA).
        - NINGUNO: Se mantiene la predicción de Machine Learning.
        """
        # Normalizar urgencia_ml a mayúsculas
        ml_normalizada = str(urgencia_predicha_ml).upper().strip()
        nivel_ml = ORDEN_URGENCIA.get(ml_normalizada, 0)

        nivel_riesgo = cls.detectar_nivel_riesgo(texto, categoria)

        if nivel_riesgo == "CRITICO":
            nivel_final = ORDEN_URGENCIA["ALTA"]
        elif nivel_riesgo == "ALTO":
            nivel_final = min(nivel_ml + 1, ORDEN_URGENCIA["ALTA"])
        else:
            nivel_final = nivel_ml

        urgencia_final = NIVELES_INVERSOS[nivel_final]
        hubo_ajuste = nivel_final > nivel_ml

        return {
            "urgencia_ml": ml_normalizada,
            "nivel_riesgo": nivel_riesgo,
            "urgencia_final": urgencia_final,
            "hubo_ajuste": hubo_ajuste
        }
