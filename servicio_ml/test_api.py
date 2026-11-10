"""
Script de prueba automatizado para verificar el funcionamiento de la API
y los pipelines de inferencia de Categoría (Fase 1), Sentimiento (Fase 2) y Urgencia (Fase 3).
"""
import sys
from pathlib import Path

# Agregar raíz al PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from starlette.testclient import TestClient
from servicio_ml.api.main import app


def ejecutar_pruebas():
    print("=" * 70)
    print("INICIANDO PRUEBAS DEL SERVICIO DE INFERENCIA (FASE 1, FASE 2 & FASE 3)")
    print("=" * 70)

    with TestClient(app) as client:
        # 1. Probar Health Check
        print("\n[1] Probando GET /health...")
        res_health = client.get("/health")
        assert res_health.status_code == 200, f"Error en health: {res_health.text}"
        data_health = res_health.json()
        print(f"Status: {res_health.status_code}")
        print(f"Respuesta: {data_health}")
        assert data_health["status"] == "healthy"
        assert all(data_health["componentes"].values()), "Todos los componentes deben estar cargados (True)"
        assert "modelo_svm_lineal_urgencia" in data_health["componentes"]
        assert "motor_priorizacion_reglas" in data_health["componentes"]

        # 2. Casos de prueba para Categoría (Fase 1)
        casos_categoria = [
            {
                "esperado": "PAVIMENTACION",
                "texto": "Existe un bache enorme y peligroso en la calzada de San Martín con Ercilla, varios autos han roto neumáticos."
            },
            {
                "esperado": "ALCANTARILLADO",
                "texto": "La tapa de la cámara de alcantarillado está abierta en medio de la vereda y salen aguas servidas con un olor fétido."
            },
            {
                "esperado": "ILUMINACION",
                "texto": "La luminaria de la plaza está completamente apagada y el poste hace chispas dejando la cuadra a oscuras."
            }
        ]

        print("\n[2] Probando POST /api/v1/clasificar/categoria (Fase 1)...")
        for i, caso in enumerate(casos_categoria, start=1):
            res = client.post("/api/v1/clasificar/categoria", json={"texto": caso["texto"]})
            assert res.status_code == 200, f"Error en predicción categoría {i}: {res.text}"
            data = res.json()
            print(f"\nCaso Categoría {i}:")
            print(f"  Texto: {caso['texto']}")
            print(f"  Predicción DB: {data['categoria']} (Esperado: {caso['esperado']})")
            print(f"  Confianza:     {data['confianza'] * 100:.2f}%")
            print(f"  Tiempo ms:     {data['tiempo_ms']} ms")
            assert data["categoria"] == caso["esperado"], f"Discrepancia en categoría: {data['categoria']} != {caso['esperado']}"
            assert data["tiempo_ms"] < 300.0, f"Latencia excesiva: {data['tiempo_ms']} ms"

        # 3. Casos de prueba para Sentimiento (Fase 2)
        casos_sentimiento = [
            {
                "esperado": "ENOJADO",
                "texto": "La vereda del pasaje San Miguel está llena de agrietados que no permiten andar con seguridad. Esto es una vergüenza, ¿cómo pueden dejarlo así?"
            },
            {
                "esperado": "NEUTRAL",
                "texto": "En la avenida Libertador Sur, entre los pasajes Tucapel y San Martín, tengo un bache grande que está afectando mi auto. Ya reporté esto antes pero no me dieron respuesta."
            },
            {
                "esperado": "FRUSTRADO",
                "texto": "Ya hace casi un mes que se formó ese hoyo en la acera de la plaza Libertador y nadie ha pasado a repararlo. ¡Está peligroso para los niños! Ya reporté esto antes y no hubo nada."
            }
        ]

        print("\n[3] Probando POST /api/v1/clasificar/sentimiento (Fase 2)...")
        for i, caso in enumerate(casos_sentimiento, start=1):
            res = client.post("/api/v1/clasificar/sentimiento", json={"texto": caso["texto"]})
            assert res.status_code == 200, f"Error en predicción sentimiento {i}: {res.text}"
            data = res.json()
            print(f"\nCaso Sentimiento {i}:")
            print(f"  Texto: {caso['texto']}")
            print(f"  Predicción DB: {data['sentimiento']} (Esperado: {caso['esperado']})")
            print(f"  Confianza:     {data['confianza'] * 100:.2f}%")
            print(f"  Tiempo ms:     {data['tiempo_ms']} ms")
            assert data["sentimiento"] == caso["esperado"], f"Discrepancia en sentimiento: {data['sentimiento']} != {caso['esperado']}"
            assert data["tiempo_ms"] < 300.0, f"Latencia excesiva: {data['tiempo_ms']} ms"

        # 4. Casos de prueba para Urgencia y Motor de Reglas (Fase 3)
        casos_urgencia = [
            {
                "descripcion": "Riesgo Crítico con escalamiento forzado a ALTA",
                "texto": "Hay un poste caído en la esquina con cable energizado que echa chispas",
                "categoria": "ILUMINACION",
                "esperado_riesgo": "CRITICO",
                "esperado_final": "ALTA"
            },
            {
                "descripcion": "Riesgo Alto con escalamiento de 1 nivel (hubo_ajuste = True)",
                "texto": "En la villa El Bosque hay un hoyo en el pavimento que está mal nivelado, es peligroso para los carros",
                "categoria": "PAVIMENTACION",
                "esperado_riesgo": "ALTO",
                "esperado_ajuste": True
            },
            {
                "descripcion": "Sin señales de riesgo (hubo_ajuste = False, se mantiene ML)",
                "texto": "En la avenida Libertador, entre las calles Ercilla y San Martín, hay un bache tan grande que da miedo cruzar en bicicleta",
                "categoria": "PAVIMENTACION",
                "esperado_riesgo": "NINGUNO",
                "esperado_ajuste": False
            }
        ]

        print("\n[4] Probando POST /api/v1/clasificar/urgencia (Fase 3)...")
        for i, caso in enumerate(casos_urgencia, start=1):
            payload = {"texto": caso["texto"], "categoria": caso["categoria"]}
            res = client.post("/api/v1/clasificar/urgencia", json=payload)
            assert res.status_code == 200, f"Error en predicción urgencia {i}: {res.text}"
            data = res.json()
            print(f"\nCaso Urgencia {i} ({caso['descripcion']}):")
            print(f"  Texto:          {caso['texto']}")
            print(f"  Categoría:      {caso['categoria']}")
            print(f"  Urgencia ML:    {data['urgencia_ml']}")
            print(f"  Nivel Riesgo:   {data['nivel_riesgo']} (Esperado: {caso['esperado_riesgo']})")
            print(f"  Urgencia Final: {data['urgencia_final']}")
            print(f"  ¿Hubo Ajuste?:  {data['hubo_ajuste']}")
            print(f"  Confianza ML:   {data['confianza'] * 100:.2f}%")
            print(f"  Tiempo ms:      {data['tiempo_ms']} ms")

            assert data["nivel_riesgo"] == caso["esperado_riesgo"], f"Riesgo incorrecto: {data['nivel_riesgo']} != {caso['esperado_riesgo']}"
            if "esperado_final" in caso:
                assert data["urgencia_final"] == caso["esperado_final"], f"Urgencia final incorrecta: {data['urgencia_final']} != {caso['esperado_final']}"
            if "esperado_ajuste" in caso:
                assert data["hubo_ajuste"] == caso["esperado_ajuste"], f"Ajuste incorrecto: {data['hubo_ajuste']} != {caso['esperado_ajuste']}"
            assert data["tiempo_ms"] < 300.0, f"Latencia excesiva: {data['tiempo_ms']} ms"

    print("\n" + "=" * 70)
    print("¡TODAS LAS PRUEBAS DE FASE 1, FASE 2 Y FASE 3 PASARON EXITOSAMENTE!")
    print("=" * 70)


if __name__ == "__main__":
    ejecutar_pruebas()
