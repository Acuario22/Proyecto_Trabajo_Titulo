# 01_Generar_datasets.py
#OBJETIVO
#Generar n reportes de ciudadanos sinteticos en español chileno para el entrenamiento del modelo

#COMO FUNCIONA ESTE SCRIPT:
#   - Hace n llamadas a la API (n reportes por llamada)
#   - Guarda cada lote apenas lo genera (tolerante a fallos)
#   - Si el script se interrumpe, puedes retomarlo desde donde quedó sin perder los datos ya generados


#LIBRERÍAS
import requests              #Librería para hacer llamadas HTTP a la API
import pandas as pd         #Librería para manejar dataframes y exportar a CSV
import os                   #Librería para manejar archivos y directorios
import time                 #Librería para manejar tiempos de espera entre llamadas a la API
import json                 #Librería para manejar archivos JSON
import re                   #Librería para expresiones regulares (limpieza de texto)



#---------------------
#CONFIGURACIÓN PRINCIPAL
#---------------------

API_KEY = ""
archivo_salida = "dataset_reportes_sinteticos.csv"
reportes_por_lote = 10     #Número de reportes a generar por cada llamada a la API
total_reportes = 2000        #Número total de reportes a generar
espera_entre_lotes = 1   #Segundos de espera entre cada lote para evitar sobrecargar la API


#---------------------
#CATEGORIAS Y DISTRIBUCIÓN DE REPORTES
#---------------------

categorias = ["Pavimentación", "Iluminación", "Alcantarillado", "Señalética"]

#---------------------
#URGENCIA Y SENTIMIENTOS
#---------------------

urgencias = ["Baja", "Media", "Alta"]
sentimientos = ["Enojado", "Neutral", "Urgente"]


#---------------------
#FUNCION 1: CONSTRUIR EL PROMPT PARA LA API
#---------------------

def construir_prompt(categoria: str, urgencia: str, sentimiento: str, cantidad: int) -> str:
    return f"""Eres un generador de datos sintéticos para un sistema de gestión de incidencias urbanas chilenas.

    Tu tarea es generar exactamente {cantidad} reportes ciudadanos DISTINTOS con estas características:
    - Categoría: {categoria}
    - Urgencia: {urgencia}  
    - Sentimiento: {sentimiento}

    REGLAS DE ESCRITURA:
    - Escribe en español utilizado actualmente en Chile.
    - Varía el nivel de formalidad entre los reportes.
    - Aproximadamente:
        - 50% lenguaje neutral/formal
        - 35% lenguaje cotidiano
        - 15% lenguaje informal con modismos chilenos
    Los modismos deben aparecer ocasionalmente y de forma natural.
    No todos los reportes deben contener modismos.
    - Algunos reportes pueden contener errores ortográficos menores, abreviaciones o falta de tildes, como ocurre en aplicaciones reales.
    - Otros reportes deben estar correctamente redactados.
    - Varía la longitud: algunos cortos (1 oración), otros medianos (2-3 oraciones) y otros más descriptivos (4-5 oraciones).
    - No repitas estructuras de redacción.
    - Simula ciudadanos de distintas edades y niveles educacionales.
    - Aproximadamente un 10% de los reportes puede contener errores ortográficos leves.
    - Aproximadamente un 5% puede incluir abreviaciones comunes como "xq", "q", "pq", "n°", etc.
    - No abuses de estos errores.
    -Pudes usar em
    - Utiliza distintas formas de referirse a un mismo elemento urbano.
    - Ejemplos:
        * hoyo, bache,
        * tapa de alcantarillado, rejilla
        * farol,, poste de luz
        * semáforo, señal
        * vereda, acera
        - No utilices siempre la misma palabra para describir el mismo problema.
        - Las ubicaciones pueden ser genéricas o específicas.

    - Utiliza distintos tipos de referencias espaciales:
        * esquinas de calles
        * avenidas
        * pasajes
        * villas
        * plazas
        * paraderos
        * colegios
        * supermercados
        * centros de salud
        - Varía las ubicaciones entre los reportes y evita repetir los mismos lugares frecuentemente.


    
    
    REGLAS DE URGENCIA:
    - Urgencia Alta: existe riesgo físico inmediato.
    - Urgencia Media: afecta significativamente el uso normal del espacio.
    - Urgencia Baja: molestia menor sin riesgo inmediato.

    - Sentimiento Urgente: el ciudadano solicita pronta respuesta.
    - Sentimiento Enojado: expresa molestia o frustración.
    - Sentimiento Neutral: describe el problema objetivamente.

    Un reporte puede tener sentimiento urgente y urgencia baja.
    Un reporte puede tener sentimiento enojado y urgencia baja.
    Sin embargo, el texto debe ser coherente con ambas etiquetas.

    Ejemplos de reportes (NO LOS INCLUYAS EN LA RESPUESTA, SOLO SON PARA GUIARTE):
    1-Hay un hoyo pequeño en la vereda del pasaje, lleva semanas así y nadie lo ha reparado (Esta enojado pero la urgencia baja).
    2-Hay una tapa de alcantarillado abierta en plena calle, alguien podría caer en cualquier momento. Necesito que se atienda de inmediato. (El sentimiento es urgente y la urgencia alta).


    Responde ÚNICAMENTE con un array JSON válido. Sin explicaciones, sin texto antes ni después, sin comillas adicionales.
    - No uses markdown.
    - No uses ```json
    - No agregues explicaciones.

    Formato exacto de cada objeto:
    {{
        "Texto_Reporte": "texto del reporte aquí",
        "Palabras_Clave_Sugeridas": ["palabra1", "palabra2", "palabra3"]
    }}

    Genera exactamente {cantidad} objetos en el array."""

#---------------------
#FUNCION 2: LLAMAR A API DE LA IA Y OBTENER LOS REPORTES

# Envía el prompt y parsear la respuesta JSON.
# Si la respuesta viene mal formateada, lo intenta limpiar.
#---------------------

def generar_lotes(categoria: str, urgencia: str, sentimiento: str, cantidad: int) -> list:

    prompt = construir_prompt(categoria, urgencia, sentimiento, cantidad)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False,
            "Temperature": 0.9,   
        }
    )

    response_json = response.json()
    texto = response_json["response"].strip()

    # Eliminar bloques ```json
    texto = re.sub(r"^```json", "", texto)
    texto = re.sub(r"^```", "", texto)
    texto = re.sub(r"```$", "", texto)
    texto = texto.strip()

    # Buscar únicamente el array JSON
    match = re.search(r"\[.*\]", texto, re.DOTALL)

    if not match:
        raise ValueError("No se encontró un array JSON válido.")
    json_texto = match.group(0)

    #Convertimos el string JSON a una lista de python
    datos = json.loads(json_texto)
    return datos


#---------------------
#FUNCION 3: CARGAR EL PROGRESO PREVIO DESDE EL ARCHIVO CSV
#
# Si el script fue interrumpido, esta función detecta
# cuántos reportes ya se generaron para no repetirlos.
#---------------------

def cargar_progreso() -> tuple[pd.DataFrame, int]:

    if os.path.exists(archivo_salida):
        df_existente = pd.read_csv(archivo_salida, encoding="utf-8-sig" )
        reportes_ya_generados = len(df_existente)
        print(f"Retomando desde donde se quedó:  {reportes_ya_generados} reportes ya generados.")
        return df_existente, reportes_ya_generados
    else:
        print("No se encontró un archivo previo. Comenzando desde cero.")
        return pd.DataFrame(), 0
    


#---------------------
#MAIN: GENERAR LOS REPORTES Y GUARDAR EN CSV
#---------------------

def main():

    #Cargar el progreso previo (si existe)
    df_total, reportes_generados = cargar_progreso()
    todos_los_reportes = df_total.to_dict("records") if not df_total.empty else []

    #----------------------------
    #PLAN DE GENERACIÓN:
    #Queremos generar reportes distribuidos equitativamente entre las categorías, urgencias y sentimientos.
    #
    # Creamos una lista con todas las combinaciones posibles:
    # 4 categorías × 3 urgencias × 3 sentimientos = 36 combinaciones
    # Distribuimos los 10.000 reportes entre estas combinaciones.
    #----------------------------

    plan = []
    for categoria in categorias:
        for urgencia in urgencias:
            for sentimiento in sentimientos:
                plan.append({
                    "categoria":    categoria,
                    "urgencia":     urgencia,
                    "sentimiento":  sentimiento
                })
    
    #calculamos cuántos reportes corresponden a cada combinación
    # #10.000 / 36 combinaciones ≈ 277 por combinación
    reportes_por_combinacion = total_reportes // len(plan)

    print(f"\nPlan de generación:")
    print(f"   Total reportes:      {total_reportes}")
    print(f"   Combinaciones:       {len(plan)}")
    print(f"   Por combinación:     {reportes_por_combinacion}")
    print(f"   Lote por llamada:    {reportes_por_lote}")
    print(f"   Total llamadas API:  {(total_reportes // reportes_por_lote)}\n")
    
    contador_id = reportes_generados  # el ID parte desde donde quedamos

    #----------------------------
    #BUCLE PRINCIPAL DE GENERACIÓN
    # Recorremos cada combinación de categoría, urgencia y sentimiento
    #-----------------------------

    for combo in plan:
        categoria =     combo["categoria"]
        urgencia =      combo["urgencia"]
        sentimiento =   combo["sentimiento"]

        generados_en_combo = 0

        #Generamos en lotes de 50 hasta llegar al total de reportes por combinación
        while generados_en_combo < reportes_por_combinacion:

            #Si ya generamos suficientes reportes para esta combinación, paramos
            if contador_id >= total_reportes:
                break

            #calculamos cuántos reportes generar en este lote (puede ser menos de 50 si estamos cerca del límite)
            cantidad_este_lote = min(reportes_por_lote, reportes_por_combinacion - generados_en_combo)

            print(f"Generando lote | {categoria} | {urgencia} | {sentimiento} | {cantidad_este_lote} reportes...")

            try:
                #Llamamos a la función que interactúa con la API
                lote_datos = generar_lotes(categoria, urgencia, sentimiento, cantidad_este_lote)

                #Processamos cada reporte del lote
                for reporte in lote_datos:
                    contador_id += 1

                    #formateamos el ID como REP-0001, REP-0002, etc.
                    id_reporte = f"REP-{str(contador_id).zfill(5)}"

                    #Convertimos la lista de palabras clave a un string
                    palabras = reporte.get("Palabras_Clave_Sugeridas", [])
                    palabras_str = ";".join(palabras) if isinstance(palabras, list) else str(palabras)

                    todos_los_reportes.append({
                        "ID_Reporte":                   id_reporte,
                        "Texto_Reporte":                reporte.get("Texto_Reporte", ""),
                        "Palabras_Clave_Sugeridas":     palabras_str,
                        "Categoria":                    categoria,
                        "Urgencia":                     urgencia,
                        "Sentimiento":                  sentimiento
                    })

                generados_en_combo += len(lote_datos)
                #Guardamos el progreso después de cada lote para no perder datos en caso de interrupción
                df_guardado = pd.DataFrame(todos_los_reportes)
                df_guardado.to_csv(archivo_salida, index=False, encoding="utf-8-sig")
                print(f"   Guardado. Total acumulado: {contador_id/total_reportes} reportes.")

                #pausa entre lotes para no sobrecargar la API
                time.sleep(espera_entre_lotes)

            except json.JSONDecodeError as e:
                #Si la respuesta no es un JSON válido, lo registramos y seguimos con el siguiente lote
                print(f"Error de formato JSON en este lote, Saltando... ({e})")

                time.sleep(5)
                continue

            except Exception as e:
                #cualquier otro error inesperado
                print(f"Error en la API: {e}")
                print("Esperando 30 segundos antes de intentar el siguiente lote...")
                time.sleep(30)
                continue

    #FIN
    print("\n" + "="*50)
    print(f"Generación completada")
    print(f"Total reportes generados: {len(todos_los_reportes)}")
    print(f"Archivo guardado en:      {archivo_salida}")
    print("="*50)


# Punto de entrada del script
if __name__ == "__main__":
    main()

