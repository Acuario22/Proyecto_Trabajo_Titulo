# 01_Generar_datasets.py
#OBJETIVO
#Generar n reportes de ciudadanos sinteticos en español chileno para el entrenamiento del modelo

#COMO FUNCIONA ESTE SCRIPT:
#   - Hace n llamadas a la API (n reportes por llamada)
#   - Guarda cada lote apenas lo genera (tolerante a fallos)
#   - Si el script se interrumpe, puedes retomarlo desde donde quedó sin perder los datos ya generados
#   - Valida cada reporte generado contra un vocabulario prohibido por categoría
#     (evita "fuga léxica" entre categorías, ej. mencionar "tapa de alcantarillado"
#     en un reporte de Pavimentación). Los reportes que no pasan la validación
#     se descartan y el script sigue pidiendo más hasta completar la cuota exacta.


#LIBRERÍAS
import requests             #Librería para hacer llamadas HTTP a la API
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
reportes_por_lote = 5      #Número de reportes a generar por cada llamada a la API (lotes chicos = más coherencia)
total_reportes = 2000        #Número total de reportes a generar
espera_entre_lotes = 1.0   #Segundos de espera entre cada lote para evitar sobrecargar la API
max_intentos_por_combo = 15  #Tope de llamadas extra por combinación antes de rendirse (evita loops infinitos)


#---------------------
#CATEGORIAS Y DISTRIBUCIÓN DE REPORTES
#---------------------

categorias = ["Pavimentación", "Alcantarillado", "Iluminación"]

#---------------------
#URGENCIA Y SENTIMIENTOS
#---------------------

urgencias = ["Baja", "Media", "Alta"]
sentimientos = ["Enojado", "Neutral", "Frustrado"]


#---------------------
#VOCABULARIO PROHIBIDO POR CATEGORÍA (auditoría de fuga léxica)
#---------------------
# Si un reporte de una categoría menciona palabras de esta lista, se descarta
# y se vuelve a generar. Esto evita, por ejemplo, que un reporte de
# "Pavimentación" mencione "tapa de alcantarillado".

vocabulario_prohibido = {
    "Pavimentación": ["poste de luz", "farol", "tapa de alcantarillado", "cañería",
                       "fuga de agua", "rejilla", "alcantarilla"],
    "Alcantarillado": ["bache", "hoyo en la calzada", "poste de luz", "farol",
                        "vereda dañada", "acera dañada"],
    "Iluminación": ["bache", "hoyo en la calzada", "tapa de alcantarillado", "cañería",
                     "vereda dañada", "acera dañada", "alcantarilla"]
}


#---------------------
#FUNCION 1: CONSTRUIR EL PROMPT PARA LA API
#---------------------

def construir_prompt(categoria: str, urgencia: str, sentimiento: str, cantidad: int) -> str:
    return f"""
    Eres un generador de datos sintéticos para un sistema de gestión de incidencias urbanas chilenas.

    Tu tarea es generar exactamente {cantidad} reportes ciudadanos DISTINTOS con estas características:
    - Categoría: {categoria}
    - Urgencia: {urgencia}  
    - Sentimiento: {sentimiento}

    VOCABULARIO PERMITIDO SEGÚN CATEGORÍA:
    - Pavimentación: el reporte debe tratar sobre baches, hoyos en la calzada,
    veredas o aceras dañadas, pavimento agrietado o hundido, lomos de toro mal
    instalados, desniveles peligrosos en la vía.
    NO menciones postes de luz, faroles, tapas de alcantarillado, cañerías ni fugas de agua.
    - Alcantarillado: el reporte debe tratar sobre tapas de alcantarillado sueltas,
    quebradas o abiertas, rejillas dañadas, inundaciones con aguas de mal olor,
    fugas de agua, cañerías rotas o mal uso del sistema de alcantarillado.
    NO menciones baches, hoyos en la calzada, postes de luz ni faroles.
    - Iluminación: el reporte debe tratar sobre postes de luz caídos, dañados o
    desequilibrados, faroles apagados, focos quemados, semáforos sin funcionar,
    falta de iluminación en la vía pública.
    NO menciones baches, hoyos en la calzada, tapas de alcantarillado ni cañerías.

    REGLAS DE ESCRITURA:
    - Escribe en español utilizado actualmente en Chile.
    - Varía el nivel de formalidad entre los reportes.
    - La mayoría debe ser neutral/formal, algunos cotidianos y un porcentaje menor con modismos informales
    Los modismos deben aparecer ocasionalmente y de forma natural.
    No todos los reportes deben contener modismos.
    - Algunos reportes pueden contener errores ortográficos menores, abreviaciones o falta de tildes, como ocurre en aplicaciones reales.
    - Otros reportes deben estar correctamente redactados.
    - Varía la longitud: algunos cortos (1 oración), otros medianos (2-3 oraciones) y otros más descriptivos (4-5 oraciones).
    - No repitas estructuras de redacción.
    - Simula ciudadanos de distintas edades y niveles educacionales.
    - Incluye ocasionalmente errores ortográficos leves o abreviaciones comunes("xq", "q", "pq", "n°", etc), pero nunca comprometas la comprensión del reporte.
    - No abuses de estos errores.
    - Utiliza distintas formas de referirse al mismo elemento urbano, PERO SIN
    salirte de la categoría asignada:
        * Si es Pavimentación: hoyo / bache, vereda / acera, calzada / pavimento
        * Si es Alcantarillado: tapa de alcantarillado / rejilla, cañería / ducto,
          alcantarilla / desagüe
        * Si es Iluminación: farol / poste de luz, foco / ampolleta, semáforo / señal lumínica
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
    - Urgencia Alta: Existe riesgo físico, peligro inminente de accidente o interrupción crítica de la seguridad pública.
    - Urgencia Media: Afecta significativamente el uso normal del espacio urbano pero no es de riesgo inminente.
    - Urgencia Baja: Molestia menor estética o funcional sin peligro físico.

    REGLAS DE SENTIMIENTO:
    - Sentimiento Frustrado: Expresa cansancio o resignación por un problema que persiste en el tiempo sin solución, o por falta de respuesta previa del municipio (ej. "ya reporté esto antes y nada", "llevamos meses así").
    - Sentimiento Enojado: Expresa molestia enérgica, reclamo directo o indignación con el servicio municipal (ej. "esto es una vergüenza", "exijo que lo arreglen ya").
    - Sentimiento Neutral: Relato objetivo y descriptivo de la falla en terreno, sin carga emocional evidente.

    Un reporte puede ser calificado de Urgencia Baja pero con Sentimiento Enojado
    (ej. un bache en un pasaje que molesta al vecino pero no provoca choques)

    Ejemplos de estilo y estructura (NO LOS INCLUYAS EN LA RESPUESTA, SOLO SON PARA GUIARTE):
    1-Un reporte de Urgencia Baja con Sentimiento Enojado describe una molestia que
      lleva tiempo sin resolverse pero no representa peligro inmediato.
    2-Un reporte de Urgencia Alta con Sentimiento Enojado describe una situación de
      riesgo físico real y exige atención inmediata.
    3-Un reporte de Sentimiento Frustrado menciona que ya se había reportado antes
      y no hubo respuesta.
    4-Un reporte de Sentimiento Neutral es descriptivo y objetivo, sin carga emocional.

    VERIFICACIÓN FINAL (obligatoria antes de responder):
    Antes de generar tu respuesta, revisa cada reporte y confirma que:
    - El texto NO contiene ninguna palabra de las categorías prohibidas para "{categoria}".
    - Si detectas que un reporte menciona un elemento de otra categoría, corrígelo o
      reemplázalo por un elemento válido de "{categoria}" antes de incluirlo en la respuesta.

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
            "options": {
                "temperature": 0.9
            }
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
#FUNCION 2B: VALIDAR COHERENCIA LÉXICA DE UN REPORTE
#---------------------
# Devuelve True si el texto NO contiene vocabulario prohibido para su categoría

def validar_coherencia(texto: str, categoria: str) -> bool:
    texto_lower = str(texto).lower()
    prohibidas = vocabulario_prohibido.get(categoria, [])
    return not any(palabra in texto_lower for palabra in prohibidas)


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
    # Creamos una lista con todas las combinaciones posibles:
    # 3 categorías × 3 urgencias × 3 sentimientos = 27 combinaciones
    # Distribuimos los 2000 reportes entre estas combinaciones,
    # repartiendo el resto para que la suma total sea EXACTA.
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

    base = total_reportes // len(plan)
    resto = total_reportes % len(plan)
    for i, combo in enumerate(plan):
        combo["cantidad"] = base + (1 if i < resto else 0)

    #--------------------------------------------------------------
    # AJUSTE POR REANUDACIÓN: contamos cuántos reportes YA existen
    # para cada combinación específica (categoria+urgencia+sentimiento)
    # y restamos eso de la cuota, para completar solo lo que falta
    # en vez de reintentar todo el plan desde el principio.
    #--------------------------------------------------------------
    if not df_total.empty:
        conteo_existente = df_total.groupby(
            ["Categoria", "Urgencia", "Sentimiento"]
        ).size()
    else:
        conteo_existente = pd.Series(dtype=int)

    for combo in plan:
        clave = (combo["categoria"], combo["urgencia"], combo["sentimiento"])
        ya_generados = conteo_existente.get(clave, 0)
        combo["ya_generados"] = ya_generados
        combo["cantidad_faltante"] = max(0, combo["cantidad"] - ya_generados)

    faltantes_total = sum(c["cantidad_faltante"] for c in plan)

    print(f"\nPlan de generación:")
    print(f"   Total reportes objetivo:  {total_reportes}")
    print(f"   Ya existentes:             {len(todos_los_reportes)}")
    print(f"   Faltantes a generar:       {faltantes_total}")
    print(f"   Combinaciones:             {len(plan)}")
    print(f"   Por combinación (cuota):   {base} (+1 extra en {resto} combinaciones)")
    print(f"   Lote por llamada:          {reportes_por_lote}")
    print(f"   Validación léxica:         ACTIVADA (descarta y regenera reportes inconsistentes)\n")

    # El ID sigue siendo correlativo desde el último existente
    contador_id = reportes_generados
    total_descartados_global = 0

    #----------------------------
    #BUCLE PRINCIPAL DE GENERACIÓN
    #-----------------------------

    for combo in plan:
        categoria =     combo["categoria"]
        urgencia =      combo["urgencia"]
        sentimiento =   combo["sentimiento"]
        cantidad_combo = combo["cantidad_faltante"]   # <-- solo lo que falta, no la cuota completa

        if cantidad_combo <= 0:
            continue  # esta combinación ya está completa, saltamos directo a la siguiente

        generados_en_combo = 0
        intentos_combo = 0

        #Generamos en lotes hasta llegar al total de reportes por combinación
        while generados_en_combo < cantidad_combo:

            #Si ya generamos suficientes reportes en total, paramos
            if contador_id >= total_reportes:
                break

            #Tope de seguridad para no quedar en loop infinito si el modelo falla siempre
            if intentos_combo >= max_intentos_por_combo:
                print(f"   ⚠ Se alcanzó el tope de intentos para esta combinación "
                      f"({categoria}|{urgencia}|{sentimiento}). Quedaron "
                      f"{cantidad_combo - generados_en_combo} reportes sin generar en esta cuota.")
                break

            cantidad_este_lote = min(reportes_por_lote, cantidad_combo - generados_en_combo)

            print(f"Generando lote | {categoria} | {urgencia} | {sentimiento} | {cantidad_este_lote} reportes...")

            try:
                lote_datos = generar_lotes(categoria, urgencia, sentimiento, cantidad_este_lote)

                #Filtramos los reportes que no pasan la validación de coherencia léxica
                lote_validado = [
                    r for r in lote_datos
                    if validar_coherencia(r.get("Texto_Reporte", ""), categoria)
                ]

                descartados = len(lote_datos) - len(lote_validado)
                if descartados > 0:
                    total_descartados_global += descartados
                    print(f"   ⚠ Descartados {descartados}/{len(lote_datos)} reportes "
                          f"por fuga léxica de categoría.")

                #Processamos cada reporte válido del lote
                for reporte in lote_validado:
                    #Si ya llegamos a la cuota exacta de esta combinación, no seguimos agregando
                    if generados_en_combo >= cantidad_combo or contador_id >= total_reportes:
                        break

                    contador_id += 1
                    id_reporte = f"REP-{str(contador_id).zfill(5)}"

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

                    generados_en_combo += 1

                #Guardamos el progreso después de cada lote para no perder datos en caso de interrupción
                df_guardado = pd.DataFrame(todos_los_reportes)
                df_guardado.to_csv(archivo_salida, index=False, encoding="utf-8-sig")
                print(f"   Guardado. Total acumulado: {contador_id}/{total_reportes} reportes "
                      f"(combinación: {generados_en_combo}/{cantidad_combo}).")

                intentos_combo += 1
                time.sleep(espera_entre_lotes)

            except json.JSONDecodeError as e:
                print(f"Error de formato JSON en este lote, Saltando... ({e})")
                intentos_combo += 1
                time.sleep(5)
                continue

            except Exception as e:
                print(f"Error en la API: {e}")
                print("Esperando 30 segundos antes de intentar el siguiente lote...")
                time.sleep(30)
                continue

    #FIN
    print("\n" + "="*50)
    print(f"Generación completada")
    print(f"Total reportes generados:  {len(todos_los_reportes)}")
    print(f"Total descartados por fuga léxica: {total_descartados_global}")
    print(f"Archivo guardado en:       {archivo_salida}")
    print("="*50)


# Punto de entrada del script
if __name__ == "__main__":
    main()