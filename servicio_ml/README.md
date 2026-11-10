# Microservicio de Inferencia de Machine Learning - Fase 1, Fase 2 & Fase 3

Microservicio REST desarrollado en **FastAPI** y **Python 3.12** para clasificar automáticamente reportes ciudadanos de infraestructura urbana (Categoría, Sentimiento y Urgencia con Motor de Reglas), diseñado específicamente para ser consumido por un backend externo en **Spring Boot (Java 21)** y persistir en la base de datos **MySQL** (`BD_REPORTES_URBANOS`).

---

## 🚀 Arquitectura y Componentes en Memoria

Durante el arranque (`lifespan` de FastAPI), el microservicio carga en memoria RAM:
1. **Modelo spaCy (`es_core_news_md`)**: Pipeline de NLP optimizado (sin `parser` ni `ner`) para tokenización y lematización.
2. **Vectorizador TF-IDF (`vectorizador_tf_idf.joblib`)**: Ajustado sobre 3.333 términos con unigramas y bigramas (`ngram_range=(1,2)`). Compartido entre los modelos.
3. **Clasificador Categoría (`modelo_naive_bayes_categoria.joblib`)**: Multinomial Naive Bayes (97.25% Accuracy, F1-Macro 0.9727).
4. **Clasificador Sentimiento (`modelo_regresion_logistica_sentimiento.joblib`)**: Regresión Logística (F1-Macro 0.5869).
5. **Clasificador Urgencia ML (`modelo_svm_lineal_urgencia.joblib`)**: LinearSVC con cálculo de confianza Softmax sobre la función de decisión (F1-Macro 0.4021).
6. **Motor de Priorización basado en Reglas (`MotorPriorizacion`)**: Refinamiento determinístico de urgencia ante señales textuales de riesgo (`CRITICO`, `ALTO`, `NINGUNO`).

---

## 📁 Estructura del Proyecto

```
servicio_ml/
├── requirements.txt                   # Dependencias del servicio
├── test_api.py                        # Suite de pruebas automatizadas (Fases 1, 2 y 3)
├── README.md                          # Documentación técnica
└── api/
    ├── __init__.py
    ├── config.py                      # Configuración y resolución de rutas de artefactos
    ├── main.py                        # Aplicación FastAPI, lifespan, CORS y health check
    ├── schemas.py                     # DTOs de validación con Pydantic v2
    ├── services/
    │   ├── __init__.py
    │   ├── preprocesador.py           # Pipeline de limpieza regex + spaCy idéntico al entrenamiento
    │   ├── clasificador_cat.py        # Inferencia y normalización para Categoría (MySQL: TBLCATEGORIA)
    │   ├── clasificador_sentimiento.py# Inferencia y normalización para Sentimiento (MySQL: TBLSENTIMIENTO)
    │   ├── motor_priorizacion.py      # Motor de reglas de priorización y escalamiento de riesgo
    │   └── clasificador_urgencia.py   # Servicio acoplado ML + Reglas (MySQL: TBLURGENCIA)
    └── routers/
        ├── __init__.py
        └── clasificacion.py           # Routers REST (/categoria, /sentimiento, /urgencia)
```

---

## ⚙️ Cómo Levantar el Servidor

Desde la raíz del proyecto (`Proyecto_Trabajo_Titulo`):

```powershell
# Levantar el servidor en el puerto 8000 con recarga automática
.venv\Scripts\uvicorn servicio_ml.api.main:app --host 0.0.0.0 --port 8000 --reload
```

* **Swagger UI / Documentación interactiva:** `http://localhost:8000/docs`
* **Redoc:** `http://localhost:8000/redoc`
* **Health Check:** `http://localhost:8000/health`

---

## 📡 Especificación de la API

### 1. Health Check
* **Método:** `GET /health`
* **Respuesta Exitosa (200 OK):**
```json
{
  "status": "healthy",
  "componentes": {
    "spacy_nlp": true,
    "vectorizador_tfidf": true,
    "modelo_naive_bayes_categoria": true,
    "modelo_regresion_logistica_sentimiento": true,
    "modelo_svm_lineal_urgencia": true,
    "motor_priorizacion_reglas": true
  },
  "version": "1.0.0"
}
```

---

### 2. Clasificar Categoría (Fase 1)
* **Método:** `POST /api/v1/clasificar/categoria`
* **Request Body:**
```json
{
  "texto": "Existe un bache enorme y peligroso en la calzada de San Martín con Ercilla, varios autos han roto neumáticos."
}
```
* **Response Body (200 OK):**
```json
{
  "categoria": "PAVIMENTACION",
  "confianza": 0.6309,
  "tiempo_ms": 7.23,
  "categoria_original": "Pavimentación"
}
```
> **Valores MySQL (`TBLCATEGORIA`):** `'PAVIMENTACION'`, `'ALCANTARILLADO'`, `'ILUMINACION'`.

---

### 3. Clasificar Sentimiento (Fase 2)
* **Método:** `POST /api/v1/clasificar/sentimiento`
* **Request Body:**
```json
{
  "texto": "La vereda del pasaje San Miguel está llena de agrietados que no permiten andar con seguridad. Esto es una vergüenza, ¿cómo pueden dejarlo así?"
}
```
* **Response Body (200 OK):**
```json
{
  "sentimiento": "ENOJADO",
  "confianza": 0.6443,
  "tiempo_ms": 8.41,
  "sentimiento_original": "Enojado"
}
```
> **Valores MySQL (`TBLSENTIMIENTO`):** `'ENOJADO'`, `'NEUTRAL'`, `'FRUSTRADO'`.

---

### 4. Clasificar Urgencia con Motor de Reglas (Fase 3)
* **Método:** `POST /api/v1/clasificar/urgencia`
* **Request Body:**
```json
{
  "texto": "En la villa El Bosque hay un hoyo en el pavimento que está mal nivelado, es peligroso para los carros",
  "categoria": "PAVIMENTACION"
}
```
* **Response Body (200 OK):**
```json
{
  "urgencia_ml": "BAJA",
  "nivel_riesgo": "ALTO",
  "urgencia_final": "MEDIA",
  "hubo_ajuste": true,
  "confianza": 0.3825,
  "tiempo_ms": 14.75,
  "urgencia_ml_original": "Baja",
  "urgencia_final_original": "Media"
}
```

> **Valores MySQL (`TBLURGENCIA` y `TBLREPORTES`):**
> * `urgencia_ml`: `'BAJA'`, `'MEDIA'`, `'ALTA'`
> * `nivel_riesgo`: `'CRITICO'`, `'ALTO'`, `'NINGUNO'`
> * `urgencia_final`: `'BAJA'`, `'MEDIA'`, `'ALTA'`
> * `hubo_ajuste`: `true` / `false`

#### Lógica de Escalamiento del Motor de Reglas:
1. **Riesgo CRITICO**: Si detecta palabras como *"cable energizado"*, *"electrocución"*, *"explosión"*, *"muerte"*, *"tapa abierta"*, etc., escala **forzadamente a ALTA**.
2. **Riesgo ALTO**: Si detecta palabras como *"peligro"*, *"peligroso"*, *"accidente"*, *"se cayó"*, etc., escala en **+1 nivel de severidad** (`BAJA` $\rightarrow$ `MEDIA`, `MEDIA` $\rightarrow$ `ALTA`).
3. **Riesgo NINGUNO**: Mantiene la predicción original de Machine Learning (`urgencia_final = urgencia_ml`, `hubo_ajuste = false`).

---

## ☕ Ejemplo de Integración en Spring Boot (Java 21)

### 1. Records DTO
```java
package cl.municipio.reportes.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

// Urgencia
public record UrgenciaPredictRequest(
    String texto,
    String categoria
) {
    public UrgenciaPredictRequest(String texto) {
        this(texto, null);
    }
}

public record UrgenciaPredictResponse(
    @JsonProperty("urgencia_ml") String urgenciaMl,
    @JsonProperty("nivel_riesgo") String nivelRiesgo,
    @JsonProperty("urgencia_final") String urgenciaFinal,
    @JsonProperty("hubo_ajuste") Boolean huboAjuste,
    Double confianza,
    @JsonProperty("tiempo_ms") Double tiempoMs,
    @JsonProperty("urgencia_ml_original") String urgenciaMlOriginal,
    @JsonProperty("urgencia_final_original") String urgenciaFinalOriginal
) {}
```

### 2. Cliente HTTP con `RestClient`
```java
package cl.municipio.reportes.client;

import cl.municipio.reportes.dto.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

@Component
public class MLInferenceClient {

    private final RestClient restClient;

    public MLInferenceClient(RestClient.Builder builder, @Value("${ml.service.base-url:http://localhost:8000}") String mlBaseUrl) {
        this.restClient = builder.baseUrl(mlBaseUrl).build();
    }

    public CategoriaPredictResponse predecirCategoria(String textoReporte) {
        return restClient.post()
                .uri("/api/v1/clasificar/categoria")
                .contentType(MediaType.APPLICATION_JSON)
                .body(new CategoriaPredictRequest(textoReporte))
                .retrieve()
                .body(CategoriaPredictResponse.class);
    }

    public SentimientoPredictResponse predecirSentimiento(String textoReporte) {
        return restClient.post()
                .uri("/api/v1/clasificar/sentimiento")
                .contentType(MediaType.APPLICATION_JSON)
                .body(new SentimientoPredictRequest(textoReporte))
                .retrieve()
                .body(SentimientoPredictResponse.class);
    }

    public UrgenciaPredictResponse predecirUrgencia(String textoReporte, String nombreCategoria) {
        return restClient.post()
                .uri("/api/v1/clasificar/urgencia")
                .contentType(MediaType.APPLICATION_JSON)
                .body(new UrgenciaPredictRequest(textoReporte, nombreCategoria))
                .retrieve()
                .body(UrgenciaPredictResponse.class);
    }
}
```

---

## 🧪 Ejecución de Pruebas Unitarias / Integración

Para validar los pipelines de las tres fases simultáneamente:

```powershell
.venv\Scripts\python.exe servicio_ml/test_api.py
```
