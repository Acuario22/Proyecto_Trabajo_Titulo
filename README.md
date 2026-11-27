# Prototipo PLN para Gestión de Incidencias Urbanas — Módulo ML e Inferencia

Módulo de Procesamiento de Lenguaje Natural (PLN), entrenamiento de modelos supervisados y microservicio local de inferencia desarrollado para la Prueba de Concepto (POC) del Trabajo de Título **"Prototipo de sistema basado en PLN para la gestión de incidencias urbanas en ciudades inteligentes"** (Ingeniería en Informática, UTEM).

- **Repositorio hermano:** [Plataforma Web Administrativa](https://github.com/Acuario22/Proyecto-Trabajo-Titulo-Web)

---

## Resumen Técnico

Todo el flujo —generación sintética local con Qwen/Ollama, preprocesamiento con spaCy, entrenamiento con Scikit-learn e inferencia— se ejecuta **estrictamente en entorno local**, resguardando el cumplimiento de la Ley N.° 19.628 sobre Protección de la Vida Privada.

### Modelos integrados en el microservicio

| Tarea                                                            | Modelo seleccionado                        |    F1-Score Macro (Test)     | Rol en la POC                            |
| :--------------------------------------------------------------- | :----------------------------------------- | :--------------------------: | :--------------------------------------- |
| **Categoría** (`Pavimentación`, `Alcantarillado`, `Iluminación`) | Naive Bayes Multinomial                    |  **0,9727** _(OOD: 0,9188)_  | Clasificación automática                 |
| **Sentimiento** (`Enojado`, `Frustrado`, `Neutral`)              | Regresión Logística                        |            0,5869            | Apoyo referencial (revisión humana)      |
| **Urgencia** (`Alta`, `Media`, `Baja`)                           | SVM Lineal (`LinearSVC`) + Motor de reglas | 0,4021 _(Recall Alta: 0,57)_ | Apoyo referencial + escalamiento crítico |

---

## Estructura del Repositorio

```text
Proyecto_Trabajo_Titulo/
├── 1.Fase de generación/        # Generación del dataset sintético local vía Ollama (qwen2.5:7b)
├── Fase de Validación - Inicial/# Auditoría estructural, balance de clases y fuga léxica
├── 2.Fase de prepocesamiento/   # Limpieza textual, tokenización, stopwords y lematización (spaCy)
├── 3.Fase de Vectorizacion/     # Ajuste y serialización de la matriz y vectorizador TF-IDF
├── 4.Fase de entrenamiento/     # Entrenamiento, GridSearchCV, modelos .joblib y motor de reglas
├── Fase de Validación - Final/  # Validación externa OOD sobre 37 reportes ciudadanos reales
├── servicio_ml/                 # Microservicio de inferencia REST en FastAPI (puerto 8000)
│   ├── api/                     # Endpoints, esquemas Pydantic y servicios de clasificación
│   ├── requirements.txt         # Dependencias mínimas del microservicio
│   └── test_api.py              # Pruebas rápidas de integración
└── requirements.txt             # Dependencias completas de entrenamiento y análisis

```

---

## Requisitos Previos

- **Python 3.12**
- Modelo de español de spaCy: `es_core_news_md`
- _(Opcional)_ **Ollama** con `qwen2.5:7b` únicamente si se desea regenerar el dataset sintético. Los datasets y modelos `.joblib` ya están incluidos en el repositorio.

---

## Instalación y Levantamiento Local

### 1. Crear entorno virtual e instalar dependencias

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r servicio_ml/requirements.txt
python -m spacy download es_core_news_md

```

**Linux / macOS:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r servicio_ml/requirements.txt
python -m spacy download es_core_news_md

```

_(Si deseas reejecutar los scripts de entrenamiento y gráficos, instala también `pip install -r requirements.txt`.)_

### 2. Levantar el microservicio de inferencia (Puerto 8000)

Desde la **raíz del repositorio**, con el entorno virtual activo:

```bash
uvicorn servicio_ml.api.main:app --host 0.0.0.0 --port 8000 --reload

```

- **Puerto:** `8000`
- **Swagger UI:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`

Para ejecutar las pruebas de integración sin levantar el servidor:

```bash
python servicio_ml/test_api.py

```

---

## Endpoints Disponibles

| Método | Ruta                             | Descripción                                               |
| ------ | -------------------------------- | --------------------------------------------------------- |
| `GET`  | `/`                              | Información general del servicio                          |
| `GET`  | `/health`                        | Estado de los modelos y componentes cargados en memoria   |
| `POST` | `/api/v1/clasificar/categoria`   | Predice la categoría de infraestructura urbana            |
| `POST` | `/api/v1/clasificar/sentimiento` | Predice el sentimiento referencial del reporte            |
| `POST` | `/api/v1/clasificar/urgencia`    | Predice urgencia ML y aplica el motor de reglas de riesgo |

**Ejemplo rápido (`POST /api/v1/clasificar/urgencia`):**

```bash
curl -X POST http://localhost:8000/api/v1/clasificar/urgencia \
  -H "Content-Type: application/json" \
  -d '{"texto": "Hay un hoyo enorme y peligroso en la calzada", "categoria": "PAVIMENTACION"}'

```

---

## Reproducción del Pipeline (Opcional)

En caso de querer regenerar los artefactos `.joblib` desde cero, ejecutar en orden desde la raíz:

1. `1.Fase de generación/Generar_datasets.py` _(requiere Ollama local)_
2. `Fase de Validación - Inicial/Estructura_validacion.py` y `analizar_dataset.py`
3. `2.Fase de prepocesamiento/preprocesamiento_texto.py` y `tocken_stepwo_lematizacion.py`
4. `3.Fase de Vectorizacion/vectorizacion_TF_IDF.py`
5. `4.Fase de entrenamiento/` (`naive_bayes_multinomial.py`, `regresion_logistica.py`, `svm_lineal.py`, `evaluacion_sistema_hibrido.py`)
6. `Fase de Validación - Final/validacion_ood_categoria.py`

---

## Autor

- **Jhon Santa Cruz** — Ingeniería en Informática, Universidad Tecnológica Metropolitana (UTEM).
