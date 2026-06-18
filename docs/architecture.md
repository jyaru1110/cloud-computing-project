# Arquitectura

## Principio de diseno: GCS como contrato

La imagen Docker es generica. No contiene el modelo ni los datos. Esto desacopla entrenamiento de serving:

- El **Custom Training Job** lee datos de GCS, entrena el modelo y escribe artefactos a GCS.
- **Cloud Run** descarga los artefactos de GCS al arrancar (cold start) y los carga en memoria.
- Cuando el pipeline reentrena, los artefactos en GCS se actualizan. La siguiente instancia de Cloud Run carga el modelo nuevo.

La misma imagen sirve para ambos modos (`--mode train` y `--mode serve`) via el entrypoint `src/serving/train.py`.

## Componentes GCP

### Cloud Build

Ejecuta el CI/CD. Construye la imagen Docker y la sube a Artifact Registry. En este proyecto los builds se ejecutaron con `gcloud builds submit --tag ...` directamente, sin usar un archivo `cloudbuild.yaml`.

### Cloud Storage

Almacena todos los artefactos que no pertenecen al codigo fuente:

| Ruta | Contenido |
|---|---|
| `train.csv` | Dataset de entrenamiento (159,571 filas, 65 MB) |
| `model/*.joblib` | 6 LinearSVC calibrados + TF-IDF vectorizador |
| `model/metadata.json` | Umbrales F2-optimal, metricas, configuracion |
| `cache/nomic_embeddings_full.npz` | Embeddings cacheados (272 MB) |
| `pipeline_templates/mlops_pipeline.json` | Pipeline KFP compilado |

### Artifact Registry

Almacena la imagen Docker `toxic-classifier:latest`. La imagen contiene Python 3.11, scikit-learn, FastAPI y el codigo fuente. No contiene el modelo.

### Vertex AI Custom Training Job

Ejecuta el entrenamiento en una VM efimera. Lee datos y embeddings de GCS, entrena 6 LinearSVC calibrados, sube artefactos a GCS. Tipo de maquina: `n1-highmem-4` (26 GB RAM). El primer intento con `n1-standard-4` (15 GB) fallo por OOM.

### Cloud Run

Sirve la API de prediccion. Lee el modelo de GCS al arrancar. Expone `/health`, `/predict`, `/model_info`. Escala a cero cuando no hay trafico. Cada prediccion computa TF-IDF localmente y llama la Synthetic API para embeddings.

### Cloud Scheduler + Pub/Sub + Cloud Function

Documentado como paso futuro para reentrenamiento automatico. No se implemento. Actualmente el reentrenamiento es manual via REST API.

### Secret Manager

Almacena la API key de Synthetic (para nomic-embed-text-v1.5). El Custom Training Job puede leer la key desde Secret Manager si no se pasa como env var.

## Flujo de datos

1. El dataset (`train.csv`) se carga a GCS manualmente.
2. El Custom Training Job lee el dataset y el cache de embeddings desde GCS.
3. Si no hay cache de embeddings, computa via Synthetic API y sube el cache a GCS.
4. Entrena 6 LinearSVC + CalibratedClassifierCV sobre TF-IDF + embeddings.
5. Sube los 7 joblib + metadata.json a GCS.
6. Cloud Run lee los artefactos de GCS al arrancar y carga el modelo en memoria.
7. Cada request a `/predict` limpia el texto, computa TF-IDF localmente, obtiene embeddings via API, concatena y predice.

## Flujo CI/CD

1. Push a GitHub.
2. Cloud Build construye la imagen Docker.
3. Cloud Build sube la imagen a Artifact Registry.
4. Opcionalmente compila y sube el pipeline KFP.
5. Opcionalmente lanza un Custom Training Job.

## Ciclo de vida MLOps

- **Ingesta de datos:** carga manual a GCS.
- **Validacion de datos:** componente KFP que verifica filas, etiquetas, textos no vacios, balance.
- **Feature engineering:** TF-IDF char_wb (2,5) + nomic-embed-text-v1.5 (768d).
- **Entrenamiento:** Vertex AI Custom Training Job.
- **Evaluacion:** AUC por etiqueta, AUC macro, gate de despliegue (>= 0.95).
- **Despliegue condicional:** si la metrica pasa, los artefactos ya estan en GCS. Cloud Run los carga al arrancar.
- **Serving:** Cloud Run con GCS-first model loading.
- **Monitoreo:** Cloud Logging, Cloud Monitoring, drift detection.
- **Reentrenamiento:** manual via REST API. Automatizado en documentacion (Cloud Scheduler semanal via Pub/Sub + Cloud Function).
- **Rollback:** versiones anteriores de artefactos en GCS, desplegar revision anterior en Cloud Run.
