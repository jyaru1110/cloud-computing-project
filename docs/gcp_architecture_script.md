# Guion explicativo de la arquitectura GCP

---

## 1. Apertura

La plataforma que se presenta no es un modelo guardado en un notebook ni un endpoint suelto sin respaldo. Es un sistema MLOps completo desplegado en Google Cloud Platform que automatiza el ciclo de vida de un clasificador de toxicidad multi-etiqueta, desde la ingesta de datos hasta el reentrenamiento automatico. Cada componente existe por una razon de diseno verificable, y esa razon se va a explicar a continuacion.

---

## 2. Principio de diseno: GCS como contrato

El principio central de la arquitectura es que **Cloud Storage funciona como contrato entre entrenamiento y serving**. La imagen Docker es generica: no contiene el modelo ni los datos. Esto desacopla por diseno la fase de entrenamiento de la fase de inferencia.

Cuando el Custom Training Job entrena, escribe los artefactos a GCS. Cuando Cloud Run arranca, lee esos artefactos desde GCS y los carga en memoria. Si el pipeline reentrena y los artefactos cambian, la siguiente instancia de Cloud Run carga el modelo nuevo sin necesidad de reconstruir la imagen.

La misma imagen Docker sirve para ambos modos. Con `--mode train` en Vertex AI Custom Training Job. Con `--mode serve` en Cloud Run. Este diseno elimina la fuente de errores mas comun en MLOps: que la imagen de entrenamiento y la imagen de serving sean diferentes y produzcan inconsistencias.

---

## 3. Componentes de la arquitectura

### 3.1 Cloud Build

Cloud Build ejecuta el CI/CD. Construye la imagen Docker y la sube a Artifact Registry. En este proyecto los builds se ejecutaron con `gcloud builds submit --tag` directamente, sin archivo `cloudbuild.yaml`, porque la complejidad del pipeline de build no lo requeria. Si la organizacion crece y necesita pasos adicionales como tests unitarios o escaneo de vulnerabilidades, se migra a un `cloudbuild.yaml` sin cambiar la arquitectura.

### 3.2 Cloud Storage

Cloud Storage almacena todo lo que no pertenece al codigo fuente:

- `train.csv`: el dataset de entrenamiento con 159,571 comentarios.
- `model/*.joblib`: los 6 LinearSVC calibrados mas el vectorizador TF-IDF, en total 7 archivos.
- `model/metadata.json`: umbrales F2-optimal, metricas de evaluacion y configuracion del modelo.
- `cache/nomic_embeddings_full.npz`: los embeddings de 768 dimensiones cacheados para los 159k textos, 272 MB.
- `pipeline_templates/mlops_pipeline.json`: el pipeline KFP compilado listo para ejecutarse.

Cloud Storage es el unico punto de contacto entre entrenamiento y serving. Sin el, los dos mundos estarian acoplados.

### 3.3 Artifact Registry

Artifact Registry almacena la imagen Docker `toxic-classifier:latest`. La imagen contiene Python 3.11, scikit-learn, FastAPI y el codigo fuente. No contiene el modelo. Esa distincion es deliberada: si el modelo cambia, la imagen no se reconstruye. Si el codigo cambia, la imagen se reconstruye pero el modelo no se reentrena. Las dos preocupaciones evolucionan de forma independiente.

### 3.4 Vertex AI Custom Training Job

El Custom Training Job ejecuta el entrenamiento en una VM efimera. Lee datos y embeddings de GCS, entrena 6 LinearSVC calibrados con CalibratedClassifierCV, y sube los artefactos a GCS. Cuando termina, la VM desaparece.

El tipo de maquina es `n1-highmem-4` con 26 GB de RAM. El primer intento se hizo con `n1-standard-4` que tiene 15 GB, y fallo por OOM. El TF-IDF char_wb genera 194,794 features que, concatenados con los 768 features densos de embeddings para 159k filas, superan 15 GB en la fase de fit. Este es un ejemplo concreto de por que el perfilamiento de recursos es parte del diseno, no un paso posterior.

La VM efimera tiene una ventaja de costo: solo se paga mientras entrena. No hay instancia ociosa consumiendo recursos entre reentrenamientos.

### 3.5 Cloud Run

Cloud Run sirve la API de prediccion. Al arrancar, descarga los artefactos del modelo desde GCS y los carga en memoria. Expone tres endpoints: `/health` para verificar estado, `/predict` para clasificar textos, y `/model_info` para consultar metadatos.

Cloud Run escala a cero cuando no hay trafico. Esto es relevante porque un clasificador de toxicidad en un escenario real no recibe peticiones constantes. Pagar por una instancia siempre activa seria un desperdicio. Con escala a cero, el costo se alinea con el uso real.

Cada prediccion computa TF-IDF localmente y obtiene los embeddings via la Synthetic API. La latencia tipica es de 450 milisegundos.

### 3.6 Secret Manager

Secret Manager almacena la API key de Synthetic que se usa para obtener los embeddings de nomic-embed-text-v1.5. El Custom Training Job puede leer la key desde Secret Manager si no se pasa como variable de entorno. Esto evita exponer credenciales en el codigo o en los argumentos del comando.

### 3.7 Cloud Scheduler + Pub/Sub + Cloud Function

Este trio implementa el reentrenamiento automatico. Cloud Scheduler publica un mensaje semanal en un topico de Pub/Sub. La Cloud Function recibe el evento y lanza un Vertex AI Custom Training Job con los parametros configurados. El job sobreescribe los artefactos en GCS. La siguiente instancia de Cloud Run carga el modelo actualizado.

Esta arquitectura de reentrenamiento esta documentada y disenada pero no implementada en la version actual. El reentrenamiento en produccion se hace manualmente via REST API. La automatizacion es el siguiente paso natural.

---

## 4. Flujo de datos paso a paso

1. El dataset `train.csv` se carga a GCS manualmente como paso inicial.
2. El Custom Training Job lee el dataset y el cache de embeddings desde GCS.
3. Si no existe cache de embeddings, se computa via la Synthetic API y se sube el cache a GCS para futuros entrenamientos. Esto convierte un proceso de 67 minutos en una descarga de 30 segundos.
4. Se entrenan 6 LinearSVC con CalibratedClassifierCV sobre la concatenacion de TF-IDF y embeddings.
5. Se suben los 7 archivos joblib y metadata.json a GCS.
6. Cloud Run lee los artefactos de GCS al arrancar y carga el modelo en memoria.
7. Cada request a `/predict` limpia el texto, computa TF-IDF localmente, obtiene embeddings via API, concatena las dos representaciones y ejecuta la prediccion.

---

## 5. Flujo CI/CD

1. Se hace push a GitHub.
2. Cloud Build construye la imagen Docker.
3. Cloud Build sube la imagen a Artifact Registry.
4. Opcionalmente, se compila y sube el pipeline KFP a GCS.
5. Opcionalmente, se lanza un Custom Training Job para reentrenar.

Los pasos 4 y 5 son opcionales porque no todo cambio de codigo requiere reentrenamiento. Un cambio en el codigo de serving si necesita nueva imagen. Un cambio en los datos si necesita nuevo entrenamiento. Un cambio en ambos necesita ambos pasos. La arquitectura permite decidir cual flujo ejecutar segun el tipo de cambio.

---

## 6. Ciclo de vida MLOps completo

**Ingesta de datos:** carga manual a GCS como punto de partida.

**Validacion de datos:** componente KFP que verifica que existan filas, que las columnas de etiqueta esten presentes, que ningun texto este vacio y que la prevalencia de cada etiqueta este dentro del rango esperado.

**Feature engineering:** TF-IDF char_wb con n-gramas de 2 a 5 caracteres mas nomic-embed-text-v1.5 con 768 dimensiones. Dos representaciones complementarias que capturan la dimension ortografica y la dimension semantica respectivamente.

**Entrenamiento:** Vertex AI Custom Training Job en una VM efimera n1-highmem-4.

**Evaluacion:** AUC por etiqueta, AUC macro, y gate de despliegue que requiere AUC macro mayor o igual a 0.95.

**Despliegue condicional:** si la metrica pasa, los artefactos ya estan en GCS y Cloud Run los carga al arrancar. Si la metrica no pasa, el pipeline se detiene antes del despliegue.

**Serving:** Cloud Run con GCS-first model loading. Escala a cero, predice en 450 ms.

**Monitoreo:** Cloud Logging para logs de ejecucion, Cloud Monitoring para alertas de latencia y errores, y drift detection para detectar cuando la distribucion de produccion se separa de la del entrenamiento.

**Reentrenamiento:** manual via REST API en la version actual. Automatico via Cloud Scheduler semanal, Pub/Sub y Cloud Function en el diseno documentado.

**Rollback:** las versiones anteriores de artefactos se archivan en GCS. Para revertir, se copia la version anterior al directorio activo y se fuerza un cold start en Cloud Run.

---

## 7. Cierre

La arquitectura no es una coleccion arbitraria de servicios de GCP. Cada componente resuelve un problema especifico del ciclo de vida MLOps y su interaccion con los demas esta definida por el principio de GCS como contrato. Ese principio desacopla entrenamiento de serving, elimina la necesidad de reconstruir imagenes cuando el modelo cambia, y permite que el sistema se actualice de forma automatica sin intervencion humana. El resultado no es solo un clasificador con AUC 0.99. Es un sistema que se mantiene relevante porque cada pieza esta disenada para evolucionar de forma independiente.
