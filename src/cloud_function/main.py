"""Cloud Function triggered by Pub/Sub to launch Vertex AI Custom Training Job.

Triggered by Cloud Scheduler every Monday at 2 AM via Pub/Sub topic
'retrain-trigger'. Launches the toxic classifier retraining job on
Vertex AI using the same Docker image deployed to Cloud Run.
"""

import base64
import json
import datetime

from google.cloud import aiplatform

PROJECT = "mlops-toxic-classifier"
LOCATION = "us-central1"
BUCKET = "mlops-toxic-classifier-ml"
IMAGE = f"us-central1-docker.pkg.dev/{PROJECT}/mlops-containers/toxic-classifier:latest"
MACHINE_TYPE = "n1-highmem-4"
# The Custom Training Job will use the default compute service account
# which already has aiplatform.user, storage.objectAdmin, etc.
# No need to specify it explicitly.

EMBEDDINGS_CACHE_URI = f"gs://{BUCKET}/cache/nomic_embeddings_full.npz"


def trigger_retraining(event, context):
    """Entry point for Pub/Sub-triggered Cloud Function.

    Reads the Pub/Sub message, launches a Vertex AI Custom Training Job
    with the same configuration used in manual retraining, and logs the
    job name for tracking.
    """
    pubsub_message = base64.b64decode(event["data"]).decode("utf-8")
    message = json.loads(pubsub_message) if pubsub_message.strip() else {}

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    display_name = f"toxic-classifier-retrain-{timestamp}"

    aiplatform.init(project=PROJECT, location=LOCATION, staging_bucket=f"gs://{BUCKET}")

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": MACHINE_TYPE,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": IMAGE,
                "command": ["python", "src/serving/train.py"],
                "args": [
                    "--mode", "train",
                    "--project-id", PROJECT,
                    "--gcs-data-uri", f"gs://{BUCKET}/train.csv",
                    "--gcs-output-uri", f"gs://{BUCKET}/model",
                    "--gcs-embeddings-cache-uri", EMBEDDINGS_CACHE_URI,
                    "--synthetic-api-key-secret", "synthetic-api-key",
                ],
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=worker_pool_specs,
        staging_bucket=f"gs://{BUCKET}",
    )

    job.submit()

    print(f"Launched Custom Training Job: {display_name}")
    print(f"Job resource name: {job.resource_name}")
    print(f"Job state: {job.state}")

    return {"job_name": display_name, "job_resource_name": job.resource_name, "job_state": job.state.name}
