"""
Submit the MLOps pipeline to Vertex AI using the v2 SDK directly.
Avoids the GCPC v1 schema incompatibility with Vertex AI Pipelines.
"""

import google.cloud.aiplatform as aip

PROJECT_ID = "mlops-toxic-classifier"
REGION = "us-central1"
BUCKET = "gs://mlops-toxic-classifier-ml"
PIPELINE_ROOT = f"{BUCKET}/pipeline_root"
SA = "mlops-vertex-pipeline@mlops-toxic-classifier.iam.gserviceaccount.com"

IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/mlops-containers/toxic-classifier:latest"

aip.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)

# Step 1: Submit a Custom Training Job
# This runs our Docker with --mode train
training_job = aip.CustomTrainingJob(
    display_name="jigsaw-toxic-training",
    container_image_uri=IMAGE,
    command=["--mode", "train",
             "--gcs-data-uri", f"{BUCKET}/train.csv",
             "--gcs-output-uri", f"{BUCKET}/model_artifacts",
             "--project-id", PROJECT_ID],
    model_args=[],  # no model upload from training (we upload manually)
    staging_bucket=BUCKET,
)

print("Submitting custom training job...")
print(f"  Image: {IMAGE}")
print(f"  Data:  {BUCKET}/train.csv")
print(f"  SA:    {SA}")

# Run the training job -- this launches a VM, runs training, uploads artifacts to GCS
# We skip the model upload step (model_args) because we handle it manually
job = training_job.run(
    model_display_name="jigsaw-toxic-svc-embed",
    service_account=SA,
    machine_type="n1-standard-4",
    bigquery_destination=None,  # no BQ
    args=[],  # already in command
    environment={
        "SYNTHETIC_API_KEY": "syn_aa37e9b92fa823a7b7a9eab01f24ad06",
    },
    model_labels={"source": "mlops-training", "model": "linearsvc_charwb_embed"},
)

print(f"\nTraining job state: {job.state}")
print(f"Training job: {job.display_name}")
