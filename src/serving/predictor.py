"""
Toxic Comment Classification API.

Serving endpoint that accepts raw text, computes TF-IDF char_wb features
locally and nomic-embed embeddings via Synthetic API, concatenates them,
and runs 6 calibrated LinearSVC models to produce per-label probabilities.

Model loading: downloads artifacts from GCS at startup if not present locally.
This decouples the Docker image (generic) from the model (versioned in GCS).
The pipeline writes to GCS, Cloud Run reads from GCS. GCS is the contract.
"""

from __future__ import annotations

import os, re, time, json
from typing import Optional

import numpy as np
import joblib
from scipy.sparse import hstack
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ============================================================
# Configuration
# ============================================================

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

SYNTHETIC_API_KEY = os.environ.get("SYNTHETIC_API_KEY", "")
SYNTHETIC_API_URL = "https://api.synthetic.new/openai/v1/embeddings"
EMBEDDING_MODEL = "hf:nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768
EMBEDDING_TASK_TYPE = "classification"

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
GCS_MODEL_URI = os.environ.get("GCS_MODEL_URI", "")  # e.g. gs://bucket/model
PROJECT_ID = os.environ.get("PROJECT_ID", os.environ.get("GCP_PROJECT", ""))

# F2-optimal thresholds from training
THRESHOLDS = {
    "toxic": 0.15,
    "severe_toxic": 0.10,
    "obscene": 0.10,
    "threat": 0.15,
    "insult": 0.15,
    "identity_hate": 0.10,
}

# ============================================================
# Text preprocessing
# ============================================================

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"ip:\d+\.\d+\.\d+\.\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\d]", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


# ============================================================
# Embedding client
# ============================================================

def get_embeddings(texts: list[str], api_key: str) -> np.ndarray:
    """Compute nomic-embed embeddings via Synthetic API."""
    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    prefixed = [f"{EMBEDDING_TASK_TYPE}: {t}" for t in texts]
    payload = {
        "model": EMBEDDING_MODEL,
        "input": prefixed,
    }

    for attempt in range(3):
        try:
            resp = requests.post(SYNTHETIC_API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                embeddings = [d["embedding"] for d in data["data"]]
                return np.array(embeddings, dtype=np.float32)
            elif resp.status_code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                raise HTTPException(
                    status_code=502,
                    detail=f"Embedding API error: {resp.status_code} {resp.text[:200]}"
                )
        except requests.exceptions.Timeout:
            if attempt == 2:
                raise HTTPException(status_code=504, detail="Embedding API timeout")
            time.sleep(1)

    raise HTTPException(status_code=502, detail="Embedding API failed after 3 retries")


# ============================================================
# Model loading: GCS-first, local fallback
# ============================================================

class ToxicityPredictor:
    def __init__(self, model_dir: str, gcs_model_uri: str = "", project_id: str = ""):
        self.model_dir = model_dir
        self.gcs_model_uri = gcs_model_uri
        self.project_id = project_id
        self.tfidf = None
        self.models = {}
        self.thresholds = THRESHOLDS
        self.model_source = "unknown"

        # Step 1: If no local model, download from GCS
        if not os.path.exists(os.path.join(model_dir, "tfidf_charwb_2_5.joblib")):
            if gcs_model_uri:
                self._download_from_gcs(gcs_model_uri, project_id)
                self.model_source = f"gcs:{gcs_model_uri}"
            else:
                raise RuntimeError(
                    f"No model found in {model_dir} and no GCS_MODEL_URI set. "
                    "Either bake model into image or set GCS_MODEL_URI env var."
                )
        else:
            self.model_source = f"local:{model_dir}"

        # Step 2: Load models into memory
        self._load_models()

    def _download_from_gcs(self, gcs_uri: str, project_id: str):
        """Download model artifacts from GCS to local model_dir."""
        from google.cloud import storage

        print(f"Downloading model from GCS: {gcs_uri}")
        parts = gcs_uri.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        client = storage.Client(project=project_id) if project_id else storage.Client()
        bucket = client.bucket(bucket_name)

        os.makedirs(self.model_dir, exist_ok=True)
        count = 0
        for blob in bucket.list_blobs(prefix=prefix):
            if blob.name.endswith((".joblib", ".json")):
                filename = os.path.basename(blob.name)
                local_path = os.path.join(self.model_dir, filename)
                blob.download_to_filename(local_path)
                print(f"  Downloaded: {filename} ({blob.size / 1024:.0f} KB)")
                count += 1

        if count == 0:
            raise RuntimeError(f"No .joblib or .json files found at gs://{bucket_name}/{prefix}")
        print(f"Downloaded {count} model files from GCS")

    def _load_models(self):
        t0 = time.time()
        tfidf_path = os.path.join(self.model_dir, "tfidf_charwb_2_5.joblib")
        self.tfidf = joblib.load(tfidf_path)

        for label in LABEL_COLS:
            model_path = os.path.join(self.model_dir, f"svc_{label}.joblib")
            self.models[label] = joblib.load(model_path)

        # Load thresholds from metadata if available
        meta_path = os.path.join(self.model_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if "f2_optimal_thresholds" in meta:
                self.thresholds = meta["f2_optimal_thresholds"]

        elapsed = time.time() - t0
        n_features = self.tfidf.get_feature_names_out().shape[0] if hasattr(self.tfidf, 'get_feature_names_out') else 0
        print(f"Loaded {len(self.models)} models + TF-IDF ({n_features} features) from {self.model_source} in {elapsed:.1f}s")

    def predict(self, texts: list[str], api_key: str) -> list[dict]:
        """Predict toxicity probabilities for a batch of texts."""
        t_total = time.time()

        # Step 1: Clean text for TF-IDF
        clean_texts = [clean_text(t) for t in texts]

        # Step 2: TF-IDF features (local, fast)
        t0 = time.time()
        X_tfidf = self.tfidf.transform(clean_texts)
        tfidf_time = time.time() - t0

        # Step 3: Embedding features (API call)
        t0 = time.time()
        X_emb = get_embeddings(texts, api_key)
        emb_time = time.time() - t0

        # Step 4: Concatenate
        X = hstack([X_tfidf, X_emb]).tocsr()

        # Step 5: Predict per label
        t0 = time.time()
        results = []
        for i, text in enumerate(texts):
            row = X[i]
            probs = {}
            labels = {}
            for label in LABEL_COLS:
                prob = float(self.models[label].predict_proba(row)[0, 1])
                probs[label] = round(prob, 4)
                labels[label] = prob >= self.thresholds.get(label, 0.5)
            results.append({
                "text": text[:200] + "..." if len(text) > 200 else text,
                "probabilities": probs,
                "labels": labels,
                "thresholds": self.thresholds,
            })
        predict_time = time.time() - t0

        total_time = time.time() - t_total
        print(f"Predicted {len(texts)} texts: tfidf={tfidf_time:.2f}s, emb={emb_time:.2f}s, predict={predict_time:.2f}s, total={total_time:.2f}s")
        return results


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="Toxic Comment Classification API",
    description="Multi-label toxicity classifier: LinearSVC + char_wb TF-IDF + nomic-embed",
    version="2.0.0",
)

predictor: Optional[ToxicityPredictor] = None


@app.on_event("startup")
def startup():
    global predictor
    predictor = ToxicityPredictor(
        model_dir=MODEL_DIR,
        gcs_model_uri=GCS_MODEL_URI,
        project_id=PROJECT_ID,
    )


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=128, description="Comments to classify")
    api_key: Optional[str] = Field(None, description="Synthetic API key (or set SYNTHETIC_API_KEY env var)")


class PredictionResult(BaseModel):
    text: str
    probabilities: dict
    labels: dict
    thresholds: dict


class PredictResponse(BaseModel):
    predictions: list[PredictionResult]
    model_info: dict
    latency_ms: int


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": predictor is not None,
        "n_labels": len(LABEL_COLS) if predictor else 0,
        "model_source": predictor.model_source if predictor else "not loaded",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    api_key = request.api_key or SYNTHETIC_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key provided. Set SYNTHETIC_API_KEY env var or pass api_key in request."
        )

    t0 = time.time()
    predictions = predictor.predict(request.texts, api_key)
    latency_ms = int((time.time() - t0) * 1000)

    return PredictResponse(
        predictions=predictions,
        model_info={
            "algorithm": "LinearSVC + CalibratedClassifierCV (sigmoid)",
            "features": "TF-IDF char_wb (2,5) + nomic-embed-text-v1.5 (768d)",
            "n_labels": len(LABEL_COLS),
            "labels": LABEL_COLS,
            "auc_macro": 0.9903,
            "model_source": predictor.model_source,
        },
        latency_ms=latency_ms,
    )


@app.get("/model_info")
def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return {
        "algorithm": "LinearSVC + CalibratedClassifierCV (sigmoid, cv=3)",
        "features": "TF-IDF char_wb (2,5) + nomic-embed-text-v1.5 (classification, 768d)",
        "tfidf_features": predictor.tfidf.get_feature_names_out().shape[0] if hasattr(predictor.tfidf, 'get_feature_names_out') else "N/A",
        "embedding_features": EMBEDDING_DIM,
        "labels": LABEL_COLS,
        "thresholds": predictor.thresholds,
        "model_source": predictor.model_source,
        "metrics": {
            "auc_macro": 0.9903,
            "f1_macro": 0.6388,
        },
    }
