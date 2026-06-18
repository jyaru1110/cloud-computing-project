"""
Training entrypoint for Vertex AI Custom Training Job.

Modes:
  train: Download data from GCS, compute features, train model, upload artifacts
  serve: Run FastAPI serving endpoint (default)

Usage:
  python -m src.serving.app --mode train --gcs-data-uri gs://... --gcs-output-uri gs://...
  python -m src.serving.app --mode serve
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack


LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
RANDOM_STATE = 42


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"ip:\d+\.\d+\.\d+\.\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\d]", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def compute_nomic_embeddings(texts: list[str], api_key: str) -> np.ndarray:
    """Compute nomic-embed embeddings via Synthetic API."""
    import requests

    url = "https://api.synthetic.new/openai/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    BATCH_SIZE = 128

    n = len(texts)
    all_embeddings = []
    t0 = time.time()

    for i in range(0, n, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        prefixed = [f"classification: {t}" for t in batch]
        payload = {"model": "hf:nomic-ai/nomic-embed-text-v1.5", "input": prefixed}

        for attempt in range(3):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    all_embeddings.extend([d["embedding"] for d in data["data"]])
                    break
                elif resp.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                else:
                    if attempt == 2:
                        raise RuntimeError(f"API error: {resp.status_code}")
                    time.sleep(2)
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2)

        done = min(i + BATCH_SIZE, n)
        if done % 12800 == 0 or done == n:
            elapsed = time.time() - t0
            print(f"  {done}/{n} ({done/n*100:.0f}%), {elapsed:.0f}s")
        time.sleep(0.1)

    return np.array(all_embeddings, dtype=np.float32)


def train_mode(args):
    """Full training pipeline: GCS download -> features -> train -> upload."""
    from google.cloud import storage

    print("=" * 64)
    print("TRAINING MODE: Jigsaw Toxic Comment Classifier")
    print("=" * 64)

    # Get API key
    api_key = os.environ.get("SYNTHETIC_API_KEY", "")
    if not api_key:
        # Try to read from Secret Manager
        try:
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
            secret_name = f"projects/{args.project_id}/secrets/{args.synthetic_api_key_secret}/versions/latest"
            response = client.access_secret_version(request={"name": secret_name})
            api_key = response.payload.data.decode("UTF-8")
            print("Loaded API key from Secret Manager")
        except Exception as e:
            print(f"Warning: Could not load API key from Secret Manager: {e}")

    # Download data from GCS
    print(f"\nDownloading data from {args.gcs_data_uri}...")
    parts = args.gcs_data_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1]

    gcs_client = storage.Client(project=args.project_id)
    bucket = gcs_client.bucket(bucket_name)
    data = bucket.blob(blob_path).download_as_bytes()

    from io import BytesIO
    df = pd.read_csv(BytesIO(data))
    print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")

    # Preprocess
    df["clean_text"] = df["comment_text"].fillna("").apply(clean_text)

    # TF-IDF char_wb
    print("\nComputing TF-IDF char_wb features...")
    tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5),
        sublinear_tf=True, min_df=3, max_df=0.7,
    )
    X_tfidf = tfidf.fit_transform(df["clean_text"])
    print(f"TF-IDF: {X_tfidf.shape[1]} features")

    # Embeddings: try GCS cache first, then API, then skip
    emb = None
    cache_gcs_uri = args.gcs_embeddings_cache_uri
    if cache_gcs_uri:
        print(f"\nChecking GCS embeddings cache at {cache_gcs_uri}...")
        cache_parts = cache_gcs_uri.replace("gs://", "").split("/", 1)
        cache_bucket_name = cache_parts[0]
        cache_blob_path = cache_parts[1]
        cache_bucket = gcs_client.bucket(cache_bucket_name)
        cache_blob = cache_bucket.blob(cache_blob_path)
        if cache_blob.exists():
            print(f"  Cache hit! Downloading {cache_blob_path}...")
            tmp_npz = "/tmp/embeddings_cache.npz"
            cache_blob.download_to_filename(tmp_npz)
            cached = np.load(tmp_npz, allow_pickle=True)
            emb = cached["embeddings"] if "embeddings" in cached else cached[list(cached.keys())[0]]
            print(f"  Loaded embeddings: {emb.shape}")
            # Verify row count matches
            cache_n_rows = int(cached["n_rows"]) if "n_rows" in cached else emb.shape[0]
            if cache_n_rows != len(df):
                print(f"  WARNING: Cache has {cache_n_rows} rows but data has {len(df)}. Recomputing.")
                emb = None
            elif "data_hash" in cached:
                expected_hash = hex(hash(tuple(df["id"].tolist())) & 0xFFFFFFFF)[2:]
                if str(cached["data_hash"]) != expected_hash:
                    print(f"  WARNING: Data hash mismatch. Cache may be stale. Recomputing.")
                    emb = None
        else:
            print("  Cache miss.")

    if emb is None and api_key:
        print("\nComputing nomic-embed features via API...")
        texts = df["comment_text"].fillna("").tolist()
        emb = compute_nomic_embeddings(texts, api_key)
        print(f"Embeddings: {emb.shape}")
        # Upload to cache for future runs
        if cache_gcs_uri:
            print(f"  Uploading embeddings cache to {cache_gcs_uri}...")
            tmp_npz = "/tmp/embeddings_cache.npz"
            data_hash = hex(hash(tuple(df["id"].tolist())) & 0xFFFFFFFF)[2:]
            np.savez_compressed(tmp_npz, embeddings=emb, data_hash=[data_hash], n_rows=[len(df)])
            cache_bucket = gcs_client.bucket(cache_parts[0])
            cache_blob = cache_bucket.blob(cache_parts[1])
            cache_blob.upload_from_filename(tmp_npz)
            print("  Cache uploaded.")
    elif emb is None and not api_key:
        print("\nWARNING: No API key and no embeddings cache. Training without embeddings.")

    # Concatenate features
    if emb is not None:
        X = hstack([X_tfidf, emb])
    else:
        X = X_tfidf

    # Train per-label models
    y = df[LABEL_COLS].values
    model_dir = Path(args.model_dir if args.model_dir else "/tmp/model")
    model_dir.mkdir(parents=True, exist_ok=True)

    print("\nTraining per-label LinearSVC + CalibratedClassifierCV...")
    metrics = {}
    for j, label in enumerate(LABEL_COLS):
        t0 = time.time()
        svc = LinearSVC(class_weight="balanced", max_iter=5000, C=0.1, random_state=RANDOM_STATE)
        cal = CalibratedClassifierCV(svc, cv=3, method="sigmoid")
        cal.fit(X, y[:, j])
        joblib.dump(cal, model_dir / f"svc_{label}.joblib")

        prob = cal.predict_proba(X)[:, 1]
        auc = roc_auc_score(y[:, j], prob)
        metrics[label] = {"auc_train": round(auc, 4)}
        print(f"  {label}: AUC_train={auc:.4f}, {time.time()-t0:.0f}s")

    # Save TF-IDF
    joblib.dump(tfidf, model_dir / "tfidf_charwb_2_5.joblib")

    # Save metadata
    meta = {
        "model_type": "LinearSVC + CalibratedClassifierCV (sigmoid, cv=3)",
        "features": "TF-IDF char_wb (2,5)" + (" + nomic-embed-text-v1.5 (768d)" if emb is not None else ""),
        "tfidf_features": X_tfidf.shape[1],
        "embedding_features": emb.shape[1] if emb is not None else 0,
        "label_cols": LABEL_COLS,
        "metrics": metrics,
        "f2_optimal_thresholds": {
            "toxic": 0.15, "severe_toxic": 0.10, "obscene": 0.10,
            "threat": 0.15, "insult": 0.15, "identity_hate": 0.10,
        },
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Upload model artifacts to GCS
    output_parts = args.gcs_output_uri.replace("gs://", "").split("/", 1)
    out_bucket_name = output_parts[0]
    out_prefix = output_parts[1] if len(output_parts) > 1 else "model"

    out_bucket = gcs_client.bucket(out_bucket_name)
    for file_path in model_dir.iterdir():
        blob = out_bucket.blob(f"{out_prefix}/{file_path.name}")
        blob.upload_from_filename(str(file_path))
        print(f"Uploaded: gs://{out_bucket_name}/{out_prefix}/{file_path.name}")

    print(f"\nTraining complete. Macro AUC train: {np.mean([m['auc_train'] for m in metrics.values()]):.4f}")


def serve_mode():
    """Start FastAPI serving endpoint."""
    import uvicorn
    # Import the FastAPI app from predictor.py
    from src.serving.predictor import app
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))


def main():
    parser = argparse.ArgumentParser(description="Jigsaw Toxic Comment MLOps")
    parser.add_argument("--mode", choices=["train", "serve"], default="serve",
                        help="Operation mode: train or serve")
    parser.add_argument("--project-id", default=os.environ.get("PROJECT_ID", ""))
    parser.add_argument("--gcs-data-uri", default="")
    parser.add_argument("--gcs-output-uri", default="")
    parser.add_argument("--model-dir", default=os.environ.get("AIP_MODEL_DIR", "/tmp/model"))
    parser.add_argument("--synthetic-api-key-secret", default="synthetic-api-key")
    parser.add_argument("--gcs-embeddings-cache-uri", default=os.environ.get("GCS_EMBEDDINGS_CACHE_URI", ""),
                        help="GCS URI for cached embeddings (npz). If exists, skip API calls.")
    args = parser.parse_args()

    if args.mode == "train":
        train_mode(args)
    else:
        serve_mode()


if __name__ == "__main__":
    main()
