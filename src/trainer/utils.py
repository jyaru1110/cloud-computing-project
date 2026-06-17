"""
Utilidades de carga y preprocesamiento para el pipeline de entrenamiento.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_from_gcs(bucket_name: str, blob_path: str, local_cache: Path | None = None) -> pd.DataFrame:
    """Descarga un CSV de GCS y lo carga, con cache local opcional."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if local_cache and local_cache.exists():
        return pd.read_csv(local_cache)

    if local_cache:
        local_cache.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_cache))

    data = pd.read_csv(local_cache) if local_cache else pd.read_csv(
        f"gs://{bucket_name}/{blob_path}"
    )
    return data
