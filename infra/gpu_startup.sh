#!/bin/bash
set -euo pipefail

# Install CUDA driver (required for GPU)
apt-get update
apt-get install -y python3-pip python3-venv git

# Install NVIDIA driver + CUDA
apt-get install -y linux-modules-extra-$(uname -r)
echo "NVIDIA" > /etc/driver_modules
apt-get install -y nvidia-driver-535 || apt-get install -y nvidia-driver-525 || true

# Create work dir
mkdir -p /opt/analysis
cd /opt/analysis

# Install uv + Python
pip3 install uv
uv init --bare
uv add pandas scipy scikit-learn matplotlib seaborn nltk wordcloud statsmodels pysentimiento empath

# Download data from GCS
gcloud storage cp gs://mlops-toxic-comments-ml/data/train.csv /opt/analysis/train.csv

# Run the analysis
cat > run_analysis.py << 'PYEOF'
import os, sys, time, json, warnings, pathlib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project
sys.path.insert(0, "/opt/analysis")

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

print("=" * 64)
print("ANALISIS DE SENTIMIENTO AVANZADO - GPU INSTANCE")
print("=" * 64)

# 1. Load data
df = pd.read_csv("/opt/analysis/train.csv")
print(f"Dataset cargado: {df.shape[0]} filas")

# Base features
df["text_len"] = df["comment_text"].str.len()
df["word_count"] = df["comment_text"].str.split().str.len()
df["caps_ratio"] = df["comment_text"].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
df["exclaim_ratio"] = df["comment_text"].str.count("!") / df["text_len"]
df["question_ratio"] = df["comment_text"].str.count(r"\?") / df["text_len"]
df["unique_word_ratio"] = df["comment_text"].apply(lambda x: len(set(x.lower().split())) / max(len(x.split()), 1))
df["any_toxic"] = (df[LABEL_COLS].sum(axis=1) > 0).astype(int)

# 2. VADER
import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
t0 = time.time()
scores = [sia.polarity_scores(t) for t in df["comment_text"]]
print(f"VADER: {time.time()-t0:.0f}s")
sent_df = pd.DataFrame(scores)
sent_df.columns = ["sent_neg", "sent_neu", "sent_pos", "sent_compound"]
df = pd.concat([df, sent_df], axis=1)

# 3. pysentimiento emotion (BATCHED with GPU if available)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

model_name = "pysentimiento/robertuito-emotion-analysis"
print(f"Cargando modelo {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = model.to(device)

model.eval()
id2label = model.config.id2label

t0 = time.time()
BATCH_SIZE = 512
all_probas = []
all_preds = []
texts = df["comment_text"].tolist()

for b in range(0, len(texts), BATCH_SIZE):
    batch = texts[b:b+BATCH_SIZE]
    enc = tokenizer(batch, truncation=True, max_length=128, padding=True, return_tensors="pt")
    if device == "cuda":
        enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        probs = F.softmax(out.logits, dim=-1).cpu().numpy()
    all_probas.append(probs)
    all_preds.extend(out.logits.argmax(dim=-1).cpu().numpy().tolist())
    if (b // BATCH_SIZE + 1) % 20 == 0:
        print(f"  emotion batch {b+BATCH_SIZE}/{len(texts)} ({time.time()-t0:.0f}s)")

all_probas = np.concatenate(all_probas, axis=0)
emotion_labels = sorted(id2label.values())
emo_rows = []
for i in range(len(texts)):
    row = {f"emo_{id2label[j]}": float(all_probas[i, j]) for j in id2label}
    row["emo_pred"] = id2label[all_preds[i]]
    emo_rows.append(row)
emo_df = pd.DataFrame(emo_rows)
df = pd.concat([df, emo_df], axis=1)
print(f"pysentimiento emotion: {time.time()-t0:.0f}s (device={device})")

# 4. empath
from empath import Empath
lex = Empath()
sample = lex.analyze("test", normalize=True)
all_categories = sorted(sample.keys())

t0 = time.time()
empath_rows = []
for i, text in enumerate(df["comment_text"]):
    analysis = lex.analyze(text, normalize=True)
    empath_rows.append({f"emp_{c}": analysis.get(c, 0.0) for c in all_categories})
    if (i+1) % 20000 == 0:
        print(f"  empath {i+1}/{len(df)} ({time.time()-t0:.0f}s)")
empath_df = pd.DataFrame(empath_rows)
df = pd.concat([df, empath_df], axis=1)
print(f"empath: {time.time()-t0:.0f}s")

# 5. Save results
df.to_parquet("/opt/analysis/results_full.parquet", index=False)
print("Resultados guardados en /opt/analysis/results_full.parquet")
print(f"Shape: {df.shape}")

# Upload to GCS
import subprocess
subprocess.run(["gcloud", "storage", "cp", "/opt/analysis/results_full.parquet",
                "gs://mlops-toxic-comments-ml/analysis_output/results_full.parquet"], check=False)
print("Subido a GCS")

# Signal completion
open("/opt/analysis/DONE", "w").close()
print("ANALISIS COMPLETADO")
PYEOF

# Run in background
cd /opt/analysis
uv run python run_analysis.py > /opt/analysis/analysis.log 2>&1 &
echo "Analysis started in background. Check /opt/analysis/analysis.log"
