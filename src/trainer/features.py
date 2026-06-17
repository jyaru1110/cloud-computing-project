"""
Feature engineering para Jigsaw Toxic Comment Classification.

Pipeline de features:
  1. Texto simple (longitud, caps, exclamaciones, etc.)
  2. TF-IDF (n-gramas 1-3, top 20k features)
  3. VADER (valencia: neg, neu, pos, compound)
  4. EMPATH (categorias tematicas, top 15 por correlacion con any_toxic)
  5. DistilBERT embeddings (768d, congelados) -- opcional

El vectorizador TF-IDF y el selector de features se ajustan en train
y se aplican en test para evitar data leakage.

Herramientas de IA utilizadas: Claude (generacion de codigo y estructura).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pointbiserialr


LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Columnas EMPATH pre-seleccionadas por relevancia para toxicidad
EMPATH_RELEVANT = [
    "swearing_terms", "negative_emotion", "ridicule", "hate",
    "death", "kill", "violence", "pain", "suffering", "aggression",
    "crime", "weapon", "shame", "fight", "anger",
]


def compute_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de texto simple: longitud, enfasis, diversidad lexica."""
    text = df["comment_text"].fillna("")
    out = pd.DataFrame(index=df.index)
    out["text_len"] = text.str.len()
    out["word_count"] = text.str.split().str.len()
    out["caps_ratio"] = text.apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    out["exclaim_ratio"] = text.str.count("!") / out["text_len"].replace(0, 1)
    out["question_ratio"] = text.str.count(r"\?") / out["text_len"].replace(0, 1)
    out["unique_word_ratio"] = text.apply(
        lambda x: len(set(x.lower().split())) / max(len(x.split()), 1)
    )
    return out


def compute_vader_features(df: pd.DataFrame, cache_path: Optional[Path] = None) -> pd.DataFrame:
    """Features de sentimiento VADER. Usa cache si existe."""
    if cache_path and cache_path.exists():
        sent_full = pd.read_csv(cache_path)
        sent_full.columns = ["sent_neg", "sent_neu", "sent_pos", "sent_compound"]
        # Si el cache tiene mas filas que df, alinear por indice posicional
        if len(sent_full) >= len(df):
            # Asumir que df es un subconjunto de los datos originales
            # Si df tiene indice original, usarlo; si no, slicing
            if df.index.max() < len(sent_full):
                return sent_full.iloc[df.index].reset_index(drop=True)
            else:
                return sent_full.iloc[:len(df)].reset_index(drop=True)
        return sent_full

    # Computar VADER si no hay cache
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    scores = df["comment_text"].fillna("").apply(sia.polarity_scores)
    sent = pd.DataFrame(scores.tolist())
    sent.columns = ["sent_neg", "sent_neu", "sent_pos", "sent_compound"]

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        sent.to_csv(cache_path, index=False)

    return sent


def compute_empath_features(
    df: pd.DataFrame,
    cache_csv: Optional[Path] = None,
    cache_parquet: Optional[Path] = None,
    top_k: int = 15,
) -> pd.DataFrame:
    """Features de categorias tematicas EMPATH. Usa cache si existe."""
    emp_full = None
    if cache_parquet and cache_parquet.exists():
        emp_full = pd.read_parquet(cache_parquet)
    elif cache_csv and cache_csv.exists():
        emp_full = pd.read_csv(cache_csv)

    if emp_full is not None:
        # Alinear cache con df por indice posicional
        if len(emp_full) >= len(df):
            if df.index.max() < len(emp_full):
                emp = emp_full.iloc[df.index].reset_index(drop=True)
            else:
                emp = emp_full.iloc[:len(df)].reset_index(drop=True)
        else:
            emp = emp_full
    else:
        from empath import Empath
        lexicon = Empath()
        scores = df["comment_text"].fillna("").apply(
            lambda x: lexicon.analyze(x, normalize=True) or {}
        )
        emp = pd.DataFrame(scores.tolist()).fillna(0)

        if cache_csv:
            cache_csv.parent.mkdir(parents=True, exist_ok=True)
            emp.to_csv(cache_csv, index=False)
        if cache_parquet:
            cache_parquet.parent.mkdir(parents=True, exist_ok=True)
            emp.to_parquet(cache_parquet, index=False)

    # Seleccionar top_k categorias por correlacion con any_toxic
    # Si any_toxic no esta disponible, usar las pre-seleccionadas
    emp_cols = [c for c in EMPATH_RELEVANT if c in emp.columns]
    if "any_toxic" in df.columns and len(emp_cols) < top_k:
        corrs = {}
        for c in emp.columns:
            if c in EMPATH_RELEVANT:
                continue
            try:
                r, _ = pointbiserialr(df["any_toxic"], emp[c])
                corrs[c] = abs(r)
            except Exception:
                pass
        extra = sorted(corrs, key=corrs.get, reverse=True)[: top_k - len(emp_cols)]
        emp_cols.extend(extra)

    selected = emp[emp_cols].copy()
    selected.columns = [f"emp_{c}" for c in selected.columns]
    return selected


def compute_distilbert_features(
    df: pd.DataFrame,
    cache_path: Optional[Path] = None,
    batch_size: int = 64,
) -> Optional[np.ndarray]:
    """Embeddings congelados de DistilBERT (768d). Usa cache si existe."""
    if cache_path and cache_path.exists():
        return np.load(cache_path)

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        print("transformers/torch no disponibles, saltando embeddings DistilBERT")
        return None

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    texts = df["comment_text"].fillna("").tolist()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            output = model(**encoded)
        # Mean pooling sobre la ultima capa oculta
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        embeddings = (output.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
        all_embeddings.append(embeddings.numpy())

    result = np.vstack(all_embeddings)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, result)

    return result


class FeaturePipeline:
    """
    Pipeline de feature engineering con fit/transform para evitar data leakage.
    El vectorizador TF-IDF se ajusta solo en train y se aplica en test.
    """

    def __init__(
        self,
        max_tfidf_features: int = 20000,
        tfidf_ngram_range: tuple = (1, 3),
        use_vader: bool = True,
        use_empath: bool = True,
        use_distilbert: bool = False,
        vader_cache: Optional[Path] = None,
        empath_cache_csv: Optional[Path] = None,
        empath_cache_parquet: Optional[Path] = None,
        distilbert_cache: Optional[Path] = None,
        empath_top_k: int = 15,
    ):
        self.max_tfidf_features = max_tfidf_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.use_vader = use_vader
        self.use_empath = use_empath
        self.use_distilbert = use_distilbert
        self.vader_cache = vader_cache
        self.empath_cache_csv = empath_cache_csv
        self.empath_cache_parquet = empath_cache_parquet
        self.distilbert_cache = distilbert_cache
        self.empath_top_k = empath_top_k

        self.tfidf = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=tfidf_ngram_range,
            sublinear_tf=True,
            min_df=3,
            max_df=0.95,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b",
        )
        self._fitted = False
        self._text_feat_cols: list[str] = []
        self._vader_feat_cols: list[str] = []
        self._empath_feat_cols: list[str] = []
        self._empath_selected_cats: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Ajustar TF-IDF y seleccionar categorias EMPATH sobre train."""
        # Limpiar texto para TF-IDF
        clean_text = df["comment_text"].fillna("").apply(self._clean_text)
        self.tfidf.fit(clean_text)

        # Features de texto simple (columnas fijas)
        self._text_feat_cols = [
            "text_len", "word_count", "caps_ratio",
            "exclaim_ratio", "question_ratio", "unique_word_ratio",
        ]

        # VADER (columnas fijas)
        if self.use_vader:
            self._vader_feat_cols = ["sent_neg", "sent_neu", "sent_pos", "sent_compound"]

        # EMPATH: seleccionar categorias
        if self.use_empath:
            if self.empath_cache_parquet and self.empath_cache_parquet.exists():
                emp = pd.read_parquet(self.empath_cache_parquet)
            elif self.empath_cache_csv and self.empath_cache_csv.exists():
                emp = pd.read_csv(self.empath_cache_csv)
            else:
                # Si no hay cache, no se pueden seleccionar categorias en fit
                emp = None

            if emp is not None and "any_toxic" in df.columns:
                corrs = {}
                for c in EMPATH_RELEVANT:
                    if c in emp.columns:
                        try:
                            r, _ = pointbiserialr(df["any_toxic"], emp[c])
                            corrs[c] = abs(r)
                        except Exception:
                            pass
                # Completar con top por varianza si faltan
                for c in emp.columns:
                    if c in corrs:
                        continue
                    try:
                        r, _ = pointbiserialr(df["any_toxic"], emp[c])
                        corrs[c] = abs(r)
                    except Exception:
                        pass

                selected = sorted(corrs, key=corrs.get, reverse=True)[: self.empath_top_k]
                self._empath_selected_cats = selected
                self._empath_feat_cols = [f"emp_{c}" for c in selected]
            else:
                self._empath_selected_cats = EMPATH_RELEVANT[: self.empath_top_k]
                self._empath_feat_cols = [f"emp_{c}" for c in self._empath_selected_cats]

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, include_labels: bool = True) -> dict:
        """
        Transformar DataFrame en diccionario con X (sparse), y (array),
        feature_names y metadatos.

        Retorna:
            {
                "X": scipy sparse matrix o numpy array,
                "y": numpy array (n, 6) si include_labels,
                "feature_names_tfidf": list,
                "feature_names_dense": list,
                "label_cols": list,
            }
        """
        if not self._fitted:
            raise RuntimeError("FeaturePipeline no ajustado. Llamar fit() primero.")

        from scipy.sparse import hstack as sparse_hstack, csr_matrix

        # 1. TF-IDF (sparse)
        clean_text = df["comment_text"].fillna("").apply(self._clean_text)
        X_tfidf = self.tfidf.transform(clean_text)

        # 2. Features densos
        dense_parts = []

        # Texto simple
        text_feats = compute_text_features(df)
        dense_parts.append(text_feats[self._text_feat_cols].values)

        # VADER
        if self.use_vader:
            vader_feats = compute_vader_features(df, self.vader_cache)
            dense_parts.append(vader_feats[self._vader_feat_cols].values)

        # EMPATH
        if self.use_empath:
            empath_feats = compute_empath_features(
                df,
                cache_csv=self.empath_cache_csv,
                cache_parquet=self.empath_cache_parquet,
                top_k=self.empath_top_k,
            )
            # Asegurar que tenemos las columnas seleccionadas
            available = [c for c in self._empath_feat_cols if c in empath_feats.columns]
            if len(available) < len(self._empath_feat_cols):
                # Rellenar con ceros las que faltan
                for c in self._empath_feat_cols:
                    if c not in empath_feats.columns:
                        empath_feats[c] = 0.0
            dense_parts.append(empath_feats[self._empath_feat_cols].values)

        # Concatenar densos
        if dense_parts:
            X_dense = np.hstack(dense_parts)
            X_dense_sparse = csr_matrix(X_dense)
        else:
            X_dense_sparse = None

        # Sparse + denso
        if X_dense_sparse is not None:
            X = sparse_hstack([X_tfidf, X_dense_sparse])
        else:
            X = X_tfidf

        # Nombres de features
        tfidf_names = list(self.tfidf.get_feature_names_out())
        dense_names = []
        dense_names.extend(self._text_feat_cols)
        if self.use_vader:
            dense_names.extend(self._vader_feat_cols)
        if self.use_empath:
            dense_names.extend(self._empath_feat_cols)

        result = {
            "X": X,
            "feature_names_tfidf": tfidf_names,
            "feature_names_dense": dense_names,
            "label_cols": LABEL_COLS,
        }

        if include_labels and all(c in df.columns for c in LABEL_COLS):
            result["y"] = df[LABEL_COLS].values

        return result

    def fit_transform(self, df: pd.DataFrame, include_labels: bool = True) -> dict:
        """Fit y transform en un solo paso."""
        self.fit(df)
        return self.transform(df, include_labels=include_labels)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Limpieza minima para TF-IDF."""
        text = text.lower()
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"ip:\d+\.\d+\.\d+\.\d+", " ", text)
        return text
