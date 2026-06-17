"""
Modelo multi-etiqueta para Jigsaw Toxic Comment Classification.

Estrategia: classifier chains con LightGBM por etiqueta.

El orden de la cadena respeta la jerarquia documentada en el EDA:
  1. toxic (etiqueta mas prevalente, 9.58%)
  2. obscene (5.29%, co-ocurre con toxic 73%)
  3. insult (4.94%, co-ocurre con obscene 73%)
  4. severe_toxic (1.00%, subconjunto estricto de toxic)
  5. identity_hate (0.88%, parcialmente independiente)
  6. threat (0.30%, la mas rara y dificil)

Cada modelo recibe como features adicionales las probabilidades
predichas por los modelos anteriores en la cadena. Esto modela
explicitamente la estructura de dependencia entre etiquetas.

Para etiquetas con prevalencia < 2%, se usa scale_pos_weight
inverso a la frecuencia para compensar el desbalance extremo.

Herramientas de IA utilizadas: Claude (generacion de codigo y estructura).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack, issparse, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    average_precision_score, brier_score_loss,
)
import lightgbm as lgb

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Orden de la cadena: respeta la jerarquia del EDA
CHAIN_ORDER = ["toxic", "obscene", "insult", "severe_toxic", "identity_hate", "threat"]

# Hiperparametros por etiqueta
# Las etiquetas raras necesitan arboles mas profundos y pesos mas altos
LABEL_CONFIG = {
    "toxic": {
        "n_estimators": 500,
        "max_depth": 7,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
    },
    "severe_toxic": {
        "n_estimators": 800,
        "max_depth": 9,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_child_samples": 20,
    },
    "obscene": {
        "n_estimators": 500,
        "max_depth": 7,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
    },
    "threat": {
        "n_estimators": 1000,
        "max_depth": 11,
        "learning_rate": 0.02,
        "num_leaves": 255,
        "min_child_samples": 5,
    },
    "insult": {
        "n_estimators": 500,
        "max_depth": 7,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
    },
    "identity_hate": {
        "n_estimators": 800,
        "max_depth": 9,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_child_samples": 10,
    },
}

BASE_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Peso inverso a la frecuencia para compensar desbalance."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


class ClassifierChainLGBM:
    """
    Cadena de clasificadores LightGBM que respeta la jerarquia de etiquetas.

    Cada etiqueta se predice en orden, y la probabilidad predicha de las
    etiquetas anteriores se agrega como feature adicional. Esto modela
    la estructura de dependencia documentada en el EDA.
    """

    def __init__(self, chain_order: Optional[list[str]] = None):
        self.chain_order = chain_order or CHAIN_ORDER
        self.models: dict[str, lgb.LGBMClassifier] = {}
        self.feature_names: list[str] = []
        self.chain_features: list[str] = []
        self._fitted = False

    def fit(
        self,
        X,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> "ClassifierChainLGBM":
        """
        Entrenar la cadena de modelos.

        Args:
            X: features (sparse o denso), shape (n, d)
            y: etiquetas, shape (n, 6), columnas en orden de LABEL_COLS
            feature_names: nombres de features originales
        """
        if feature_names is not None:
            self.feature_names = list(feature_names)

        n_chain_features = len(self.chain_order)
        self.chain_features = [f"chain_{label}" for label in self.chain_order]

        # Preparar array de probabilidades de la cadena
        chain_probs = np.zeros((X.shape[0], n_chain_features))

        for i, label in enumerate(self.chain_order):
            label_idx = LABEL_COLS.index(label)
            y_label = y[:, label_idx]

            # Agregar probabilidades anteriores como features
            if i > 0:
                if issparse(X):
                    X_aug = sparse_hstack([X, csr_matrix(chain_probs[:, :i])])
                else:
                    X_aug = np.hstack([X, chain_probs[:, :i]])
            else:
                X_aug = X

            # Hiperparametros especificos por etiqueta
            config = LABEL_CONFIG.get(label, {})
            params = {**BASE_PARAMS, **config}
            params["scale_pos_weight"] = compute_scale_pos_weight(y_label)

            model = lgb.LGBMClassifier(**params)
            model.fit(X_aug, y_label)

            self.models[label] = model

            # Predecir probabilidades en train para el siguiente modelo de la cadena
            chain_probs[:, i] = model.predict_proba(X_aug)[:, 1]

            n_pos = int(y_label.sum())
            pct = n_pos / len(y_label) * 100
            print(f"  {label}: {n_pos} positivos ({pct:.2f}%), scale_pos_weight={params['scale_pos_weight']:.1f}")

        self._fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Predecir probabilidades para las 6 etiquetas en orden de LABEL_COLS."""
        if not self._fitted:
            raise RuntimeError("Modelo no entrenado.")

        n = X.shape[0]
        n_chain = len(self.chain_order)
        chain_probs = np.zeros((n, n_chain))
        all_probs = np.zeros((n, len(LABEL_COLS)))

        for i, label in enumerate(self.chain_order):
            if i > 0:
                if issparse(X):
                    X_aug = sparse_hstack([X, csr_matrix(chain_probs[:, :i])])
                else:
                    X_aug = np.hstack([X, chain_probs[:, :i]])
            else:
                X_aug = X

            probs = self.models[label].predict_proba(X_aug)[:, 1]
            chain_probs[:, i] = probs
            all_probs[:, LABEL_COLS.index(label)] = probs

        return all_probs

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Predecir con umbral fijo."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def predict_f2_optimal(self, X, y: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Predecir con umbral optimo por F2 para cada etiqueta.
        Retorna predicciones y diccionario de umbrales optimos.
        """
        probs = self.predict_proba(X)
        optimal_thresholds = {}
        preds = np.zeros_like(probs, dtype=int)

        for j, label in enumerate(LABEL_COLS):
            best_f2 = 0
            best_t = 0.5
            for t in np.arange(0.1, 0.9, 0.05):
                p = (probs[:, j] >= t).astype(int)
                if p.sum() > 0 and p.sum() < len(p):
                    f2 = f1_score(y[:, j], p, beta=2, zero_division=0)
                    if f2 > best_f2:
                        best_f2 = f2
                        best_t = t
            optimal_thresholds[label] = best_t
            preds[:, j] = (probs[:, j] >= best_t).astype(int)

        return preds, optimal_thresholds

    def feature_importance(self) -> pd.DataFrame:
        """Importancia de features por etiqueta (split importance de LightGBM)."""
        rows = []
        for label, model in self.models.items():
            imp = model.feature_importances_
            all_names = self.feature_names + self.chain_features[: self.chain_order.index(label)]
            if len(imp) != len(all_names):
                # Truncar o rellenar si hay mismatch
                min_len = min(len(imp), len(all_names))
                names = all_names[:min_len]
                imp = imp[:min_len]
            else:
                names = all_names

            for name, importance in zip(names, imp):
                rows.append({
                    "label": label,
                    "feature": name,
                    "importance": importance,
                })

        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        """Guardar modelo y metadata."""
        path.mkdir(parents=True, exist_ok=True)

        # Guardar cada modelo individual
        for label, model in self.models.items():
            joblib.dump(model, path / f"lgbm_{label}.joblib")

        # Metadata
        meta = {
            "chain_order": self.chain_order,
            "feature_names": self.feature_names,
            "chain_features": self.chain_features,
            "label_cols": LABEL_COLS,
        }
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "ClassifierChainLGBM":
        """Cargar modelo guardado."""
        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(chain_order=meta["chain_order"])
        instance.feature_names = meta["feature_names"]
        instance.chain_features = meta["chain_features"]

        for label in meta["chain_order"]:
            instance.models[label] = joblib.load(path / f"lgbm_{label}.joblib")

        instance._fitted = True
        return instance
