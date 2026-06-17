"""
Entrenamiento del modelo multi-etiqueta para Jigsaw Toxic Comments.

Flujo:
  1. Carga datos (local o GCS)
  2. Feature engineering (TF-IDF + VADER + EMPATH + texto simple)
  3. Split train/test estratificado por any_toxic
  4. Entrena classifier chain LightGBM
  5. Evalua con bootstrap CI, F2-optimal threshold, ECE
  6. Compara contra baseline LogisticRegression
  7. Guarda modelo, metricas y graficas

Uso:
  python src/trainer/train.py                          # local
  python src/trainer/train.py --gcs-bucket mlops-...    # GCS

Herramientas de IA utilizadas: Claude (generacion de codigo y estructura).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer.features import FeaturePipeline, LABEL_COLS
from src.trainer.model import ClassifierChainLGBM, CHAIN_ORDER, BASE_PARAMS
from src.trainer.evaluation import (
    evaluate_multilabel, format_results_table, compute_ece,
    find_f2_optimal_threshold, fbeta_score,
)

RANDOM_STATE = 42
OUTPUT_DIR = PROJECT_ROOT / "reports" / "training"
IMG_DIR = OUTPUT_DIR / "imgs"


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenar modelo Jigsaw Toxic Comments")
    parser.add_argument("--gcs-bucket", type=str, default=None,
                        help="Bucket GCS para leer datos (ej: mlops-toxic-comments-ml)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directorio de salida para modelo y metricas")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Submuestreo para prueba rapida (None = dataset completo)")
    parser.add_argument("--tfidf-max-features", type=int, default=5000)
    parser.add_argument("--tfidf-ngram-range", type=int, nargs=2, default=[1, 2])
    parser.add_argument("--use-distilbert", action="store_true",
                        help="Incluir embeddings DistilBERT (requiere torch)")
    return parser.parse_args()


def load_data(gcs_bucket: str | None = None, sample_size: int | None = None) -> pd.DataFrame:
    """Carga train.csv desde local o GCS."""
    if gcs_bucket:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob("data/train.csv")
        local_path = PROJECT_ROOT / "data" / "train_from_gcs.csv"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            blob.download_to_filename(str(local_path))
            print(f"Descargado de GCS: {local_path}")
        df = pd.read_csv(local_path)
    else:
        data_path = PROJECT_ROOT / "raw" / "juegos" / "train.csv"
        df = pd.read_csv(data_path)

    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    if sample_size and sample_size < len(df):
        df["any_toxic"] = (df[LABEL_COLS].sum(axis=1) > 0).astype(int)
        df, _ = train_test_split(
            df, test_size=1 - sample_size / len(df),
            stratify=df["any_toxic"], random_state=RANDOM_STATE
        )
        print(f"Submuestreo a {sample_size} filas (estratificado por any_toxic)")
        df = df.reset_index(drop=True)

    return df


def train_baseline_lr(X_train, y_train, X_test, y_test, max_iter=300) -> dict:
    """Baseline: LogisticRegression con class_weight balanced sobre TODOS los features.

    Se usa solver='liblinear' que es mas rapido para datasets grandes con
    muchas features. El penalty L1 (lasso) ademas realiza seleccion de
    features implicita, lo que lo hace mas adecuado para espacios
    sparse de alta dimensionalidad como TF-IDF.
    """
    print("\n=== Baseline: LogisticRegression (TF-IDF + densos, liblinear) ===")
    results = {}

    for j, label in enumerate(LABEL_COLS):
        t0 = time.time()
        lr = LogisticRegression(
            class_weight="balanced",
            max_iter=max_iter,
            random_state=RANDOM_STATE,
            solver="liblinear",
            penalty="l1",
            C=0.1,
        )
        lr.fit(X_train, y_train[:, j])
        prob = lr.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test[:, j], prob)
        elapsed = time.time() - t0
        results[label] = {"auc": auc}
        print(f"  {label}: AUC={auc:.4f} ({elapsed:.0f}s)")

    return results


def main():
    args = parse_args()
    t_start = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    img_dir = out_dir / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos
    print("=" * 64)
    print("ENTRENAMIENTO: JIGSAW TOXIC COMMENTS")
    print("=" * 64)
    df = load_data(gcs_bucket=args.gcs_bucket, sample_size=args.sample_size)

    # any_toxic para estratificacion
    df["any_toxic"] = (df[LABEL_COLS].sum(axis=1) > 0).astype(int)

    # 2. Feature engineering
    print("\n=== Feature Engineering ===")
    vader_cache = PROJECT_ROOT / "data" / "sentiment_scores.csv"
    empath_csv = PROJECT_ROOT / "data" / "empath_scores.csv"
    empath_parquet = PROJECT_ROOT / "data" / "empath_scores.parquet"
    distilbert_cache = PROJECT_ROOT / "data" / "distilbert_embeddings.npy"

    fp = FeaturePipeline(
        max_tfidf_features=args.tfidf_max_features,
        tfidf_ngram_range=tuple(args.tfidf_ngram_range),
        use_vader=True,
        use_empath=True,
        use_distilbert=args.use_distilbert,
        vader_cache=vader_cache,
        empath_cache_csv=empath_csv,
        empath_cache_parquet=empath_parquet,
        distilbert_cache=distilbert_cache if args.use_distilbert else None,
        empath_top_k=15,
    )

    # Split train/test
    print("\n=== Train/Test Split (80/20, estratificado por any_toxic) ===")
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["any_toxic"], random_state=RANDOM_STATE
    )
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    for label in LABEL_COLS:
        train_pos = train_df[label].sum()
        test_pos = test_df[label].sum()
        print(f"  {label}: train {train_pos} ({train_pos/len(train_df)*100:.2f}%), "
              f"test {test_pos} ({test_pos/len(test_df)*100:.2f}%)")

    # Fit pipeline en train, transform ambos
    t_feat = time.time()
    train_data = fp.fit_transform(train_df, include_labels=True)
    test_data = fp.transform(test_df, include_labels=True)
    feat_time = time.time() - t_feat
    print(f"\nFeatures: {train_data['X'].shape[1]} total "
          f"({len(train_data['feature_names_tfidf'])} TF-IDF + "
          f"{len(train_data['feature_names_dense'])} densos)")
    print(f"Tiempo feature engineering: {feat_time:.1f}s")

    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]

    all_feature_names = train_data["feature_names_tfidf"] + train_data["feature_names_dense"]

    # 3. Baseline LogisticRegression (TF-IDF + densos)
    t_bl = time.time()
    baseline_results = train_baseline_lr(X_train, y_train, X_test, y_test)
    bl_time = time.time() - t_bl
    print(f"Tiempo baseline LR: {bl_time:.1f}s")

    # 4. Classifier Chain LightGBM
    print("\n=== Classifier Chain LightGBM ===")
    print(f"Orden de la cadena: {CHAIN_ORDER}")
    t_lgb = time.time()
    chain_model = ClassifierChainLGBM(chain_order=CHAIN_ORDER)
    chain_model.fit(X_train, y_train, feature_names=all_feature_names)
    lgb_time = time.time() - t_lgb
    print(f"Tiempo LightGBM: {lgb_time:.1f}s")

    # 5. Evaluacion
    print("\n=== Evaluacion en Test ===")
    y_prob = chain_model.predict_proba(X_test)

    eval_results = evaluate_multilabel(y_test, y_prob, n_bootstrap=200)
    results_table = format_results_table(eval_results)
    print(results_table.to_string(index=False))

    # 6. Comparacion LightGBM vs baseline
    print("\n=== Comparacion LightGBM vs LogisticRegression ===")
    comparison_rows = []
    for label in LABEL_COLS:
        lgbm_auc = eval_results["per_label"][label]["auc_roc"]["point"]
        lr_auc = baseline_results[label]["auc"]
        delta = lgbm_auc - lr_auc
        lgbm_ci = eval_results["per_label"][label]["auc_roc"]
        sig = "Si" if lgbm_ci["ci_lower"] > lr_auc else "No"
        comparison_rows.append({
            "etiqueta": label,
            "LR_AUC": round(lr_auc, 4),
            "LGBM_AUC": lgbm_auc,
            "LGBM_IC95": f"[{lgbm_ci['ci_lower']}, {lgbm_ci['ci_upper']}]",
            "delta_AUC": f"{delta:+.4f}",
            "LGBM_supera_LR": sig,
        })

    comp_df = pd.DataFrame(comparison_rows)
    print(comp_df.to_string(index=False))

    # 7. Feature importance
    print("\n=== Feature Importance (top 10 por etiqueta) ===")
    imp_df = chain_model.feature_importance()

    for label in CHAIN_ORDER:
        top = imp_df[imp_df["label"] == label].nlargest(10, "importance")
        print(f"\n  {label}:")
        for _, row in top.iterrows():
            print(f"    {row['feature']}: {row['importance']}")

    # 8. Guardar modelo y resultados
    model_dir = out_dir / "model"
    chain_model.save(model_dir)
    print(f"\nModelo guardado en: {model_dir}")

    # Guardar feature pipeline
    joblib.dump(fp, model_dir / "feature_pipeline.joblib")

    # Guardar metricas
    metrics_out = {
        "eval_results": {
            label: {
                k: v if not isinstance(v, dict) else v
                for k, v in metrics.items()
            }
            for label, metrics in eval_results["per_label"].items()
        },
        "macro": eval_results["macro"],
        "thresholds": eval_results["thresholds"],
        "baseline_lr": baseline_results,
        "comparison": comparison_rows,
        "feature_engineering": {
            "n_tfidf_features": len(train_data["feature_names_tfidf"]),
            "n_dense_features": len(train_data["feature_names_dense"]),
            "dense_feature_names": train_data["feature_names_dense"],
            "tfidf_ngram_range": list(fp.tfidf_ngram_range),
            "tfidf_max_features": fp.max_tfidf_features,
        },
        "timing": {
            "feature_engineering_s": round(feat_time, 1),
            "baseline_lr_s": round(bl_time, 1),
            "lightgbm_s": round(lgb_time, 1),
            "total_s": round(time.time() - t_start, 1),
        },
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False, default=str)

    # 9. Graficas
    _plot_auc_comparison(comp_df, img_dir)
    _plot_feature_importance(imp_df, img_dir)
    _plot_calibration(y_test, y_prob, img_dir)
    _plot_roc_curves(y_test, y_prob, img_dir)
    _plot_threshold_analysis(y_test, y_prob, img_dir)

    total_time = time.time() - t_start
    print(f"\n{'=' * 64}")
    print(f"Entrenamiento completado en {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Modelo: {model_dir}")
    print(f"Metricas: {out_dir / 'metrics.json'}")
    print(f"Graficas: {img_dir}")
    print(f"{'=' * 64}")


# ============================================================
# GRAFICAS
# ============================================================
def _plot_auc_comparison(comp_df, img_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(comp_df))
    width = 0.35
    ax.bar(x - width/2, comp_df["LR_AUC"], width, label="LogisticRegression", color="#3498db")
    ax.bar(x + width/2, comp_df["LGBM_AUC"], width, label="LightGBM (chain)", color="#9b59b6")
    # IC95 para LightGBM
    for i, (_, row) in enumerate(comp_df.iterrows()):
        ci = row["LGBM_IC95"].strip("[]").split(", ")
        ci_low, ci_high = float(ci[0]), float(ci[1])
        ax.plot([x[i]+width/2, x[i]+width/2], [ci_low, ci_high], color="black", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["etiqueta"], rotation=45, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC: LogisticRegression vs LightGBM Classifier Chain")
    ax.legend()
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{img_dir}/34_auc_lr_vs_lgbm.png", dpi=150)
    plt.close(fig)


def _plot_feature_importance(imp_df, img_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, label in enumerate(CHAIN_ORDER):
        sub = imp_df[imp_df["label"] == label].nlargest(10, "importance")
        sub = sub.sort_values("importance", ascending=True)
        cats = [c.replace("emp_", "") if c.startswith("emp_") else c for c in sub["feature"]]
        colors = ["#9b59b6" if c.startswith("chain_") or c.startswith("emp_") or c.startswith("sent_")
                  else "#3498db" for c in sub["feature"]]
        axes[i].barh(cats, sub["importance"], color=colors)
        axes[i].set_title(f"Top 10 features: {label}")
        axes[i].set_xlabel("Importancia (split)")

    # Leyenda manual
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", label="TF-IDF / texto"),
        Patch(facecolor="#9b59b6", label="VADER / EMPATH / chain"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10)
    fig.suptitle("Feature importance por etiqueta (LightGBM classifier chain)")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(f"{img_dir}/35_feature_importance.png", dpi=150)
    plt.close(fig)


def _plot_calibration(y_test, y_prob, img_dir):
    from sklearn.calibration import calibration_curve

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for j, label in enumerate(LABEL_COLS):
        ax = axes[j]
        yt = y_test[:, j]
        yp = y_prob[:, j]

        frac_pos, mean_pred = calibration_curve(yt, yp, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "s-", color="#9b59b6", label="LightGBM")
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Perfect calibration")

        ece = compute_ece(yt, yp)
        ax.set_title(f"{label} (ECE={ece:.4f})")
        ax.set_xlabel("Probabilidad predicha")
        ax.set_ylabel("Fraccion de positivos")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Calibracion de probabilidades por etiqueta")
    fig.tight_layout()
    fig.savefig(f"{img_dir}/36_calibration.png", dpi=150)
    plt.close(fig)


def _plot_roc_curves(y_test, y_prob, img_dir):
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for j, label in enumerate(LABEL_COLS):
        ax = axes[j]
        yt = y_test[:, j]
        yp = y_prob[:, j]

        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)
        ax.plot(fpr, tpr, color="#9b59b6", linewidth=2, label=f"AUC={auc:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
        ax.set_title(label)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=9)

    fig.suptitle("Curvas ROC por etiqueta (LightGBM classifier chain)")
    fig.tight_layout()
    fig.savefig(f"{img_dir}/37_roc_curves.png", dpi=150)
    plt.close(fig)


def _plot_threshold_analysis(y_test, y_prob, img_dir):
    """F1, F2 y Precision-Recall por umbral para cada etiqueta."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for j, label in enumerate(LABEL_COLS):
        ax = axes[j]
        yt = y_test[:, j]
        yp = y_prob[:, j]

        thresholds = np.arange(0.05, 0.95, 0.01)
        f1s, f2s, precs, recs = [], [], [], []

        for t in thresholds:
            preds = (yp >= t).astype(int)
            if preds.sum() > 0 and preds.sum() < len(preds):
                f1s.append(f1_score(yt, preds, zero_division=0))
                f2s.append(fbeta_score(yt, preds, beta=2))
                precs.append(precision_score(yt, preds, zero_division=0))
                recs.append(recall_score(yt, preds, zero_division=0))
            else:
                f1s.append(0)
                f2s.append(0)
                precs.append(0)
                recs.append(0)

        ax.plot(thresholds, f1s, color="#3498db", label="F1")
        ax.plot(thresholds, f2s, color="#e74c3c", label="F2")
        ax.plot(thresholds, precs, color="#2ecc71", linestyle="--", label="Precision", alpha=0.7)
        ax.plot(thresholds, recs, color="#f39c12", linestyle="--", label="Recall", alpha=0.7)

        opt_t, _ = find_f2_optimal_threshold(yt, yp)
        ax.axvline(opt_t, color="red", linestyle=":", alpha=0.5, label=f"F2 opt t={opt_t:.2f}")

        ax.set_title(label)
        ax.set_xlabel("Umbral")
        ax.legend(fontsize=7)

    fig.suptitle("Metricas por umbral de decision (F2-optimo marcado)")
    fig.tight_layout()
    fig.savefig(f"{img_dir}/38_threshold_analysis.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
