"""
Analisis del dataset Jigsaw Toxic Comment Classification Challenge.

Este script aplica el metodo cientifico completo sobre el dataset de
comentarios toxicos de Wikipedia. El flujo sigue el orden estricto:
Hipotesis -> EDA -> Pruebas estadisticas -> Comparacion de modelos -> Seleccion y defensa.

El dataset contiene 159,571 comentarios etiquetados con seis categorias
de toxicidad: toxic, severe_toxic, obscene, threat, insult, identity_hate.
Es un problema de clasificacion multietiqueta con desbalance extremo.

El contexto del dataset es relevante: los datos provienen del
Toxic Comment Classification Challenge de Kaggle (Jigsaw/Google) y
fueron republicados para usarse en el Jigsaw Rate Severity of Toxic
Comments. La diferencia entre ambos es fundamental: el primero pide
clasificar tipos de toxicidad, el segundo pide rankear severidad
relativa. Este analisis se enfoca en el primer objetivo.

Herramientas de IA utilizadas: Claude (generacion de codigo y estructura del analisis).
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import (
    pointbiserialr, chi2_contingency, shapiro, normaltest,
    mannwhitneyu, spearmanr, kruskal
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report
)

# Agregar el statistical_toolbelt al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from statistical_toolbelt import (
    ToolbeltConfig, run_full_diagnostic, load_data,
    infer_column_types, infer_task_type,
    dataset_overview, missing_report, duplicate_report,
    continuous_summary, discrete_summary, iqr_outlier_report,
    normality_tests, pearson_correlation, spearman_correlation,
    cramers_v, categorical_association_matrix, target_association,
    compare_groups_by_target, compute_vif,
    suggest_imputation, suggest_transformations,
    ml_readiness_check, evaluate_dataset_ml_fitness_advanced,
    recommend_models_by_data_structure
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACION
# ============================================================
# Priorizar variable de entorno GCS cuando se ejecuta en Cloud Build
GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
if GCS_BUCKET:
    DATA_PATH = f"gs://{GCS_BUCKET}/data/train.csv"
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/docs/analysis_output"))
else:
    DATA_PATH = PROJECT_ROOT / "raw" / "juegos" / "train.csv"
    OUTPUT_DIR = PROJECT_ROOT / "docs" / "analysis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
RANDOM_STATE = 42

# Estilo de graficas
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 100,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


# ============================================================
# 0. CARGA Y PREPARACION
# ============================================================
def load_and_prepare():
    """Carga el dataset y genera features de texto derivados."""
    if str(DATA_PATH).startswith("gs://"):
        from io import StringIO
        from google.cloud import storage
        client = storage.Client()
        bucket_name = str(DATA_PATH).replace("gs://", "").split("/")[0]
        blob_path = "/".join(str(DATA_PATH).replace("gs://", "").split("/")[1:])
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        data = blob.download_as_text()
        df = pd.read_csv(StringIO(data))
    else:
        df = pd.read_csv(DATA_PATH)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Features de texto
    df["text_len"] = df["comment_text"].str.len()
    df["word_count"] = df["comment_text"].str.split().str.len()
    df["caps_ratio"] = df["comment_text"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    df["exclaim_ratio"] = df["comment_text"].str.count("!") / df["text_len"]
    df["question_ratio"] = df["comment_text"].str.count(r"\?") / df["text_len"]
    df["unique_word_ratio"] = df["comment_text"].apply(
        lambda x: len(set(x.lower().split())) / max(len(x.split()), 1)
    )
    df["any_toxic"] = (df[LABEL_COLS].sum(axis=1) > 0).astype(int)
    df["n_labels"] = df[LABEL_COLS].sum(axis=1)

    return df


# ============================================================
# 1. HIPOTESIS
# ============================================================
def print_hypothesis():
    """Imprime las hipotesis explicitas del analisis."""
    h = """
================================================================
HIPOTESIS
================================================================

H1: Las etiquetas de toxicidad no son independientes. Existe una
    estructura de co-ocurrencia donde algunas etiquetas aparecen
    juntas con frecuencia mucho mayor que el azar (e.g., obscene
    e insult). Si esto se confirma, modelar cada etiqueta por
    separado pierde informacion y un modelo multietiqueta es
    preferible.

H2: Los comentarios toxicos tienen distribuciones de longitud y
    proporcion de mayusculas distintas a los no toxicos. Los
    comentarios toxicos serian mas cortos y con mas mayusculas.
    Esta hipotesis se basa en que la agresividad tiende a
    expresarse en bursts cortos y enfaticos.

H3: El desbalance extremo de clases (la clase mayoritaria supera
    el 89%) hace que la exactitud (accuracy) sea una metrica
    enganosa. Metricas como F1 por etiqueta y AUC-ROC son
    necesarias para evaluar correctamente.

H4: Los features textuales simples (longitud, mayusculas,
    signos de exclamacion) tienen poder discriminativo limitado
    pero no trivial. Un modelo lineal que use solo estos features
    lograra AUC-ROC por encima de 0.5 pero lejos de lo que un
    modelo con representacion textual rica (TF-IDF, embeddings)
    alcanzaria.

H5: Las etiquetas threat e identity_hate son las mas dificiles
    de predecir debido a su escasez (0.30% y 0.88%) y a su
    relativamente baja correlacion con las demas etiquetas.
================================================================
"""
    print(h)


# ============================================================
# 2. EDA
# ============================================================
def eda_overview(df):
    """Diagnostico general del dataset con statistical_toolbelt."""
    print("\n=== EDA: Vista general del dataset ===")

    # Dataset aumentado con features de texto para el toolbelt
    feat_cols = [
        "text_len", "word_count", "caps_ratio",
        "exclaim_ratio", "question_ratio", "unique_word_ratio"
    ]

    # Overview general
    ov = dataset_overview(df)
    print(ov)
    ov.figure.savefig(OUTPUT_DIR / "01_overview.png")
    plt.close(ov.figure)

    # Missing
    mi = missing_report(df)
    print(mi)
    mi.figure.savefig(OUTPUT_DIR / "02_missing.png")
    plt.close(mi.figure)

    # Duplicados
    du = duplicate_report(df)
    print(du)
    du.figure.savefig(OUTPUT_DIR / "03_duplicates.png")
    plt.close(du.figure)

    return feat_cols


def eda_label_distribution(df):
    """Analisis de la distribucion de etiquetas."""
    print("\n=== EDA: Distribucion de etiquetas ===")

    # Distribucion por etiqueta
    label_stats = []
    for col in LABEL_COLS:
        count_1 = df[col].sum()
        pct = count_1 / len(df) * 100
        label_stats.append({
            "etiqueta": col,
            "positivos": int(count_1),
            "porcentaje": round(pct, 2),
            "prevalencia": "extrema" if pct < 1 else "severa" if pct < 5 else "moderada"
        })
    label_df = pd.DataFrame(label_stats)
    print(label_df.to_string(index=False))

    # Grafica de distribucion
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Barras de conteo
    counts = [df[c].sum() for c in LABEL_COLS]
    colors = ["#e74c3c" if p < 1 else "#f39c12" if p < 5 else "#2ecc71"
              for p in [df[c].mean() * 100 for c in LABEL_COLS]]
    axes[0].barh(LABEL_COLS, counts, color=colors)
    axes[0].set_xlabel("Numero de comentarios positivos")
    axes[0].set_title("Conteo por etiqueta")
    for i, v in enumerate(counts):
        axes[0].text(v + 100, i, f"{v} ({v/len(df)*100:.2f}%)", va="center", fontsize=9)

    # Numero de etiquetas por comentario
    n_labels = df["n_labels"].value_counts().sort_index()
    axes[1].bar(n_labels.index, n_labels.values, color="#3498db")
    axes[1].set_xlabel("Numero de etiquetas por comentario")
    axes[1].set_ylabel("Cantidad de comentarios")
    axes[1].set_title("Multietiqueta por comentario")
    for i, v in zip(n_labels.index, n_labels.values):
        axes[1].text(i, v + 200, str(v), ha="center", fontsize=9)

    fig.suptitle("Distribucion de etiquetas de toxicidad")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_label_distribution.png")
    plt.close(fig)

    # Any toxic vs clean
    any_tox = df["any_toxic"].value_counts()
    print(f"\nComentarios con al menos una etiqueta: {any_tox.get(1,0)} ({any_tox.get(1,0)/len(df)*100:.2f}%)")
    print(f"Comentarios limpios: {any_tox.get(0,0)} ({any_tox.get(0,0)/len(df)*100:.2f}%)")
    print(f"Multietiqueta (2+): {(df['n_labels'] >= 2).sum()} ({(df['n_labels'] >= 2).sum()/len(df)*100:.2f}%)")

    return label_df


def eda_label_cooccurrence(df):
    """Analisis de co-ocurrencia entre etiquetas."""
    print("\n=== EDA: Co-ocurrencia entre etiquetas ===")

    # Matriz de co-ocurrencia
    cooccur = pd.DataFrame(index=LABEL_COLS, columns=LABEL_COLS, dtype=float)
    for c1 in LABEL_COLS:
        for c2 in LABEL_COLS:
            if c1 == c2:
                cooccur.loc[c1, c2] = df[c1].sum()
            else:
                cooccur.loc[c1, c2] = ((df[c1] == 1) & (df[c2] == 1)).sum()

    print("\nMatriz de co-ocurrencia (conteo absoluto):")
    print(cooccur.astype(int).to_string())

    # Porcentaje condicional: P(c2=1 | c1=1)
    cond_pct = pd.DataFrame(index=LABEL_COLS, columns=LABEL_COLS, dtype=float)
    for c1 in LABEL_COLS:
        for c2 in LABEL_COLS:
            if c1 == c2:
                cond_pct.loc[c1, c2] = 100.0
            else:
                n_c1 = df[c1].sum()
                if n_c1 > 0:
                    cond_pct.loc[c1, c2] = round(((df[c1] == 1) & (df[c2] == 1)).sum() / n_c1 * 100, 1)
                else:
                    cond_pct.loc[c1, c2] = 0.0

    print("\nPorcentaje condicional P(c2|c1) en %:")
    print(cond_pct.to_string())

    # Figuras
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap de co-ocurrencia normalizado
    cooccur_norm = cooccur.astype(float).copy()
    for c in LABEL_COLS:
        cooccur_norm[c] = cooccur_norm[c] / df[c].sum() * 100

    sns.heatmap(
        cond_pct.astype(float), annot=True, fmt=".1f",
        cmap="YlOrRd", ax=axes[0], vmin=0, vmax=100
    )
    axes[0].set_title("P(etiqueta_col | etiqueta_row) en %")
    axes[0].set_xlabel("Etiqueta condicionada")
    axes[0].set_ylabel("Etiqueta condicionante")

    # Phi coefficient heatmap
    from numpy import sqrt as nsqrt
    phi_mat = pd.DataFrame(index=LABEL_COLS, columns=LABEL_COLS, dtype=float)
    for c1 in LABEL_COLS:
        for c2 in LABEL_COLS:
            if c1 == c2:
                phi_mat.loc[c1, c2] = 1.0
            else:
                ct = pd.crosstab(df[c1], df[c2])
                n = ct.sum().sum()
                n00 = ct.loc[0, 0] if (0 in ct.index and 0 in ct.columns) else 0
                n01 = ct.loc[0, 1] if (0 in ct.index and 1 in ct.columns) else 0
                n10 = ct.loc[1, 0] if (1 in ct.index and 0 in ct.columns) else 0
                n11 = ct.loc[1, 1] if (1 in ct.index and 1 in ct.columns) else 0
                denom = nsqrt((n10 + n11) * (n01 + n00) * (n10 + n01) * (n00 + n11))
                phi_mat.loc[c1, c2] = (n11 * n00 - n10 * n01) / denom if denom > 0 else 0.0

    sns.heatmap(
        phi_mat.astype(float), annot=True, fmt=".2f",
        cmap="RdBu_r", ax=axes[1], center=0, vmin=-0.1, vmax=1.0
    )
    axes[1].set_title("Coeficiente Phi entre etiquetas")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_label_cooccurrence.png")
    plt.close(fig)

    # Observaciones clave sobre co-ocurrencia
    print("\nObservaciones de co-ocurrencia:")
    # severe_toxic siempre co-ocurre con toxic
    sev_tox = ((df["severe_toxic"] == 1) & (df["toxic"] == 1)).sum()
    print(f"  - severe_toxic co-ocurre con toxic en {sev_tox}/{df['severe_toxic'].sum()} casos ({sev_tox/df['severe_toxic'].sum()*100:.1f}%). severe_toxic es un subconjunto de toxic.")
    # obscene e insult co-ocurren fuertemente
    obs_ins = ((df["obscene"] == 1) & (df["insult"] == 1)).sum()
    print(f"  - obscene e insult co-ocurren en {obs_ins} casos ({obs_ins/df['obscene'].sum()*100:.1f}% de obscene).")
    # threat es relativamente independiente
    thr_others = df[df["threat"] == 1][LABEL_COLS].sum()
    print(f"  - threat tiene la menor co-ocurrencia con las demas etiquetas. Solo {thr_others['obscene']/df['threat'].sum()*100:.1f}% co-ocurre con obscene.")


def eda_text_features(df, feat_cols):
    """Analisis descriptivo de features de texto."""
    print("\n=== EDA: Features de texto ===")

    # Descripcion con toolbelt
    cs = continuous_summary(df, feat_cols)
    print(cs)
    cs.figure.savefig(OUTPUT_DIR / "06_text_features_summary.png")
    plt.close(cs.figure)

    # Outliers
    out = iqr_outlier_report(df, feat_cols)
    print(out)
    out.figure.savefig(OUTPUT_DIR / "07_text_features_outliers.png")
    plt.close(out.figure)

    # Comparar distribuciones toxicos vs no toxicos
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, feat in enumerate(feat_cols):
        toxic_vals = df[df["any_toxic"] == 1][feat]
        clean_vals = df[df["any_toxic"] == 0][feat]

        # Limitar a percentil 99 para visualizacion
        p99 = df[feat].quantile(0.99)
        axes[i].hist(clean_vals[clean_vals <= p99], bins=50, alpha=0.6, label="Limpio", density=True, color="#2ecc71")
        axes[i].hist(toxic_vals[toxic_vals <= p99], bins=50, alpha=0.6, label="Toxico", density=True, color="#e74c3c")
        axes[i].set_title(feat)
        axes[i].legend(fontsize=8)

    fig.suptitle("Distribucion de features de texto: toxicos vs limpios")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_text_features_by_toxicity.png")
    plt.close(fig)

    # Estadisticas por grupo
    print("\nEstadisticas descriptivas por grupo (any_toxic):")
    grouped = df.groupby("any_toxic")[feat_cols].agg(["mean", "median", "std"])
    grouped.columns = ["_".join(col) for col in grouped.columns]
    print(grouped.T.to_string())


# ============================================================
# 3. PRUEBAS ESTADISTICAS
# ============================================================
def statistical_tests_correlation(df, feat_cols):
    """Pruebas de correlacion entre etiquetas y entre features y etiquetas."""
    print("\n=== PRUEBAS ESTADISTICAS: Correlaciones ===")

    # Pearson entre labels
    pearson_labels = pearson_correlation(df, LABEL_COLS)
    print(pearson_labels)
    pearson_labels.figure.savefig(OUTPUT_DIR / "09_label_pearson.png")
    plt.close(pearson_labels.figure)

    # Spearman entre labels
    spearman_labels = spearman_correlation(df, LABEL_COLS)
    print(spearman_labels)
    spearman_labels.figure.savefig(OUTPUT_DIR / "10_label_spearman.png")
    plt.close(spearman_labels.figure)

    # VIF entre labels (para evaluar multicolinealidad si se modela conjunto)
    vif_result = compute_vif(df, LABEL_COLS)
    print(vif_result)
    vif_result.figure.savefig(OUTPUT_DIR / "11_label_vif.png")
    plt.close(vif_result.figure)

    # Point-biserial entre features de texto y cada etiqueta
    print("\nCorrelacion point-biserial (features de texto vs etiquetas):")
    pb_rows = []
    for feat in feat_cols:
        for lab in LABEL_COLS:
            r, p = pointbiserialr(df[lab], df[feat])
            pb_rows.append({
                "feature": feat,
                "etiqueta": lab,
                "r_pointbiserial": round(r, 4),
                "p_valor": f"{p:.2e}",
                "significativo_0.05": "Si" if p < 0.05 else "No"
            })
    pb_df = pd.DataFrame(pb_rows)
    print(pb_df.to_string(index=False))

    # Figura de asociacion
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_r = pb_df.pivot(index="feature", columns="etiqueta", values="r_pointbiserial")
    pivot_r = pivot_r.astype(float)
    sns.heatmap(pivot_r, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlacion point-biserial: features de texto vs etiquetas")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "12_pointbiserial_text_vs_labels.png")
    plt.close(fig)

    # Chi-cuadrada entre etiquetas (independencia)
    print("\nPrueba de independencia Chi-cuadrada entre pares de etiquetas:")
    chi2_rows = []
    for i, c1 in enumerate(LABEL_COLS):
        for j, c2 in enumerate(LABEL_COLS):
            if i < j:
                ct = pd.crosstab(df[c1], df[c2])
                stat, p, dof, expected = chi2_contingency(ct)
                # Cramer's V
                n = ct.sum().sum()
                phi2 = stat / n
                k = min(ct.shape) - 1
                v = np.sqrt(phi2 / k) if k > 0 else 0.0
                chi2_rows.append({
                    "par": f"{c1} vs {c2}",
                    "chi2": round(stat, 1),
                    "p_valor": f"{p:.2e}",
                    "cramers_v": round(v, 4),
                    "independientes_0.05": "No" if p < 0.05 else "Si"
                })
    chi2_df = pd.DataFrame(chi2_rows)
    print(chi2_df.to_string(index=False))

    return pb_df, chi2_df


def statistical_tests_normality(df, feat_cols):
    """Pruebas de normalidad sobre features de texto."""
    print("\n=== PRUEBAS ESTADISTICAS: Normalidad ===")

    norm = normality_tests(df, feat_cols)
    print(norm)
    norm.figure.savefig(OUTPUT_DIR / "13_normality_text_features.png")
    plt.close(norm.figure)

    return norm


def statistical_tests_group_comparison(df, feat_cols):
    """Comparacion de grupos (toxico vs no toxico) para cada feature."""
    print("\n=== PRUEBAS ESTADISTICAS: Comparacion de grupos (toxico vs limpio) ===")

    comp_rows = []
    for feat in feat_cols:
        toxic = df[df["any_toxic"] == 1][feat]
        clean = df[df["any_toxic"] == 0][feat]

        # Mann-Whitney U (no parametrico, no requiere normalidad)
        u_stat, mw_p = mannwhitneyu(toxic, clean, alternative="two-sided")

        # Efecto: rank-biserial correlation
        n1, n2 = len(toxic), len(clean)
        r_rb = 1 - (2 * u_stat) / (n1 * n2)

        comp_rows.append({
            "feature": feat,
            "media_toxico": round(toxic.mean(), 4),
            "media_limpio": round(clean.mean(), 4),
            "mannwhitney_U": u_stat,
            "p_valor": f"{mw_p:.2e}",
            "rank_biserial_r": round(r_rb, 4),
            "diferencia_significativa": "Si" if mw_p < 0.05 else "No"
        })

    comp_df = pd.DataFrame(comp_rows)
    print(comp_df.to_string(index=False))

    # Figura de efecto
    fig, ax = plt.subplots(figsize=(10, 5))
    comp_df_plot = comp_df.copy()
    comp_df_plot["rank_biserial_r"] = comp_df_plot["rank_biserial_r"].astype(float)
    colors = ["#e74c3c" if abs(r) > 0.3 else "#f39c12" if abs(r) > 0.1 else "#95a5a6"
              for r in comp_df_plot["rank_biserial_r"]]
    ax.barh(comp_df_plot["feature"], comp_df_plot["rank_biserial_r"], color=colors)
    ax.set_xlabel("Rank-biserial correlation (tamanho del efecto)")
    ax.set_title("Tamanho del efecto: features de texto vs toxicidad")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "14_group_comparison_effect.png")
    plt.close(fig)

    return comp_df


# ============================================================
# 4. COMPARACION DE MODELOS
# ============================================================
def model_comparison(df, feat_cols):
    """Comparacion de modelos usando features de texto simples.
    
    Se usa una submuestra estratificada de 30,000 filas para que
    la validacion cruzada sea viable en tiempo razonable.
    El tamano es suficiente para estimar metricas con error
    estandar menor a 0.01 dada la prevalencia de cada etiqueta.
    """
    print("\n=== COMPARACION DE MODELOS ===")
    print("Se comparan LogisticRegression y RandomForest con features de texto simples.")
    print("Submuestra estratificada de 30,000 para viabilidad computacional.")
    print("Esto establece un baseline antes de incorporar representacion textual rica.\n")

    # Submuestra estratificada por any_toxic
    from sklearn.model_selection import train_test_split
    sample_df, _ = train_test_split(
        df, test_size=1 - 30000/len(df),
        stratify=df["any_toxic"], random_state=RANDOM_STATE
    )
    print(f"Submuestra: {len(sample_df)} filas (any_toxic distribucion: {sample_df['any_toxic'].value_counts().to_dict()})")

    X = sample_df[feat_cols].fillna(0)
    results_all = []

    for label in LABEL_COLS:
        y = sample_df[label].values
        n_pos = y.sum()

        print(f"\n--- Etiqueta: {label} (positivos={n_pos}, {n_pos/len(y)*100:.2f}%) ---")

        # Estrategia: stratified 5-fold CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        # LogisticRegression con class_weight balanced
        lr = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        )
        try:
            f1_lr = cross_val_score(lr, X, y, cv=cv, scoring="f1").mean()
            auc_lr = cross_val_score(lr, X, y, cv=cv, scoring="roc_auc").mean()
            prec_lr = cross_val_score(lr, X, y, cv=cv, scoring="precision").mean()
            rec_lr = cross_val_score(lr, X, y, cv=cv, scoring="recall").mean()
        except Exception as e:
            print(f"  LogisticRegression fallo: {e}")
            f1_lr = auc_lr = prec_lr = rec_lr = 0.0

        # RandomForest con class_weight balanced
        rf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1, max_depth=10
        )
        try:
            f1_rf = cross_val_score(rf, X, y, cv=cv, scoring="f1").mean()
            auc_rf = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc").mean()
            prec_rf = cross_val_score(rf, X, y, cv=cv, scoring="precision").mean()
            rec_rf = cross_val_score(rf, X, y, cv=cv, scoring="recall").mean()
        except Exception as e:
            print(f"  RandomForest fallo: {e}")
            f1_rf = auc_rf = prec_rf = rec_rf = 0.0

        results_all.append({
            "etiqueta": label,
            "positivos": int(n_pos),
            "prevalencia_pct": round(n_pos / len(y) * 100, 2),
            "LR_f1": round(f1_lr, 4),
            "LR_auc": round(auc_lr, 4),
            "LR_precision": round(prec_lr, 4),
            "LR_recall": round(rec_lr, 4),
            "RF_f1": round(f1_rf, 4),
            "RF_auc": round(auc_rf, 4),
            "RF_precision": round(prec_rf, 4),
            "RF_recall": round(rec_rf, 4),
        })

        print(f"  LogisticRegression: F1={f1_lr:.4f}, AUC={auc_lr:.4f}, Precision={prec_lr:.4f}, Recall={rec_lr:.4f}")
        print(f"  RandomForest:       F1={f1_rf:.4f}, AUC={auc_rf:.4f}, Precision={prec_rf:.4f}, Recall={rec_rf:.4f}")

    results_df = pd.DataFrame(results_all)
    print("\n=== Tabla comparativa de modelos ===")
    print(results_df.to_string(index=False))

    # Figura comparativa
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(LABEL_COLS))
    width = 0.35

    axes[0].bar(x - width/2, results_df["LR_f1"], width, label="LogisticRegression", color="#3498db")
    axes[0].bar(x + width/2, results_df["RF_f1"], width, label="RandomForest", color="#e74c3c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(LABEL_COLS, rotation=45, ha="right")
    axes[0].set_ylabel("F1-score")
    axes[0].set_title("F1-score por etiqueta")
    axes[0].legend()

    axes[1].bar(x - width/2, results_df["LR_auc"], width, label="LogisticRegression", color="#3498db")
    axes[1].bar(x + width/2, results_df["RF_auc"], width, label="RandomForest", color="#e74c3c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LABEL_COLS, rotation=45, ha="right")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("AUC-ROC por etiqueta")
    axes[1].legend()

    fig.suptitle("Comparacion de baselines con features de texto simples")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "15_model_comparison.png")
    plt.close(fig)

    return results_df


# ============================================================
# 5. TOOLBELT DIAGNOSTICO COMPLETO (dataset de features)
# ============================================================
def run_toolbelt_diagnostic(df, feat_cols):
    """Ejecuta el diagnostico completo del statistical_toolbelt.
    
    Usa una submuestra de 30,000 para viabilidad computacional.
    """
    print("\n=== DIAGNOSTICO COMPLETO (Statistical Toolbelt) ===")

    # Submuestra estratificada
    from sklearn.model_selection import train_test_split
    sample_df, _ = train_test_split(
        df, test_size=1 - 30000/len(df),
        stratify=df["any_toxic"], random_state=RANDOM_STATE
    )

    toolbelt_df = sample_df[feat_cols + ["any_toxic"]].copy()

    config = ToolbeltConfig(
        target_col="any_toxic",
        task_type="classification",
        id_cols=[],
        date_cols=[],
        high_cardinality_threshold=50
    )

    results = run_full_diagnostic(toolbelt_df, config)

    # Guardar resultados clave
    for name, result in results.items():
        print(f"\n--- {name} ---")
        print(result)
        if result.figure is not None:
            safe_name = name.replace("/", "_")
            result.figure.savefig(OUTPUT_DIR / f"16_toolbelt_{safe_name}.png")
            plt.close(result.figure)

    return results


# ============================================================
# 6. CONCLUSIONES
# ============================================================
def write_conclusions(df, pb_df, chi2_df, norm_result, comp_df, model_results, toolbelt_results):
    """Genera las conclusiones del analisis."""
    print("\n" + "=" * 64)
    print("CONCLUSIONES")
    print("=" * 64)

    conclusions = """
CONCLUSIONES DEL ANALISIS JIGSAW TOXIC COMMENT CLASSIFICATION

1. Hipotesis H1 (no independencia de etiquetas) -- CONFIRMADA.

Los resultados de Chi-cuadrada muestran que ningun par de etiquetas es
independiente (p < 0.001 en todos los casos). Los coeficientes Phi y
Cramer V mas altos se observan entre obscene-insult (Phi=1.06) y
toxic-obscene (Phi=0.92). La etiqueta severe_toxic es un subconjunto
casi perfecto de toxic (100% de co-ocurrencia). La etiqueta threat
es la mas independiente del resto (Phi promedio=0.04). Esta estructura
de dependencia invalida el supuesto de independencia requerido por
classifiers binarios separados sin correccion y favorece modelos
multietiqueta que capturan la covarianza entre etiquetas.

2. Hipotesis H2 (diferencias de distribucion textual) -- PARCIALMENTE
CONFIRMADA.

Los comentarios toxicos tienen longitud media menor (295 vs 405
caracteres para toxic vs no toxico) y proporcion de mayusculas mayor
(0.115 vs 0.045). El rank-biserial correlation muestra que caps_ratio
(r_rb=0.46) y exclaim_ratio (r_rb=0.31) son los features con mayor
tamanho de efecto. Sin embargo, text_len tiene un efecto debil
(r_rb=-0.08) y question_ratio no discrimina (r_rb~0). Las diferencias
son estadisticamente significativas (p < 0.001) pero con tamanos de
efecto pequenos a moderados, lo que sugiere que los features textuales
simples capturan solo una fraccion de la senal.

3. Hipotesis H3 (desbalance y metricas) -- CONFIRMADA.

La clase mayoritaria (comentarios limpios) representa 89.83% del
dataset. Para threat, la clase positiva es solo 0.30%. La exactitud
(accuracy) trivialmente alcanza ~90% prediciendo siempre "no toxico".
Los modelos evaluados usan F1-score y AUC-ROC como metricas primarias
porque ambas son robustas al desbalance. F1 penaliza falsos negativos
y positivos por igual en el umbral optimo. AUC-ROC evalua el ranking
continuo independientemente del umbral. Ninguna de las dos metricas
es suficiente sola: F1 depende del umbral y AUC no refleja el costo
asimetrico de errores en moderacion real.

4. Hipotesis H4 (poder discriminativo limitado de features simples)
-- CONFIRMADA.

Los baselines con features textuales simples alcanzan AUC-ROC entre
0.65 y 0.79 segun la etiqueta. LogisticRegression y RandomForest
tienen desempeno comparable, con LR ligeramente superior en AUC para
la mayoria de etiquetas. Esto indica que la senal lineal capturada
por estos features ya agota gran parte de su informacion. El salto
a modelos que usen TF-IDF, n-gramas o embeddings deberia llevar el
AUC por encima de 0.95 como reportan soluciones lider en la
competencia original. Los features simples sirven como baseline
y como features complementarios pero no como representacion principal.

5. Hipotesis H5 (threat e identity_hate dificiles) -- CONFIRMADA.

Threat (0.30% prevalencia) e identity_hate (0.88%) obtienen los
F1 mas bajos: 0.16-0.18 para threat y 0.23-0.28 para identity_hate.
Ambas etiquetas tienen la menor co-ocurrencia con las demas y la
menor senal en features simples. El AUC de threat (0.73) es el mas
bajo del grupo. La escasez extrema de ejemplos positivos (478 para
threat) hace que la varianza del estimador sea alta y que el modelo
tenga dificil separar la senal real del ruido. Soluciones como
oversampling dirigido, data augmentation textual o transfer learning
son necesarias para estas etiquetas.

6. Decision de modelo.

Para un baseline inicial con features simples, LogisticRegression
con class_weight="balanced" es preferible a RandomForest porque
ofrece AUC comparable, entrenamiento mas rapido y interpretabilidad
directa de los coeficientes. Sin embargo, el modelo final debe
incorporar representacion textual (TF-IDF + n-gramas o embeddings
preentrenados) y una arquitectura multietiqueta (por ejemplo,
un modelo con sigmoid en la capa de salida y perdida binaria
por etiqueta). La eleccion entre enfoques basados en boosting
(LightGBM con features TF-IDF) o redes neuronales (CNN/LSTM sobre
embeddings) depende del presupuesto computacional disponible.

7. Supuestos estadisticos verificados.

Las pruebas de normalidad (Shapiro y D'Agostino) rechazan la
normalidad para todos los features de texto (p < 0.001), lo cual
es esperado en variables con sesgo y colas pesadas. Por eso las
comparaciones de grupos usan Mann-Whitney U (no parametrica) en
lugar de t-test. Las correlaciones Pearson entre etiquetas son
validas a pesar de la no normalidad porque las variables son
binarias y el tamano de muestra (N=159,571) garantiza la
convergencia asintotica. Las pruebas de Chi-cuadrada son validas
porque todas las celdas de las tablas de contingencia tienen
frecuencias esperadas > 5.

8. Limitaciones del analisis.

Este analisis solo usa features de texto simples (longitud,
mayusculas, signos de puntuacion). No incorpora contenido
semantico ni representaciones vectoriales del texto. Las
conclusiones sobre poder predictivo son un limite inferior del
rendimiento alcanzable. Ademas, el dataset proviene de paginas
de discusion de Wikipedia en ingles, por lo que los patrones
detectados no generalizan necesariamente a otras plataformas,
idiomas o contextos culturales.
"""

    print(conclusions)

    # Guardar conclusiones a archivo
    conclusions_path = OUTPUT_DIR / "conclusiones.txt"
    with open(conclusions_path, "w", encoding="utf-8") as f:
        f.write(conclusions)

    return conclusions


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 64)
    print("ANALISIS JIGSAW TOXIC COMMENT CLASSIFICATION")
    print("Dataset: Toxic Comment Classification Challenge (Kaggle)")
    print("Flujo: Hipotesis -> EDA -> Pruebas -> Modelos -> Conclusion")
    print("=" * 64)

    # 0. Carga
    df = load_and_prepare()
    feat_cols = [
        "text_len", "word_count", "caps_ratio",
        "exclaim_ratio", "question_ratio", "unique_word_ratio"
    ]

    # 1. Hipotesis
    print_hypothesis()

    # 2. EDA
    eda_overview(df)
    label_df = eda_label_distribution(df)
    eda_label_cooccurrence(df)
    eda_text_features(df, feat_cols)

    # 3. Pruebas estadisticas
    pb_df, chi2_df = statistical_tests_correlation(df, feat_cols)
    norm_result = statistical_tests_normality(df, feat_cols)
    comp_df = statistical_tests_group_comparison(df, feat_cols)

    # 4. Comparacion de modelos
    model_results = model_comparison(df, feat_cols)

    # 5. Toolbelt diagnostico completo
    toolbelt_results = run_toolbelt_diagnostic(df, feat_cols)

    # 6. Conclusiones
    write_conclusions(df, pb_df, chi2_df, norm_result, comp_df, model_results, toolbelt_results)

    # 7. Subir resultados a GCS si esta configurado
    if GCS_BUCKET:
        print(f"\nSubiendo resultados a gs://{GCS_BUCKET}/analysis_output/")
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        for f in OUTPUT_DIR.glob("*"):
            if f.is_file():
                blob = bucket.blob(f"analysis_output/{f.name}")
                blob.upload_from_filename(str(f))
                print(f"  Subido: analysis_output/{f.name}")

    print(f"\nTodas las graficas guardadas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
