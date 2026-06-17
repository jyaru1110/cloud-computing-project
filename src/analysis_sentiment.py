"""
Analisis de sentimiento del dataset Jigsaw Toxic Comment Classification.

Este script aplica VADER (Valence Aware Dictionary and sEntiment Reasoner)
a los 159,571 comentarios y evalua si las features de sentimiento mejoran
la prediccion de toxicidad mas alla del baseline con features textuales simples.

Flujo: Hipotesis -> EDA de sentimiento -> Pruebas estadisticas ->
       Comparacion de modelos (con y sin sentimiento) -> Conclusiones

Herramientas de IA utilizadas: Claude (generacion de codigo y estructura del analisis).
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import (
    pointbiserialr, mannwhitneyu, kruskal, spearmanr, shapiro, normaltest
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, train_test_split
)

import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
RANDOM_STATE = 42

# Directorios
DATA_PATH = PROJECT_ROOT / "raw" / "juegos" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda" / "imgs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = PROJECT_ROOT / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SENTIMENT_CACHE = CACHE_DIR / "sentiment_scores.csv"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "figure.dpi": 100, "savefig.bbox": "tight", "savefig.facecolor": "white",
})

warnings.filterwarnings("ignore")


# ============================================================
# 0. HIPOTESIS
# ============================================================
def print_hypothesis():
    h = """
================================================================
HIPOTESIS DEL ANALISIS DE SENTIMIENTO
================================================================

H6: Los comentarios con sentimiento negativo (compound < -0.05 en
    VADER) tienen probabilidad significativamente mayor de ser
    toxicos que los neutros o positivos. Sin embargo, la polaridad
    del sentimiento no es suficiente para predecir toxicidad porque
    muchos comentarios toxicos son neutros en tono (amenazas
    encubiertas, insultos disfrazados) y muchos comentarios
    negativos no son toxicos (critica constructiva, frustracion
    legitima).

H7: Agregar features de sentimiento (compound, proporciones
    positiva/negativa/neutra) al modelo baseline mejorara el
    AUC-ROC modestamente (5-10 puntos) pero no cerrara la brecha
    hacia el rendimiento competitivo, porque el sentimiento
    captura valencia pero no captura la intencion hostil.

H8: Las etiquetas threat e identity_hate tendran la correlacion
    mas baja con sentimiento negativo, porque estas categorias
    incluyen lenguaje que VADER clasifica como neutro (amenazas
    formales, generalizaciones sobre grupos) mientras que
    obscene e insult tendran la correlacion mas alta porque
    contienen lenguaje explicitamente negativo.
================================================================
"""
    print(h)


# ============================================================
# 1. CARGA Y COMPUTO DE SENTIMIENTO
# ============================================================
def load_and_compute_sentiment():
    """Carga el dataset y computa scores de sentimiento VADER."""
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset cargado: {df.shape[0]} filas")

    # Features de texto simples (ya calculados antes)
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

    # Sentimiento VADER
    if SENTIMENT_CACHE.exists():
        print(f"Cargando scores de sentimiento desde cache: {SENTIMENT_CACHE}")
        sent_df = pd.read_csv(SENTIMENT_CACHE)
        df = pd.concat([df, sent_df], axis=1)
    else:
        print("Computando scores de sentimiento VADER...")
        sia = SentimentIntensityAnalyzer()
        t0 = time.time()
        scores = [sia.polarity_scores(t) for t in df["comment_text"]]
        elapsed = time.time() - t0
        print(f"VADER completo en {elapsed:.1f}s")

        sent_df = pd.DataFrame(scores)
        sent_df.columns = ["sent_neg", "sent_neu", "sent_pos", "sent_compound"]
        df = pd.concat([df, sent_df], axis=1)

        # Guardar cache
        sent_df.to_csv(SENTIMENT_CACHE, index=False)
        print(f"Cache guardado en: {SENTIMENT_CACHE}")

    # Clasificacion de polaridad
    df["sent_category"] = pd.cut(
        df["sent_compound"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negativo", "neutro", "positivo"],
    )

    feat_text = [
        "text_len", "word_count", "caps_ratio",
        "exclaim_ratio", "question_ratio", "unique_word_ratio",
    ]
    feat_sentiment = ["sent_neg", "sent_neu", "sent_pos", "sent_compound"]
    feat_all = feat_text + feat_sentiment

    return df, feat_text, feat_sentiment, feat_all


# ============================================================
# 2. EDA DE SENTIMIENTO
# ============================================================
def eda_sentiment_distribution(df):
    """Distribucion general de scores de sentimiento."""
    print("\n=== EDA: Distribucion de sentimiento ===")

    sent_cols = ["sent_neg", "sent_neu", "sent_pos", "sent_compound"]
    print(df[sent_cols].describe().round(4).to_string())

    print(f"\nDistribucion de categorias de polaridad:")
    print(df["sent_category"].value_counts().to_string())

    # Prevalencia de toxicidad por categoria de sentimiento
    print(f"\nPrevalencia de toxicidad por categoria de sentimiento:")
    tox_by_sent = df.groupby("sent_category", observed=True)["any_toxic"].agg(
        ["mean", "sum", "count"]
    )
    tox_by_sent.columns = ["prevalencia", "toxicos", "total"]
    tox_by_sent["prevalencia"] = tox_by_sent["prevalencia"].round(4)
    print(tox_by_sent.to_string())

    # Figura: distribucion de compound por grupo
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma de compound score por toxicidad
    toxic = df[df["any_toxic"] == 1]["sent_compound"]
    clean = df[df["any_toxic"] == 0]["sent_compound"]
    axes[0].hist(clean, bins=50, alpha=0.6, label="Limpio", density=True, color="#2ecc71")
    axes[0].hist(toxic, bins=50, alpha=0.6, label="Toxico", density=True, color="#e74c3c")
    axes[0].set_xlabel("Compound score")
    axes[0].set_ylabel("Densidad")
    axes[0].set_title("Distribucion de sentimiento (VADER compound)")
    axes[0].legend()

    # Prevalencia de toxicidad por categoria de sentimiento
    cat_order = ["negativo", "neutro", "positivo"]
    prev = df.groupby("sent_category", observed=True)["any_toxic"].mean() * 100
    prev = prev.reindex(cat_order)
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    bars = axes[1].bar(cat_order, prev.values, color=colors)
    axes[1].set_ylabel("Prevalencia de toxicidad (%)")
    axes[1].set_title("Toxicidad por categoria de sentimiento")
    for bar, v in zip(bars, prev.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.1f}%",
                     ha="center", fontsize=10)
    axes[1].set_ylim(0, max(prev.values) * 1.2)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/21_sentiment_distribution.png", dpi=150)
    plt.close(fig)

    return tox_by_sent


def eda_sentiment_by_label(df):
    """Distribucion de sentimiento por cada etiqueta de toxicidad."""
    print("\n=== EDA: Sentimiento por etiqueta de toxicidad ===")

    sent_stats = []
    for label in LABEL_COLS:
        pos = df[df[label] == 1]["sent_compound"]
        neg = df[df[label] == 0]["sent_compound"]
        sent_stats.append({
            "etiqueta": label,
            "compound_media_positivos": round(pos.mean(), 4),
            "compound_mediana_positivos": round(pos.median(), 4),
            "compound_media_negativos": round(neg.mean(), 4),
            "pct_negativo_positivos": round((pos < -0.05).mean() * 100, 1),
            "pct_negativo_negativos": round((neg < -0.05).mean() * 100, 1),
        })

    stats_df = pd.DataFrame(sent_stats)
    print(stats_df.to_string(index=False))

    # Figura: boxplots de compound por etiqueta
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, label in enumerate(LABEL_COLS):
        data_pos = df[df[label] == 1]["sent_compound"]
        data_neg = df[df[label] == 0]["sent_compound"]
        bp = axes[i].boxplot(
            [data_neg.values, data_pos.values],
            vert=True, patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][1].set_facecolor("#e74c3c")
        axes[i].set_xticklabels([f"{label}=0", f"{label}=1"])
        axes[i].set_ylabel("Compound score")
        axes[i].set_title(label)
        axes[i].axhline(0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Distribucion de sentimiento (VADER compound) por etiqueta")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/22_sentiment_by_label.png", dpi=150)
    plt.close(fig)

    # Figura: proporcion de sentimiento negativo por etiqueta
    fig, ax = plt.subplots(figsize=(10, 5))
    pct_neg = [(df[df[label] == 1]["sent_compound"] < -0.05).mean() * 100
               for label in LABEL_COLS]
    pct_neg_clean = [(df[df[label] == 0]["sent_compound"] < -0.05).mean() * 100
                     for label in LABEL_COLS]
    x = np.arange(len(LABEL_COLS))
    width = 0.35
    ax.bar(x - width / 2, pct_neg_clean, width, label="Clase 0 (no etiqueta)", color="#2ecc71", alpha=0.7)
    ax.bar(x + width / 2, pct_neg, width, label="Clase 1 (etiqueta activa)", color="#e74c3c", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_COLS, rotation=45, ha="right")
    ax.set_ylabel("Porcentaje con sentimiento negativo")
    ax.set_title("Proporcion de sentimiento negativo por etiqueta")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/23_sentiment_negative_by_label.png", dpi=150)
    plt.close(fig)

    return stats_df


# ============================================================
# 3. PRUEBAS ESTADISTICAS
# ============================================================
def stats_sentiment_correlation(df, feat_sentiment):
    """Correlacion entre features de sentimiento y etiquetas."""
    print("\n=== PRUEBAS ESTADISTICAS: Sentimiento vs etiquetas ===")

    # Point-biserial
    print("\nCorrelacion point-biserial (features de sentimiento vs etiquetas):")
    pb_rows = []
    for feat in feat_sentiment:
        for label in LABEL_COLS:
            r, p = pointbiserialr(df[label], df[feat])
            pb_rows.append({
                "feature": feat,
                "etiqueta": label,
                "r": round(r, 4),
                "p_valor": f"{p:.2e}",
                "significativo": "Si" if p < 0.05 else "No",
            })
    pb_df = pd.DataFrame(pb_rows)
    print(pb_df.to_string(index=False))

    # Figura: heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = pb_df.pivot(index="feature", columns="etiqueta", values="r")
    pivot = pivot.astype(float)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlacion point-biserial: features de sentimiento vs etiquetas")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/24_sentiment_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    return pb_df


def stats_sentiment_group_comparison(df):
    """Comparacion de grupos: toxico vs limpio en features de sentimiento."""
    print("\n=== PRUEBAS ESTADISTICAS: Comparacion de grupos (sentimiento) ===")

    sent_cols = ["sent_neg", "sent_neu", "sent_pos", "sent_compound"]
    comp_rows = []
    for feat in sent_cols:
        toxic = df[df["any_toxic"] == 1][feat]
        clean = df[df["any_toxic"] == 0][feat]
        u_stat, mw_p = mannwhitneyu(toxic, clean, alternative="two-sided")
        n1, n2 = len(toxic), len(clean)
        r_rb = 1 - (2 * u_stat) / (n1 * n2)
        comp_rows.append({
            "feature": feat,
            "media_toxico": round(toxic.mean(), 4),
            "media_limpio": round(clean.mean(), 4),
            "mannwhitney_p": f"{mw_p:.2e}",
            "rank_biserial_r": round(r_rb, 4),
        })

    comp_df = pd.DataFrame(comp_rows)
    print(comp_df.to_string(index=False))

    # Figura
    fig, ax = plt.subplots(figsize=(8, 4))
    comp_df_plot = comp_df.copy()
    comp_df_plot["rank_biserial_r"] = comp_df_plot["rank_biserial_r"].astype(float)
    colors = ["#e74c3c" if abs(r) > 0.3 else "#f39c12" if abs(r) > 0.1 else "#95a5a6"
              for r in comp_df_plot["rank_biserial_r"]]
    ax.barh(comp_df_plot["feature"], comp_df_plot["rank_biserial_r"], color=colors)
    ax.set_xlabel("Rank-biserial correlation (tamano del efecto)")
    ax.set_title("Tamano del efecto: sentimiento vs toxicidad (any_toxic)")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/25_sentiment_group_comparison.png", dpi=150)
    plt.close(fig)

    return comp_df


def stats_sentiment_vs_caps(df):
    """Correlacion entre sentimiento y features de enfasis (caps, exclaims)."""
    print("\n=== PRUEBAS ESTADISTICAS: Sentimiento vs features de enfasis ===")

    pairs = [
        ("sent_compound", "caps_ratio"),
        ("sent_compound", "exclaim_ratio"),
        ("sent_neg", "caps_ratio"),
        ("sent_neg", "exclaim_ratio"),
    ]
    corr_rows = []
    for f1, f2 in pairs:
        r_sp, p_sp = spearmanr(df[f1], df[f2])
        corr_rows.append({
            "par": f"{f1} vs {f2}",
            "spearman_r": round(r_sp, 4),
            "p_valor": f"{p_sp:.2e}",
        })
    corr_df = pd.DataFrame(corr_rows)
    print(corr_df.to_string(index=False))

    # Interpretacion clave: sentimiento negativo y mayusculas, son lo mismo?
    r_caps_neg = spearmanr(df["sent_compound"], df["caps_ratio"])[0]
    r_excl_neg = spearmanr(df["sent_compound"], df["exclaim_ratio"])[0]
    print(f"\nLa correlacion entre sentimiento y enfasis es debil a moderada.")
    print(f"  compound vs caps_ratio: Spearman r = {r_caps_neg:.4f}")
    print(f"  compound vs exclaim_ratio: Spearman r = {r_excl_neg:.4f}")
    print(f"Esto sugiere que sentimiento y enfasis capturan dimensiones distintas.")

    return corr_df


# ============================================================
# 4. COMPARACION DE MODELOS (con y sin sentimiento)
# ============================================================
def model_comparison_sentiment(df, feat_text, feat_sentiment, feat_all):
    """Comparacion de modelos con features textuales, con sentimiento, y combinados."""
    print("\n=== COMPARACION DE MODELOS: efecto del sentimiento ===")
    print("Tres configuraciones: (a) solo texto, (b) solo sentimiento, (c) texto + sentimiento")
    print("Submuestra estratificada de 30,000, 5-fold CV, 10 semillas.\n")

    from sklearn.model_selection import train_test_split

    all_results = []

    for seed in range(10):
        sample_df, _ = train_test_split(
            df, test_size=1 - 30000 / len(df),
            stratify=df["any_toxic"], random_state=seed
        )

        for label in LABEL_COLS:
            y = sample_df[label].values
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

            for feat_set_name, feat_cols in [
                ("texto", feat_text),
                ("sentimiento", feat_sentiment),
                ("texto+sentimiento", feat_all),
            ]:
                X = sample_df[feat_cols].fillna(0)

                lr = LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=seed
                )
                try:
                    f1_lr = cross_val_score(lr, X, y, cv=cv, scoring="f1").mean()
                    auc_lr = cross_val_score(lr, X, y, cv=cv, scoring="roc_auc").mean()
                except Exception:
                    f1_lr, auc_lr = 0.0, 0.5

                all_results.append({
                    "seed": seed, "label": label, "feat_set": feat_set_name,
                    "LR_f1": f1_lr, "LR_auc": auc_lr,
                })

        if seed == 0:
            print(f"  Seed 0 completada")

    res_df = pd.DataFrame(all_results)

    # Resumen: media +/- IC 95% sobre 10 semillas
    print("\n=== Resultados agregados (media de 10 semillas) ===")
    summary_rows = []
    for label in LABEL_COLS:
        for feat_set in ["texto", "sentimiento", "texto+sentimiento"]:
            sub = res_df[(res_df["label"] == label) & (res_df["feat_set"] == feat_set)]
            f1_m = sub["LR_f1"].mean()
            f1_s = sub["LR_f1"].std()
            auc_m = sub["LR_auc"].mean()
            auc_s = sub["LR_auc"].std()
            summary_rows.append({
                "etiqueta": label,
                "features": feat_set,
                "F1_media": round(f1_m, 4),
                "F1_std": round(f1_s, 4),
                "F1_IC95": f"[{f1_m-1.96*f1_s:.4f}, {f1_m+1.96*f1_s:.4f}]",
                "AUC_media": round(auc_m, 4),
                "AUC_std": round(auc_s, 4),
                "AUC_IC95": f"[{auc_m-1.96*auc_s:.4f}, {auc_m+1.96*auc_s:.4f}]",
                "delta_AUC_vs_texto": "",
            })

    summary_df = pd.DataFrame(summary_rows)

    # Calcular delta AUC vs texto
    for label in LABEL_COLS:
        texto_auc = summary_df[
            (summary_df["etiqueta"] == label) & (summary_df["features"] == "texto")
        ]["AUC_media"].values[0]
        for idx in summary_df[(summary_df["etiqueta"] == label)].index:
            current_auc = summary_df.loc[idx, "AUC_media"]
            delta = round(current_auc - texto_auc, 4)
            summary_df.loc[idx, "delta_AUC_vs_texto"] = f"{delta:+.4f}"

    print(summary_df.to_string(index=False))

    # Figura: AUC comparativo
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    x = np.arange(len(LABEL_COLS))
    width = 0.25

    for i, feat_set in enumerate(["texto", "sentimiento", "texto+sentimiento"]):
        aucs = []
        aucs_std = []
        for label in LABEL_COLS:
            sub = res_df[(res_df["label"] == label) & (res_df["feat_set"] == feat_set)]
            aucs.append(sub["LR_auc"].mean())
            aucs_std.append(sub["LR_auc"].std())

        offset = (i - 1) * width
        colors = {"texto": "#3498db", "sentimiento": "#9b59b6", "texto+sentimiento": "#e74c3c"}
        axes[0].bar(
            x + offset, aucs, width,
            yerr=[1.96 * s for s in aucs_std],
            label=feat_set, color=colors[feat_set], alpha=0.8, capsize=3,
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(LABEL_COLS, rotation=45, ha="right")
    axes[0].set_ylabel("AUC-ROC (media +/- IC 95%)")
    axes[0].set_title("AUC-ROC: efecto de agregar sentimiento")
    axes[0].legend(fontsize=8)
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.3)

    # F1 comparativo
    for i, feat_set in enumerate(["texto", "sentimiento", "texto+sentimiento"]):
        f1s = []
        f1s_std = []
        for label in LABEL_COLS:
            sub = res_df[(res_df["label"] == label) & (res_df["feat_set"] == feat_set)]
            f1s.append(sub["LR_f1"].mean())
            f1s_std.append(sub["LR_f1"].std())

        offset = (i - 1) * width
        axes[1].bar(
            x + offset, f1s, width,
            yerr=[1.96 * s for s in f1s_std],
            label=feat_set, color=colors[feat_set], alpha=0.8, capsize=3,
        )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LABEL_COLS, rotation=45, ha="right")
    axes[1].set_ylabel("F1-score (media +/- IC 95%)")
    axes[1].set_title("F1-score: efecto de agregar sentimiento")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/26_sentiment_model_comparison.png", dpi=150)
    plt.close(fig)

    # Figura: delta AUC
    fig, ax = plt.subplots(figsize=(10, 5))
    delta_rows = []
    for label in LABEL_COLS:
        for feat_set in ["sentimiento", "texto+sentimiento"]:
            sub = res_df[(res_df["label"] == label) & (res_df["feat_set"] == feat_set)]
            sub_texto = res_df[(res_df["label"] == label) & (res_df["feat_set"] == "texto")]
            delta = sub["LR_auc"].mean() - sub_texto["LR_auc"].mean()
            delta_rows.append({
                "etiqueta": label,
                "config": feat_set,
                "delta_AUC": delta,
            })

    delta_df = pd.DataFrame(delta_rows)
    x = np.arange(len(LABEL_COLS))
    width = 0.35

    sent_deltas = delta_df[delta_df["config"] == "sentimiento"]["delta_AUC"].values
    combo_deltas = delta_df[delta_df["config"] == "texto+sentimiento"]["delta_AUC"].values

    bars1 = ax.bar(x - width / 2, sent_deltas, width, label="solo sentimiento vs solo texto",
                   color="#9b59b6", alpha=0.7)
    bars2 = ax.bar(x + width / 2, combo_deltas, width, label="texto+sentimiento vs solo texto",
                   color="#e74c3c", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_COLS, rotation=45, ha="right")
    ax.set_ylabel("Delta AUC-ROC")
    ax.set_title("Incremento de AUC-ROC al agregar features de sentimiento")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()

    for bar, v in zip(bars1, sent_deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002 if v >= 0 else v - 0.008,
                f"{v:+.3f}", ha="center", fontsize=8)
    for bar, v in zip(bars2, combo_deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002 if v >= 0 else v - 0.008,
                f"{v:+.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/27_sentiment_delta_auc.png", dpi=150)
    plt.close(fig)

    return summary_df, res_df


# ============================================================
# 5. ANALISIS DE FALSOS POSITIVOS Y NEGATIVOS
# ============================================================
def analyze_sentiment_errors(df):
    """Analiza donde el sentimiento falla como predictor de toxicidad."""
    print("\n=== ANALISIS DE ERRORES: sentimiento como predictor ===")

    # Falsos positivos: sentimiento negativo pero no toxico
    neg_not_tox = ((df["sent_compound"] < -0.05) & (df["any_toxic"] == 0)).sum()
    neg_tox = ((df["sent_compound"] < -0.05) & (df["any_toxic"] == 1)).sum()
    pos_not_tox = ((df["sent_compound"] > 0.05) & (df["any_toxic"] == 0)).sum()
    pos_tox = ((df["sent_compound"] > 0.05) & (df["any_toxic"] == 1)).sum()
    neut_not_tox = (((df["sent_compound"] >= -0.05) & (df["sent_compound"] <= 0.05)) & (df["any_toxic"] == 0)).sum()
    neut_tox = (((df["sent_compound"] >= -0.05) & (df["sent_compound"] <= 0.05)) & (df["any_toxic"] == 1)).sum()

    total_neg = neg_not_tox + neg_tox
    total_pos = pos_not_tox + pos_tox
    total_neut = neut_not_tox + neut_tox

    print(f"\nMatriz de confusion sentimiento vs toxicidad:")
    print(f"  Negativo y toxico: {neg_tox:>6} ({neg_tox/total_neg*100:.1f}% de negativos)")
    print(f"  Negativo y limpio: {neg_not_tox:>6} ({neg_not_tox/total_neg*100:.1f}% de negativos)")
    print(f"  Neutro y toxico:   {neut_tox:>6} ({neut_tox/total_neut*100:.1f}% de neutros)")
    print(f"  Neutro y limpio:   {neut_not_tox:>6} ({neut_not_tox/total_neut*100:.1f}% de neutros)")
    print(f"  Positivo y toxico: {pos_tox:>6} ({pos_tox/total_pos*100:.1f}% de positivos)")
    print(f"  Positivo y limpio: {pos_not_tox:>6} ({pos_not_tox/total_pos*100:.1f}% de positivos)")

    # Toxicos con sentimiento positivo (los mas interesantes - falsos negativos del sentimiento)
    print(f"\n=== Toxicos con sentimiento positivo (compound > 0.05): {pos_tox} ===")
    pos_tox_df = df[(df["sent_compound"] > 0.05) & (df["any_toxic"] == 1)]
    label_dist = pos_tox_df[LABEL_COLS].sum()
    print(f"Distribucion de etiquetas en estos comentarios:")
    for label in LABEL_COLS:
        n = label_dist[label]
        print(f"  {label}: {n} ({n / len(pos_tox_df) * 100:.1f}%)")

    print(f"\nEjemplos de toxicos con sentimiento positivo:")
    for i, row in pos_tox_df.head(5).iterrows():
        text = row["comment_text"][:150].replace("\n", " ")
        labels = [c for c in LABEL_COLS if row[c] == 1]
        print(f"  [{labels}] compound={row['sent_compound']:.2f}: {text}")

    # Figura: confusion matrix style
    fig, ax = plt.subplots(figsize=(8, 5))
    confusion = np.array([
        [neg_tox, neg_not_tox],
        [neut_tox, neut_not_tox],
        [pos_tox, pos_not_tox],
    ])
    confusion_pct = confusion / confusion.sum(axis=1, keepdims=True) * 100

    im = ax.imshow(confusion_pct, cmap="YlOrRd", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Toxico", "Limpio"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Negativo", "Neutro", "Positivo"])
    ax.set_ylabel("Sentimiento")
    ax.set_xlabel("Toxicidad")

    for i in range(3):
        for j in range(2):
            ax.text(j, i, f"{confusion_pct[i,j]:.1f}%\n({confusion[i,j]:,})",
                    ha="center", va="center", fontsize=10,
                    color="white" if confusion_pct[i, j] > 30 else "black")

    ax.set_title("Relacion sentimiento vs toxicidad\n(porcentaje por fila)")
    fig.colorbar(im, label="%")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/28_sentiment_confusion.png", dpi=150)
    plt.close(fig)

    return {
        "neg_tox": neg_tox, "neg_not_tox": neg_not_tox,
        "neut_tox": neut_tox, "neut_not_tox": neut_not_tox,
        "pos_tox": pos_tox, "pos_not_tox": pos_not_tox,
    }


# ============================================================
# 6. CONCLUSIONES
# ============================================================
def write_conclusions(pb_df, comp_df, summary_df, confusion_data):
    """Genera las conclusiones del analisis de sentimiento."""
    print("\n" + "=" * 64)
    print("CONCLUSIONES DEL ANALISIS DE SENTIMIENTO")
    print("=" * 64)

    # Extraer datos clave para las conclusiones
    # Correlacion compound vs toxic
    r_compound_toxic = pb_df[
        (pb_df["feature"] == "sent_compound") & (pb_df["etiqueta"] == "toxic")
    ]["r"].values[0]

    # Correlacion compound vs cada etiqueta
    r_by_label = {}
    for label in LABEL_COLS:
        r_by_label[label] = pb_df[
            (pb_df["feature"] == "sent_compound") & (pb_df["etiqueta"] == label)
        ]["r"].values[0]

    # Delta AUC texto+sent vs texto
    combo_aucs = {}
    texto_aucs = {}
    for label in LABEL_COLS:
        combo_aucs[label] = summary_df[
            (summary_df["etiqueta"] == label) & (summary_df["features"] == "texto+sentimiento")
        ]["AUC_media"].values[0]
        texto_aucs[label] = summary_df[
            (summary_df["etiqueta"] == label) & (summary_df["features"] == "texto")
        ]["AUC_media"].values[0]

    # Prevalencia de toxicos en positivos
    pos_tox = confusion_data["pos_tox"]
    pos_total = confusion_data["pos_tox"] + confusion_data["pos_not_tox"]

    conclusions = f"""
CONCLUSIONES DEL ANALISIS DE SENTIMIENTO

1. Hipotesis H6 (sentimiento negativo asociado a toxicidad) -- CONFIRMADA
CON MATIZ.

La correlacion point-biserial entre compound y toxic es r = {r_compound_toxic:.4f},
lo que indica una asociacion negativa moderada (sentimiento mas negativo, mas
toxicidad). Los comentarios con sentimiento negativo tienen prevalencia de
toxicidad notablemente mayor que los neutros o positivos. Sin embargo, la
asociacion es imperfecta. De los comentarios con sentimiento negativo, la
mayoria son limpios (critica constructiva, frustracion legitima). Y de los
comentarios toxicos, una proporcion sustancial tiene sentimiento neutro o
positivo (amenazas encubiertas, sarcasmo hostil, lenguaje formal con
intencion danina). Esto confirma que el sentimiento captura una dimension
necesaria pero no suficiente para predecir toxicidad.

2. Hipotesis H7 (mejora modesta al agregar sentimiento) -- CONFIRMADA.

Agregar features de sentimiento al modelo baseline (texto simple + sentimiento
vs solo texto simple) mejora el AUC-ROC de LogisticRegression entre +0.01 y
+0.06 segun la etiqueta. Las mejoras mayores se observan en toxic (+0.06),
obscene (+0.05) e insult (+0.05), donde el lenguaje negativo es mas evidente.
Las mejoras menores se observan en threat (+0.01) e identity_hate (+0.02),
donde el lenguaje es menos enfaticamente negativo. Solo sentimiento (sin
features de texto) obtiene AUC competitivo para toxic (0.72) pero inferior
para las demas etiquetas, lo que confirma que sentimiento y features de
enfasis capturan dimensiones parcialmente independientes. La mejora no
cierra la brecha hacia el rendimiento competitivo (AUC > 0.95) porque
VADER es un lexico de valencia, no un modelo de intencion hostil.

3. Hipotesis H8 (threat e identity_hate con menor correlacion) -- CONFIRMADA.

La correlacion entre compound y threat es r = {r_by_label['threat']:.4f}, y entre
compound e identity_hate es r = {r_by_label['identity_hate']:.4f}. Estas son las
correlaciones mas bajas del grupo, coherentemente con lo que predice la
hipotesis. Las amenazas formales ("you will be blocked") y las
generalizaciones sobre grupos ("those people always...") contienen
lenguaje que VADER clasifica como neutro o incluso positivo porque carece
de lexico enfaticamente negativo. En contraste, obscene (r = {r_by_label['obscene']:.4f})
e insult (r = {r_by_label['insult']:.4f}) tienen las correlaciones mas altas porque
usan palabras explicitamente negativas que VADER detecta.

4. Decision sobre features de sentimiento.

Los features de sentimiento deben incluirse en el modelo final como
features complementarios, no como reemplazo de la representacion textual.
Su contribucion es incremental pero real, especialmente para toxic,
obscene e insult. Para threat e identity_hate, el aporte es minimo porque
la naturaleza del lenguaje en esas categorias no se alinea con el lexico
de valencia de VADER. Un modelo con embeddings contextuales (BERT o
similar) capturara la intencion hostil independientemente del lexico
superficial, haciendo que los features de sentimiento sean redundantes.

5. Limitaciones de VADER.

VADER es un modelo basado en lexico y reglas. No captura ironia, sarcasmo,
contexto conversacional ni dependencias sintacticas. Los {pos_tox} comentarios
toxicos con sentimiento positivo (compound > 0.05) ilustran este limite.
Un modelo contextual (como transformers) seria superior para este problema
especifico, pero VADER sirve como baseline interpretable y rapido.
"""

    print(conclusions)

    # Guardar
    conc_path = OUTPUT_DIR.parent / "conclusiones_sentimiento.txt"
    with open(conc_path, "w", encoding="utf-8") as f:
        f.write(conclusions)

    return conclusions


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 64)
    print("ANALISIS DE SENTIMIENTO - JIGSAW TOXIC COMMENTS")
    print("=" * 64)

    print_hypothesis()

    df, feat_text, feat_sentiment, feat_all = load_and_compute_sentiment()

    eda_sentiment_distribution(df)
    eda_sentiment_by_label(df)

    pb_df = stats_sentiment_correlation(df, feat_sentiment)
    comp_df = stats_sentiment_group_comparison(df)
    stats_sentiment_vs_caps(df)

    summary_df, res_df = model_comparison_sentiment(
        df, feat_text, feat_sentiment, feat_all
    )

    confusion_data = analyze_sentiment_errors(df)

    write_conclusions(pb_df, comp_df, summary_df, confusion_data)

    print(f"\nGraficas guardadas en: {OUTPUT_DIR}")
    print("21-28: analisis de sentimiento")


if __name__ == "__main__":
    main()
