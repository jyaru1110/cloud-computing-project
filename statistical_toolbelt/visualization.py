"""
Funciones de visualización para el Statistical Toolbelt.

Este módulo contiene las funciones privadas _build_* que construyen las figuras
de matplotlib para cada diagnóstico. Las funciones de diagnóstico en los
módulos correspondientes llaman a estos builders internamente.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# =========================================================
# Configuración de estilos centralizada
# =========================================================

# Paleta de colores para estados
COLOR_PASS = "#2ecc71"   # Verde
COLOR_WARN = "#f39c12"   # Amarillo
COLOR_FAIL = "#e74c3c"   # Rojo
COLOR_PRIMARY = "#3498db"  # Azul primario

# Configuración global de matplotlib
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _create_figure_with_text(message: str) -> Figure:
    """Crea una figura con un mensaje de texto cuando no hay datos para graficar."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center",
        fontsize=14, color="#666",
        transform=ax.transAxes
    )
    ax.set_axis_off()
    fig.tight_layout()
    return fig


# =========================================================
# Builders para overview.py
# =========================================================

def _build_missing_barplot(missing_df: pd.DataFrame) -> Figure:
    """Construye gráfico de barras horizontales de missing percentage."""
    if missing_df.empty or missing_df["missing_pct"].sum() == 0:
        return _create_figure_with_text("Sin valores faltantes")
    
    # Filtrar solo columnas con missing
    plot_df = missing_df[missing_df["missing_pct"] > 0].copy()
    if plot_df.empty:
        return _create_figure_with_text("Sin valores faltantes")
    
    # Ordenar descendente
    plot_df = plot_df.sort_values("missing_pct", ascending=True)
    
    # Asignar color por severidad
    def get_color(pct):
        if pct < 5:
            return COLOR_PASS
        elif pct < 20:
            return COLOR_WARN
        else:
            return COLOR_FAIL
    
    colors = [get_color(p) for p in plot_df["missing_pct"]]
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.4)))
    ax.barh(plot_df["column"], plot_df["missing_pct"], color=colors)
    ax.set_xlabel("Porcentaje de valores faltantes")
    ax.set_title("Reporte de valores faltantes")
    
    # Añadir etiquetas de porcentaje
    for i, (pct, col) in enumerate(zip(plot_df["missing_pct"], plot_df["column"])):
        ax.text(pct + 0.5, i, f"{pct:.1f}%", va="center", fontsize=9)
    
    ax.set_xlim(0, max(plot_df["missing_pct"]) * 1.15)
    fig.tight_layout()
    return fig


def _build_duplicate_barplot(duplicate_df: pd.DataFrame) -> Figure:
    """Construye gráfico de barras de duplicados vs únicos."""
    if duplicate_df.empty:
        return _create_figure_with_text("Sin datos de duplicados")
    
    n_rows = duplicate_df.iloc[0]["n_rows"]
    n_duplicates = duplicate_df.iloc[0]["duplicate_rows"]
    n_unique = n_rows - n_duplicates
    dup_pct = duplicate_df.iloc[0]["duplicate_pct"] * 100
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Únicos", "Duplicados"], [n_unique, n_duplicates], 
                  color=[COLOR_PASS, COLOR_WARN])
    ax.set_ylabel("Número de filas")
    ax.set_title(f"Reporte de duplicados ({dup_pct:.2f}%)")
    
    # Anotar valores
    for bar, val in zip(bars, [n_unique, n_duplicates]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + n_rows*0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=11)
    
    fig.tight_layout()
    return fig


def _build_cardinality_barplot(cardinality_df: pd.DataFrame) -> Figure:
    """Construye gráfico de cardinalidad de variables categóricas."""
    if cardinality_df.empty:
        return _create_figure_with_text("Sin variables categóricas")
    
    plot_df = cardinality_df.sort_values("n_unique", ascending=True).tail(15)
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.4)))
    ax.barh(plot_df["column"], plot_df["n_unique"], color=COLOR_PRIMARY)
    ax.set_xlabel("Número de valores únicos")
    ax.set_title("Cardinalidad de variables categóricas")
    
    for i, n in enumerate(plot_df["n_unique"]):
        ax.text(n + max(plot_df["n_unique"])*0.01, i, str(n), va="center", fontsize=9)
    
    ax.set_xlim(0, max(plot_df["n_unique"]) * 1.15)
    fig.tight_layout()
    return fig


# =========================================================
# Builders para descriptive.py
# =========================================================

def _build_continuous_boxplot(df: pd.DataFrame, cols: list) -> Figure:
    """Construye boxplots de variables continuas."""
    if not cols:
        return _create_figure_with_text("Sin variables continuas")
    
    # Normalizar para comparar en misma escala
    plot_data = df[cols].copy()
    
    # Verificar rangos muy diferentes
    mins = plot_data.min()
    maxs = plot_data.max()
    ranges = maxs - mins
    
    if ranges.max() > ranges.min() * 100:
        # Normalizar si los rangos difieren demasiado
        plot_data = (plot_data - mins) / (ranges + 1e-9)
        ylabel = "Valor normalizado"
    else:
        ylabel = "Valor"
    
    n_cols = min(len(cols), 12)
    cols_subset = cols[:n_cols]
    
    fig, ax = plt.subplots(figsize=(12, max(4, n_cols * 0.4)))
    plot_data[cols_subset].boxplot(ax=ax, vert=False, patch_artist=True)
    ax.set_xlabel(ylabel)
    ax.set_title("Distribución de variables continuas")
    fig.tight_layout()
    return fig


def _build_discrete_freq_grid(discrete_df: pd.DataFrame) -> Figure:
    """Construye grid de barras de frecuencia para discretas."""
    if discrete_df.empty:
        return _create_figure_with_text("Sin variables discretas")
    
    # Obtener variables únicas
    variables = discrete_df["variable"].unique()[:6]
    if len(variables) == 0:
        return _create_figure_with_text("Sin datos discretos")
    
    n_vars = len(variables)
    fig, axes = plt.subplots(nrows=n_vars, ncols=1, figsize=(10, 3 * n_vars))
    
    if n_vars == 1:
        axes = [axes]
    
    for ax, var in zip(axes, variables):
        var_data = discrete_df[discrete_df["variable"] == var].head(10)
        ax.bar(var_data["value"].astype(str), var_data["count"], color=COLOR_PRIMARY, alpha=0.7)
        ax.set_title(f"Distribución de {var}")
        ax.set_xlabel("Valor")
        ax.set_ylabel("Frecuencia")
        ax.tick_params(axis="x", rotation=45)
    
    fig.tight_layout()
    return fig


def _build_outlier_boxplot(outlier_df: pd.DataFrame, df: pd.DataFrame) -> Figure:
    """Construye boxplots con outliers marcados."""
    if outlier_df.empty:
        return _create_figure_with_text("Sin outliers detectados por IQR")
    
    # Obtener columnas con outliers
    cols_with_outliers = outlier_df[outlier_df["outlier_pct"] > 0]["column"].tolist()
    
    if not cols_with_outliers:
        return _create_figure_with_text("Sin outliers detectados por IQR")
    
    cols_subset = cols_with_outliers[:8]
    plot_data = df[cols_subset].copy()
    
    fig, ax = plt.subplots(figsize=(12, max(4, len(cols_subset) * 0.5)))
    
    bp = ax.boxplot([plot_data[c].dropna() for c in cols_subset],
                    vert=False, patch_artist=True)
    ax.set_yticklabels(cols_subset)
    
    # Colorear outliers en rojo
    for patch in bp["fliers"]:
        patch.set(marker="o", markerfacecolor=COLOR_FAIL, alpha=0.5, markersize=4)
    
    for median in bp["medians"]:
        median.set(color=COLOR_PRIMARY, linewidth=2)
    
    ax.set_xlabel("Valor")
    ax.set_title("Reporte de outliers (IQR)")
    fig.tight_layout()
    return fig


# =========================================================
# Builders para normality.py
# =========================================================

def _build_normality_grid(df: pd.DataFrame, cols: list, norm_df: pd.DataFrame) -> Figure:
    """Construye grid de histogram + QQ-plot para pruebas de normalidad."""
    if not cols:
        return _create_figure_with_text("Sin variables continuas para probar")
    
    cols_subset = cols[:6]
    n = len(cols_subset)
    
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, 4 * n))
    
    if n == 1:
        axes = np.array([axes])
    
    import scipy.stats as stats
    
    for i, col in enumerate(cols_subset):
        s = df[col].dropna()
        
        # Histograma
        axes[i, 0].hist(s, bins=30, color=COLOR_PRIMARY, alpha=0.7, edgecolor="white")
        axes[i, 0].set_title(f"Histograma - {col}")
        axes[i, 0].set_xlabel("Valor")
        axes[i, 0].set_ylabel("Frecuencia")
        
        # Verificar si pasa Shapiro
        normal = True
        if not norm_df.empty and col in norm_df["column"].values:
            row = norm_df[norm_df["column"] == col]
            if not row.empty:
                normal = bool(row.iloc[0]["normal_by_shapiro_0.05"])
        
        # Borde rojo si no es normal
        if not normal:
            for spine in axes[i, 0].spines.values():
                spine.set_edgecolor(COLOR_FAIL)
                spine.set_linewidth(2)
        
        # QQ-plot
        stats.probplot(s, dist="norm", plot=axes[i, 1])
        axes[i, 1].set_title(f"Q-Q Plot - {col}")
        
        if not normal:
            for spine in axes[i, 1].spines.values():
                spine.set_edgecolor(COLOR_FAIL)
                spine.set_linewidth(2)
    
    fig.tight_layout()
    return fig


# =========================================================
# Builders para correlation.py
# =========================================================

def _build_correlation_heatmap(corr_matrix: pd.DataFrame, title: str) -> Figure:
    """Construye heatmap de correlación."""
    if corr_matrix.empty or corr_matrix.shape[0] < 2:
        return _create_figure_with_text("Sin suficientes variables para correlación")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    
    # Anotaciones
    n = len(corr_matrix)
    for i in range(n):
        for j in range(n):
            val = corr_matrix.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=8, color="white" if abs(val) > 0.5 else "black")
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr_matrix.index, fontsize=8)
    
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Correlación")
    fig.tight_layout()
    return fig


def _build_association_barplot(assoc_df: pd.DataFrame) -> Figure:
    """Construye gráfico de asociación con target."""
    if assoc_df.empty:
        return _create_figure_with_text("Sin datos de asociación")
    
    plot_df = assoc_df.copy()
    plot_df["abs_score"] = plot_df["score"].abs()
    plot_df = plot_df.sort_values("abs_score", ascending=True).tail(15)
    
    # Color por significancia
    colors = []
    for _, row in plot_df.iterrows():
        if pd.notna(row.get("p_value")) and row["p_value"] < 0.05:
            colors.append(COLOR_PASS)
        else:
            colors.append("#95a5a6")
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.4)))
    ax.barh(plot_df["feature"], plot_df["abs_score"], color=colors)
    ax.set_xlabel("Valor absoluto del score")
    ax.set_title("Asociación con variable objetivo")
    fig.tight_layout()
    return fig


def _build_vif_barplot(vif_df: pd.DataFrame) -> Figure:
    """Construye gráfico horizontal de VIF."""
    if vif_df.empty or "VIF" not in vif_df.columns:
        return _create_figure_with_text("Sin datos de VIF")
    
    plot_df = vif_df.sort_values("VIF", ascending=True).tail(15)
    vif_clean = plot_df["VIF"].replace([np.inf, -np.inf], np.nan).dropna()
    
    if vif_clean.empty:
        return _create_figure_with_text("Sin datos de VIF válidos")
    
    colors = [COLOR_FAIL if v > 10 else COLOR_PASS for v in vif_clean]
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(vif_clean) * 0.4)))
    ax.barh(vif_clean.index, vif_clean.values, color=colors)
    ax.axvline(x=10, color=COLOR_FAIL, linestyle="--", linewidth=1.5, label="Corte (VIF=10)")
    ax.set_xlabel("VIF")
    ax.set_title("Factor de Inflación de Varianza (VIF)")
    ax.legend()
    fig.tight_layout()
    return fig


# =========================================================
# Builders para preprocessing.py
# =========================================================

def _build_imputation_barplot(imp_df: pd.DataFrame) -> Figure:
    """Construye gráfico de estrategias de imputación."""
    if imp_df.empty:
        return _create_figure_with_text("Sin datos de imputación")
    
    plot_df = imp_df[imp_df["missing_pct"] > 0].copy()
    if plot_df.empty:
        return _create_figure_with_text("No hay missing que imputar")
    
    plot_df = plot_df.sort_values("missing_pct", ascending=True)
    
    # Color por estrategia
    def get_imp_color(strategy):
        if strategy == "none":
            return COLOR_PASS
        elif strategy in ("mean", "most_frequent"):
            return COLOR_WARN
        else:
            return COLOR_FAIL
    
    colors = [get_imp_color(s) for s in plot_df["recommended_imputation"]]
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.4)))
    ax.barh(plot_df["column"], plot_df["missing_pct"], color=colors)
    ax.set_xlabel("Porcentaje de valores faltantes")
    ax.set_title("Estrategias de imputación sugeridas")
    ax.legend(handles=[
        plt.Rectangle((0,0),1,1, color=COLOR_PASS, label="none"),
        plt.Rectangle((0,0),1,1, color=COLOR_WARN, label="mean/most_frequent"),
        plt.Rectangle((0,0),1,1, color=COLOR_FAIL, label="median/missing_label")
    ], loc="lower right")
    fig.tight_layout()
    return fig


def _build_transformation_scatter(trans_df: pd.DataFrame) -> Figure:
    """Construye scatter de skew vs outlier_pct."""
    if trans_df.empty:
        return _create_figure_with_text("Sin datos de transformación")
    
    plot_df = trans_df.dropna(subset=["skew", "outlier_pct"])
    if plot_df.empty:
        return _create_figure_with_text("Sin datos válidos para transformación")
    
    # Color por recomendación
    color_map = {
        "standard_scaler_if_needed": COLOR_PASS,
        "log_or_yeo_johnson + robust_scaler": COLOR_PRIMARY,
        "yeo_johnson + robust_scaler": COLOR_WARN,
        "robust_scaler": COLOR_WARN,
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for rec in plot_df["recommended_transformation"].unique():
        subset = plot_df[plot_df["recommended_transformation"] == rec]
        ax.scatter(subset["skew"], subset["outlier_pct"] * 100,
                  label=rec, color=color_map.get(rec, "#95a5a6"), s=80, alpha=0.7)
    
    # Anotar nombres de columnas
    for _, row in plot_df.iterrows():
        ax.annotate(row["column"], (row["skew"], row["outlier_pct"]*100),
                   fontsize=7, alpha=0.7)
    
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=1, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=-1, color="gray", linestyle="--", alpha=0.5)
    
    ax.set_xlabel("Skewness")
    ax.set_ylabel("Porcentaje de outliers")
    ax.set_title("Transformaciones sugeridas")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# =========================================================
# Builders para readiness.py
# =========================================================

def _build_readiness_barplot(readiness_df: pd.DataFrame) -> Figure:
    """Construye barra horizontal de checks con colores por status."""
    if readiness_df.empty:
        return _create_figure_with_text("Sin checks de readiness")
    
    plot_df = readiness_df.copy()
    
    color_map = {"PASS": COLOR_PASS, "WARN": COLOR_WARN, "FAIL": COLOR_FAIL}
    colors = [color_map.get(s, "#95a5a6") for s in plot_df["status"]]
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.4)))
    ax.barh(plot_df["check"], [1]*len(plot_df), color=colors)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Checklist de preparación ML")
    
    # Leyenda
    ax.legend(handles=[
        plt.Rectangle((0,0),1,1, color=COLOR_PASS, label="PASS"),
        plt.Rectangle((0,0),1,1, color=COLOR_WARN, label="WARN"),
        plt.Rectangle((0,0),1,1, color=COLOR_FAIL, label="FAIL")
    ], loc="lower right")
    
    # Añadir detalles
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(0.5, i, row["details"], va="center", ha="left", 
               fontsize=8, color="black", transform=ax.get_yaxis_transform())
    
    fig.tight_layout()
    return fig


# =========================================================
# Builders para fitness.py
# =========================================================

def _build_fitness_grouped_barplot(fitness_df: pd.DataFrame) -> Figure:
    """Construye barras agrupadas por categoría para fitness avanzado."""
    if fitness_df.empty:
        return _create_figure_with_text("Sin datos de fitness")
    
    # Excluir el veredicto final
    plot_df = fitness_df[fitness_df["check"] != "FINAL_VERDICT"].copy()
    if plot_df.empty:
        return _create_figure_with_text("Sin checks de fitness")
    
    color_map = {"PASS": COLOR_PASS, "WARN": COLOR_WARN, "FAIL": COLOR_FAIL}
    
    fig, ax = plt.subplots(figsize=(12, max(4, len(plot_df) * 0.4)))
    
    # Ordenar por categoría
    if "category" in plot_df.columns:
        plot_df = plot_df.sort_values("category")
    
    colors = [color_map.get(s, "#95a5a6") for s in plot_df["status"]]
    
    ax.barh(plot_df["check"], [1]*len(plot_df), color=colors)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Evaluación de aptitud ML avanzada")
    
    # Añadir veredicto final
    final_row = fitness_df[fitness_df["check"] == "FINAL_VERDICT"]
    if not final_row.empty:
        verdict = final_row.iloc[0]["status"]
        ax.text(0.5, -1, f"VEREDICTO FINAL: {verdict}", 
               fontsize=14, fontweight="bold",
               color=color_map.get(verdict, "#666"),
               transform=ax.get_yaxis_transform())
    
    ax.legend(handles=[
        plt.Rectangle((0,0),1,1, color=COLOR_PASS, label="PASS"),
        plt.Rectangle((0,0),1,1, color=COLOR_WARN, label="WARN"),
        plt.Rectangle((0,0),1,1, color=COLOR_FAIL, label="FAIL")
    ], loc="lower right")
    
    fig.tight_layout()
    return fig


# =========================================================
# Builders para ml_recommendations.py
# =========================================================

def _build_model_recommendations_barplot(models_df: pd.DataFrame) -> Figure:
    """Construye gráfico de recomendaciones de modelos."""
    if models_df.empty:
        return _create_figure_with_text("Sin recomendaciones de modelos")
    
    plot_df = models_df.sort_values("final_score", ascending=True)
    
    # Color por bucket
    bucket_colors = {
        "primary_candidate": COLOR_PRIMARY,
        "strong_baseline": COLOR_PASS,
        "interpretable_baseline": "#1abc9c",
        "conditional_candidate": COLOR_WARN,
        "avoid_for_now": COLOR_FAIL,
    }
    
    colors = [bucket_colors.get(b, "#95a5a6") for b in plot_df["recommended_as"]]
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.4)))
    ax.barh(plot_df["model"], plot_df["final_score"], color=colors)
    ax.set_xlabel("Puntuación final")
    ax.set_title("Recomendación de modelos por estructura de datos")
    ax.set_xlim(0, 10)
    
    ax.legend(handles=[
        plt.Rectangle((0,0),1,1, color=v, label=k) 
        for k, v in bucket_colors.items()
    ], loc="lower right", fontsize=8)
    
    fig.tight_layout()
    return fig