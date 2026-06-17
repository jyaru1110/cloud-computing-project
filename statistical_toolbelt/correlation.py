"""
Módulo de correlación y asociación.

Este módulo contiene funciones para calcular matrices de correlación
(Pearson, Spearman), asociación entre variables categóricas (Cramer's V),
asociación con el target, comparación de grupos y cálculo de VIF.
"""

from typing import List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import (
    chi2_contingency, spearmanr, pearsonr, pointbiserialr,
    f_oneway, levene, mannwhitneyu, kruskal
)

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .types import ToolbeltResult
from . import visualization


# Configuración global
RANDOM_STATE = 42

# Verificar disponibilidad de statsmodels
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calcula el coeficiente V de Cramer entre dos variables categóricas.
    
    El V de Cramer es una medida de asociación basada en la estadística
    chi-cuadrada, normalizada entre 0 y 1.
    
    Args:
        x: Primera variable categórica.
        y: Segunda variable categórica.
        
    Returns:
        Coeficiente V de Cramer (0 = sin asociación, 1 = asociación perfecta).
    """
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty:
        return np.nan
    
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    if n == 0:
        return np.nan
    
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denom = min((kcorr-1), (rcorr-1))
    if denom <= 0:
        return np.nan
    
    return np.sqrt(phi2corr / denom)


def pearson_correlation(df: pd.DataFrame, cols: List[str]) -> ToolbeltResult:
    """Matriz de correlación de Pearson.
    
    Calcula la matriz de correlación lineal entre las columnas especificadas.
    La figura muestra un heatmap con colores divergentes.
    
    Args:
        df: DataFrame a analizar.
        cols: Lista de columnas numéricas.
        
    Returns:
        ToolbeltResult con matriz de correlación y heatmap.
    """
    if len(cols) < 2:
        empty_df = pd.DataFrame()
        fig = visualization._create_figure_with_text("Se necesitan al menos 2 columnas")
        return ToolbeltResult(data=empty_df, figure=fig, title="Correlación Pearson")
    
    corr_df = df[cols].corr(method="pearson")
    fig = visualization._build_correlation_heatmap(corr_df, "Correlación Pearson")
    
    return ToolbeltResult(
        data=corr_df,
        figure=fig,
        title="Correlación Pearson"
    )


def spearman_correlation(df: pd.DataFrame, cols: List[str]) -> ToolbeltResult:
    """Matriz de correlación de Spearman.
    
    Calcula la matriz de correlación monotónica entre las columnas.
    Útil cuando la relación no es lineal. La figura muestra un heatmap.
    
    Args:
        df: DataFrame a analizar.
        cols: Lista de columnas numéricas.
        
    Returns:
        ToolbeltResult con matriz de correlación y heatmap.
    """
    if len(cols) < 2:
        empty_df = pd.DataFrame()
        fig = visualization._create_figure_with_text("Se necesitan al menos 2 columnas")
        return ToolbeltResult(data=empty_df, figure=fig, title="Correlación Spearman")
    
    corr_df = df[cols].corr(method="spearman")
    fig = visualization._build_correlation_heatmap(corr_df, "Correlación Spearman")
    
    return ToolbeltResult(
        data=corr_df,
        figure=fig,
        title="Correlación Spearman"
    )


def categorical_association_matrix(df: pd.DataFrame, cat_cols: List[str]) -> ToolbeltResult:
    """Matriz de asociación Cramer's V para variables categóricas.
    
    Calcula el V de Cramer entre todas las pares de columnas categóricas.
    La figura muestra un heatmap con escala 0-1.
    
    Args:
        df: DataFrame a analizar.
        cat_cols: Lista de columnas categóricas.
        
    Returns:
        ToolbeltResult con matriz de asociación y heatmap.
    """
    if len(cat_cols) < 2:
        empty_df = pd.DataFrame()
        fig = visualization._create_figure_with_text("Se necesitan al menos 2 columnas categóricas")
        return ToolbeltResult(data=empty_df, figure=fig, title="Asociación categórica (Cramer V)")
    
    mat = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
    for c1 in cat_cols:
        for c2 in cat_cols:
            if c1 == c2:
                mat.loc[c1, c2] = 1.0
            else:
                mat.loc[c1, c2] = cramers_v(df[c1], df[c2])
    
    fig = visualization._build_correlation_heatmap(mat, "Asociación categórica (Cramer V)")
    
    return ToolbeltResult(
        data=mat,
        figure=fig,
        title="Asociación categórica (Cramer V)"
    )


def target_association(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    continuous_cols: List[str],
    cat_cols: List[str]
) -> ToolbeltResult:
    """Análisis de asociación de cada feature con el target.
    
    Para regresión: correlación de Pearson para continuas, ANOVA para categóricas.
    Para clasificación: point-biserial para binaria + info mutua, chi-cuadrado para categóricas.
    
    Args:
        df: DataFrame a analizar.
        target_col: Nombre de la columna objetivo.
        task_type: "classification" o "regression".
        continuous_cols: Lista de columnas numéricas.
        cat_cols: Lista de columnas categóricas.
        
    Returns:
        ToolbeltResult con scores de asociación y figura de barras.
    """
    rows = []
    y = df[target_col]
    
    if task_type == "regression":
        # Pearson para continuas vs target numérico
        for col in continuous_cols:
            tmp = df[[col, target_col]].dropna()
            if len(tmp) > 3:
                r, p = pearsonr(tmp[col], tmp[target_col])
                rows.append({"feature": col, "test": "pearson", "score": r, "p_value": p})
        
        # ANOVA para categóricas vs target numérico
        for col in cat_cols:
            tmp = df[[col, target_col]].dropna()
            if tmp[col].nunique() >= 2:
                groups = [grp[target_col].values for _, grp in tmp.groupby(col)]
                if len(groups) >= 2:
                    try:
                        stat, p = f_oneway(*groups)
                        rows.append({"feature": col, "test": "anova_vs_target", "score": stat, "p_value": p})
                    except Exception:
                        pass
    
    elif task_type == "classification":
        y_num = pd.factorize(y)[0]
        
        # Point-biserial para binaria, mutual info para multiclase
        for col in continuous_cols:
            tmp = df[[col, target_col]].dropna()
            if tmp[target_col].nunique() == 2:
                y_bin = pd.factorize(tmp[target_col])[0]
                r, p = pointbiserialr(y_bin, tmp[col])
                rows.append({"feature": col, "test": "point_biserial", "score": r, "p_value": p})
            else:
                if SKLEARN_AVAILABLE:
                    try:
                        mi = mutual_info_classif(
                            tmp[[col]], pd.factorize(tmp[target_col])[0],
                            discrete_features=False, random_state=RANDOM_STATE
                        )[0]
                        rows.append({"feature": col, "test": "mutual_info", "score": mi, "p_value": np.nan})
                    except Exception:
                        pass
                else:
                    pass
        
        # Chi-cuadrado para categóricas
        for col in cat_cols:
            tmp = df[[col, target_col]].dropna()
            if tmp[col].nunique() >= 2 and tmp[target_col].nunique() >= 2:
                ct = pd.crosstab(tmp[col], tmp[target_col])
                stat, p, _, _ = chi2_contingency(ct)
                rows.append({"feature": col, "test": "chi2", "score": stat, "p_value": p})
    
    if not rows:
        ta_df = pd.DataFrame({"feature": [], "test": [], "score": [], "p_value": []})
    else:
        ta_df = pd.DataFrame(rows).sort_values(["p_value", "score"], ascending=[True, False])
    
    fig = visualization._build_association_barplot(ta_df)
    
    return ToolbeltResult(
        data=ta_df,
        figure=fig,
        title="Asociación con variable objetivo"
    )


def compare_groups_by_target(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str]
) -> ToolbeltResult:
    """Comparación de grupos definidos por el target.
    
    Para clasificación binaria: t-test y Mann-Whitney U.
    Para clasificación multiclase: ANOVA y Kruskal-Wallis.
    También aplica prueba de Levene para homogeneidad de varianzas.
    
    Args:
        df: DataFrame a analizar.
        target_col: Nombre de la columna objetivo.
        feature_cols: Lista de features a comparar.
        
    Returns:
        ToolbeltResult con estadísticas de comparación y boxplots.
    """
    rows = []
    y = df[target_col]
    n_classes = y.nunique(dropna=True)
    
    if n_classes < 2:
        empty_df = pd.DataFrame()
        fig = visualization._create_figure_with_text("El target no tiene múltiples clases")
        return ToolbeltResult(data=empty_df, figure=fig, title="Comparación de grupos por target")
    
    for col in feature_cols:
        tmp = df[[col, target_col]].dropna()
        groups = [grp[col].values for _, grp in tmp.groupby(target_col)]
        
        if len(groups) == 2:
            # Binario: t-test y Mann-Whitney
            try:
                lev_stat, lev_p = levene(*groups)
            except Exception:
                lev_p = np.nan
            
            try:
                t_stat, t_p = stats.ttest_ind(
                    *groups, 
                    equal_var=(lev_p > 0.05 if not pd.isna(lev_p) else False)
                )
            except Exception:
                t_p = np.nan
                t_stat = np.nan
            
            try:
                mw_stat, mw_p = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            except Exception:
                mw_stat, mw_p = np.nan, np.nan
            
            rows.append({
                "feature": col,
                "groups": 2,
                "levene_p": lev_p,
                "t_test_stat": t_stat,
                "t_test_p": t_p,
                "mannwhitney_p": mw_p
            })
        elif len(groups) > 2:
            # Multiclase: ANOVA y Kruskal
            try:
                lev_stat, lev_p = levene(*groups)
            except Exception:
                lev_p = np.nan
            
            try:
                a_stat, a_p = f_oneway(*groups)
            except Exception:
                a_stat, a_p = np.nan, np.nan
            
            try:
                k_stat, k_p = kruskal(*groups)
            except Exception:
                k_stat, k_p = np.nan, np.nan
            
            rows.append({
                "feature": col,
                "groups": len(groups),
                "levene_p": lev_p,
                "anova_stat": a_stat,
                "anova_p": a_p,
                "kruskal_p": k_p
            })
    
    comp_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    
    # Figura: boxplots para las top features con mayor diferencia
    if not comp_df.empty and len(feature_cols) > 0:
        # Seleccionar algunas features para graficar
        plot_cols = feature_cols[:min(6, len(feature_cols))]
        
        # Crear boxplots
        n_cols = len(plot_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]
        
        for ax, col in zip(axes, plot_cols):
            data_to_plot = [df[df[target_col] == cls][col].dropna() 
                           for cls in df[target_col].unique() if pd.notna(cls)]
            if data_to_plot:
                ax.boxplot(data_to_plot)
                ax.set_xticklabels([str(c) for c in df[target_col].unique() if pd.notna(c)])
                ax.set_title(col)
                ax.set_ylabel("Valor")
        
        fig.suptitle("Comparación de grupos por target")
        fig.tight_layout()
    else:
        fig = visualization._create_figure_with_text("Sin datos para comparar")
    
    return ToolbeltResult(
        data=comp_df,
        figure=fig,
        title="Comparación de grupos por target"
    )


def compute_vif(df: pd.DataFrame, cols: List[str]) -> ToolbeltResult:
    """Cálculo del Factor de Inflación de Varianza (VIF).
    
    El VIF mide la multicolinealidad: valores > 10 indican problema.
    Requiere statsmodels.
    
    Args:
        df: DataFrame a analizar.
        cols: Lista de columnas numéricas.
        
    Returns:
        ToolbeltResult con VIF por feature y figura de barras.
    """
    if not STATSMODELS_AVAILABLE:
        warn_df = pd.DataFrame({"warning": ["statsmodels no está disponible para calcular VIF"]})
        fig = visualization._create_figure_with_text("statsmodels no disponible")
        return ToolbeltResult(data=warn_df, figure=fig, title="Factor de Inflación de Varianza (VIF)")
    
    tmp = df[cols].dropna()
    if tmp.empty or len(cols) < 2:
        empty_df = pd.DataFrame({"feature": [], "VIF": []})
        fig = visualization._create_figure_with_text("Sin suficientes datos para VIF")
        return ToolbeltResult(data=empty_df, figure=fig, title="Factor de Inflación de Varianza (VIF)")
    
    X = sm.add_constant(tmp)
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        rows.append({"feature": col, "VIF": variance_inflation_factor(X.values, i)})
    
    if not rows:
        vif_df = pd.DataFrame({"feature": [], "VIF": []})
    else:
        vif_df = pd.DataFrame(rows).sort_values("VIF", ascending=False)
    
    fig = visualization._build_vif_barplot(vif_df)
    
    return ToolbeltResult(
        data=vif_df,
        figure=fig,
        title="Factor de Inflación de Varianza (VIF)"
    )


# Import matplotlib.pyplot para la función compare_groups_by_target
import matplotlib.pyplot as plt