"""
Módulo de sugerencias de preprocesamiento.

Este módulo contiene funciones para sugerir estrategias de imputación
de valores faltantes y transformaciones de variables según sus características.
"""

from typing import List

import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from .types import ToolbeltResult
from . import visualization


def suggest_imputation(
    df: pd.DataFrame,
    continuous_cols: List[str],
    discrete_cols: List[str],
    cat_cols: List[str]
) -> ToolbeltResult:
    """Sugiere estrategias de imputación para cada columna con missing.
    
    Analiza el porcentaje de missing y la distribución (skew) para
    decidir la mejor estrategia:
    - Continuas sin skew: mean
    - Continuas con skew: median
    - Discretas: most_frequent
    - Categóricas: most_frequent o missing_label
    
    Args:
        df: DataFrame a analizar.
        continuous_cols: Lista de columnas continuas.
        discrete_cols: Lista de columnas discretas.
        cat_cols: Lista de columnas categóricas.
        
    Returns:
        ToolbeltResult con DataFrame de sugerencias y figura de barras.
    """
    rows = []
    
    # Continuas
    for col in continuous_cols:
        miss = df[col].isna().mean()
        skew_val = df[col].dropna().skew() if df[col].dropna().shape[0] > 2 else np.nan
        
        if miss == 0:
            strategy = "none"
        elif abs(skew_val) > 1:
            strategy = "median"
        else:
            strategy = "mean"
        
        rows.append({
            "column": col,
            "type": "continuous",
            "missing_pct": round(miss, 4),
            "recommended_imputation": strategy
        })
    
    # Discretas
    for col in discrete_cols:
        miss = df[col].isna().mean()
        strategy = "most_frequent" if miss > 0 else "none"
        rows.append({
            "column": col,
            "type": "discrete",
            "missing_pct": round(miss, 4),
            "recommended_imputation": strategy
        })
    
    # Categóricas
    for col in cat_cols:
        miss = df[col].isna().mean()
        strategy = "most_frequent_or_missing_label" if miss > 0 else "none"
        rows.append({
            "column": col,
            "type": "categorical",
            "missing_pct": round(miss, 4),
            "recommended_imputation": strategy
        })
    
    imp_df = pd.DataFrame(rows)
    
    fig = visualization._build_imputation_barplot(imp_df)
    
    return ToolbeltResult(
        data=imp_df,
        figure=fig,
        title="Estrategias de imputación sugeridas"
    )


def suggest_transformations(df: pd.DataFrame, continuous_cols: List[str]) -> ToolbeltResult:
    """Sugiere transformaciones para variables continuas.
    
    Analiza skewness y porcentaje de outliers para recomendar:
    - standard_scaler_if_needed: si skew bajo y pocos outliers
    - log_or_yeo_johnson + robust_scaler: si skew positivo y sin valores no positivos
    - yeo_johnson + robust_scaler: si skew alto
    - robust_scaler: si hay outliers significativos
    - standard_scaler: caso por defecto
    
    Args:
        df: DataFrame a analizar.
        continuous_cols: Lista de columnas continuas.
        
    Returns:
        ToolbeltResult con DataFrame de sugerencias y figura scatter.
    """
    if not continuous_cols:
        empty_df = pd.DataFrame(
            {"column": [], "skew": [], "outlier_pct": [], "recommended_transformation": []}
        )
        fig = visualization._create_figure_with_text("Sin variables continuas")
        return ToolbeltResult(
            data=empty_df,
            figure=fig,
            title="Transformaciones sugeridas"
        )
    
    rows = []
    for col in continuous_cols:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        
        skew_val = s.skew()
        has_non_positive = (s <= 0).any()
        
        # Calcular outlier %
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        outlier_pct = float(((s < lo) | (s > hi)).mean())
        
        # Decidir estrategia
        if abs(skew_val) <= 0.75 and outlier_pct < 0.03:
            rec = "standard_scaler_if_needed"
        elif not has_non_positive and skew_val > 1:
            rec = "log_or_yeo_johnson + robust_scaler"
        elif abs(skew_val) > 1:
            rec = "yeo_johnson + robust_scaler"
        elif outlier_pct >= 0.03:
            rec = "robust_scaler"
        else:
            rec = "standard_scaler"
        
        rows.append({
            "column": col,
            "skew": round(skew_val, 4),
            "outlier_pct": round(outlier_pct, 4),
            "recommended_transformation": rec
        })
    
    if not rows:
        trans_df = pd.DataFrame(
            {"column": [], "skew": [], "outlier_pct": [], "recommended_transformation": []}
        )
    else:
        trans_df = pd.DataFrame(rows).sort_values(
            ["outlier_pct", "skew"], ascending=[False, False]
        )
    
    fig = visualization._build_transformation_scatter(trans_df)
    
    return ToolbeltResult(
        data=trans_df,
        figure=fig,
        title="Transformaciones sugeridas"
    )