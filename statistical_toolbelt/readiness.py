"""
Módulo de checklist de preparación ML.

Este módulo contiene la función que verifica si el dataset cumple
los criterios mínimos necesarios para machine learning.
"""

from typing import List, Optional

import pandas as pd
from matplotlib.figure import Figure

from .types import ToolbeltResult
from . import visualization


def ml_readiness_check(
    df: pd.DataFrame,
    target_col: Optional[str],
    task_type: str,
    continuous_cols: List[str],
    discrete_cols: List[str],
    cat_cols: List[str]
) -> ToolbeltResult:
    """Checklist de preparación básica para machine learning.
    
    Verifica los siguientes criterios:
    - missing_global_below_20pct: promedio de missing global < 20%
    - duplicates_below_5pct: duplicados < 5%
    - no_columns_above_40pct_missing: ninguna columna > 40% missing
    - no_constant_columns: sin columnas constantes
    - target_available: el target está definido
    - class_balance_reasonable: para clasificación, clase mayoritaria < 90%
    - target_variance_nonzero: para regresión, el target tiene varianza
    - has_predictive_features: hay al menos una feature usable
    
    Args:
        df: DataFrame a analizar.
        target_col: Nombre de la columna objetivo.
        task_type: "classification" o "regression".
        continuous_cols: Lista de columnas continuas.
        discrete_cols: Lista de columnas discretas.
        cat_cols: Lista de columnas categóricas.
        
    Returns:
        ToolbeltResult con DataFrame de checks y figura de barras de estado.
    """
    checks = []
    
    # Missing global
    total_missing = df.isna().mean().mean()
    checks.append({
        "check": "missing_global_below_20pct",
        "status": "PASS" if total_missing < 0.20 else "WARN",
        "details": f"Missing global promedio: {total_missing:.2%}"
    })
    
    # Duplicados
    duplicated_pct = df.duplicated().mean()
    checks.append({
        "check": "duplicates_below_5pct",
        "status": "PASS" if duplicated_pct < 0.05 else "WARN",
        "details": f"Duplicados: {duplicated_pct:.2%}"
    })
    
    # Columnas con mucho missing
    high_missing_cols = [c for c in df.columns if df[c].isna().mean() > 0.40]
    checks.append({
        "check": "no_columns_above_40pct_missing",
        "status": "PASS" if len(high_missing_cols) == 0 else "WARN",
        "details": f"Columnas: {high_missing_cols[:10]}"
    })
    
    # Columnas constantes
    near_constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    checks.append({
        "check": "no_constant_columns",
        "status": "PASS" if len(near_constant_cols) == 0 else "WARN",
        "details": f"Constantes: {near_constant_cols[:10]}"
    })
    
    # Target disponible
    if target_col is not None:
        y = df[target_col]
        checks.append({
            "check": "target_available",
            "status": "PASS",
            "details": f"Task inferred: {task_type}"
        })
        
        # Balance de clases (solo clasificación)
        if task_type == "classification":
            class_dist = y.value_counts(normalize=True, dropna=True)
            max_class = class_dist.max() if len(class_dist) else 1.0
            checks.append({
                "check": "class_balance_reasonable",
                "status": "PASS" if max_class < 0.90 else "WARN",
                "details": f"Mayor clase: {max_class:.2%}"
            })
        
        # Varianza del target (solo regresión)
        if task_type == "regression":
            non_null = y.dropna()
            checks.append({
                "check": "target_variance_nonzero",
                "status": "PASS" if non_null.nunique() > 1 else "FAIL",
                "details": f"Unique target values: {non_null.nunique()}"
            })
    
    # Features disponibles
    enough_numeric = len(continuous_cols) + len(discrete_cols) > 0
    checks.append({
        "check": "has_predictive_features",
        "status": "PASS" if enough_numeric or len(cat_cols) > 0 else "FAIL",
        "details": f"continuous={len(continuous_cols)}, discrete={len(discrete_cols)}, categorical={len(cat_cols)}"
    })
    
    readiness_df = pd.DataFrame(checks)
    
    fig = visualization._build_readiness_barplot(readiness_df)
    
    return ToolbeltResult(
        data=readiness_df,
        figure=fig,
        title="Checklist de preparación ML"
    )