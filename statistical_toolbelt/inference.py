"""
Funciones de inferencia de tipos de variables y tarea.

Este módulo contiene las funciones para inferir automáticamente los tipos
de columnas (continuas, discretas, categóricas) y el tipo de tarea ML
(clasificación o regresión).
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def infer_column_types(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    id_cols: Optional[List[str]] = None,
    date_cols: Optional[List[str]] = None,
    high_card_threshold: int = 50
) -> Dict[str, List[str]]:
    """Infiere los tipos de columnas del dataset.
    
    Clasifica las columnas en: continuas numéricas, discretas numéricas,
    categóricas de baja cardinalidad, categóricas de alta cardinalidad,
    columnas ID y columnas de fecha.
    
    Args:
        df: DataFrame a analizar.
        target_col: Nombre de la columna objetivo (se excluye del análisis).
        id_cols: Lista de columnas que son IDs conocidos.
        date_cols: Lista de columnas que son fechas conocidas.
        high_card_threshold: Umbral de cardinalidad para considerar una
            categórica como de alta cardinalidad.
            
    Returns:
        Diccionario con listas de columnas por tipo:
        - continuous_numeric: columnas numéricas continuas
        - discrete_numeric: columnas numéricas discretas
        - low_cardinality_categorical: categóricas con <= high_card_threshold valores únicos
        - high_cardinality_categorical: categóricas con > high_card_threshold valores únicos
        - id_cols: columnas identificadas como ID
        - date_cols: columnas identificadas como fecha
    """
    id_cols = id_cols or []
    date_cols = date_cols or []
    
    # Identificar columnas numéricas y de objeto
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    # Clasificar numéricas en continuas vs discretas
    discrete_numeric = []
    continuous_numeric = []
    
    for col in numeric_cols:
        nunique = df[col].nunique(dropna=True)
        # Una numérica es discreta si tiene pocos valores únicos y es entera
        # o si tiene <= 10 valores únicos siendo entera
        if pd.api.types.is_integer_dtype(df[col]) and nunique <= max(20, int(len(df) * 0.05)):
            discrete_numeric.append(col)
        elif nunique <= 10 and pd.api.types.is_integer_dtype(df[col]):
            discrete_numeric.append(col)
        else:
            continuous_numeric.append(col)
    
    # Clasificar categóricas por cardinalidad
    low_cardinality_categorical = []
    high_cardinality_categorical = []
    
    for col in object_cols:
        nunique = df[col].nunique(dropna=True)
        if nunique <= high_card_threshold:
            low_cardinality_categorical.append(col)
        else:
            high_cardinality_categorical.append(col)
    
    # Excluir la columna objetivo de todos los grupos
    for col_list in [continuous_numeric, discrete_numeric, 
                     low_cardinality_categorical, high_cardinality_categorical]:
        if target_col in col_list:
            col_list.remove(target_col)
    
    return {
        "continuous_numeric": continuous_numeric,
        "discrete_numeric": discrete_numeric,
        "low_cardinality_categorical": low_cardinality_categorical,
        "high_cardinality_categorical": high_cardinality_categorical,
        "id_cols": id_cols,
        "date_cols": date_cols,
    }


def infer_task_type(
    df: pd.DataFrame,
    target_col: Optional[str],
    task_type: str = "auto"
) -> str:
    """Infiere el tipo de tarea de machine learning.
    
    Si task_type es "auto", deduce clasificación o regresión según
    el tipo de datos de la columna objetivo. Si la columna objetivo
    no es numérica, se asume clasificación. Si es numérica con pocos
    valores únicos (<= 10), también se considera clasificación.
    
    Args:
        df: DataFrame a analizar.
        target_col: Nombre de la columna objetivo.
        task_type: Fuerza un tipo específico ("auto", "classification", "regression").
        
    Returns:
        "classification", "regression" o "unsupervised_or_unknown" si no hay target.
    """
    if target_col is None:
        return "unsupervised_or_unknown"
    
    if task_type != "auto":
        return task_type
    
    s = df[target_col]
    
    if pd.api.types.is_numeric_dtype(s):
        nunique = s.nunique(dropna=True)
        # Si es entera y tiene <= 10 valores únicos, es clasificación binaria/multiclase
        if pd.api.types.is_integer_dtype(s) and nunique <= 10:
            return "classification"
        elif nunique <= 5:
            return "classification"
        else:
            return "regression"
    else:
        return "classification"