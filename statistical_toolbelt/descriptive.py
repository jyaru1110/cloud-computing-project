"""
Módulo de estadística descriptiva.

Este módulo contiene funciones para resumir variables continuas y discretas,
y para detectar outliers usando el método IQR.
"""

from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .types import ToolbeltResult
from . import visualization


def continuous_summary(df: pd.DataFrame, cols: List[str]) -> ToolbeltResult:
    """Resumen estadístico de variables continuas.
    
    Calcula estadísticas descriptivas (describe) más skewness y kurtosis
    para las columnas especificadas. La figura muestra boxplots normalizados
    de las variables para comparar distribuciones.
    
    Args:
        df: DataFrame a analizar.
        cols: Lista de columnas continuas a resumir.
        
    Returns:
        ToolbeltResult con DataFrame de estadísticas y figura de boxplots.
    """
    if not cols:
        empty_df = pd.DataFrame()
        fig = visualization._create_figure_with_text("Sin variables continuas")
        return ToolbeltResult(
            data=empty_df,
            figure=fig,
            title="Resumen de variables continuas"
        )
    
    desc = df[cols].describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
    desc["skew"] = df[cols].skew(numeric_only=True)
    desc["kurtosis"] = df[cols].kurtosis(numeric_only=True)
    
    summary_df = desc.reset_index().rename(columns={"index": "column"})
    
    fig = visualization._build_continuous_boxplot(df, cols)
    
    return ToolbeltResult(
        data=summary_df,
        figure=fig,
        title="Resumen de variables continuas"
    )


def discrete_summary(df: pd.DataFrame, cols: List[str], top_n: int = 10) -> ToolbeltResult:
    """Resumen de distribución de variables discretas.
    
    Genera un único DataFrame con columnas [variable, value, count, pct]
    donde cada fila representa un valor único de una variable discreta.
    La figura muestra un grid de barras de frecuencia.
    
    Args:
        df: DataFrame a analizar.
        cols: Lista de columnas discretas a resumir.
        top_n: Número máximo de valores por variable a incluir.
        
    Returns:
        ToolbeltResult con DataFrame de frecuencias y figura de grid de barras.
    """
    if not cols:
        empty_df = pd.DataFrame({"variable": [], "value": [], "count": [], "pct": []})
        fig = visualization._create_figure_with_text("Sin variables discretas")
        return ToolbeltResult(
            data=empty_df,
            figure=fig,
            title="Distribución de variables discretas"
        )
    
    all_rows = []
    for col in cols:
        vc = df[col].value_counts(dropna=False).head(top_n)
        for val, count in vc.items():
            all_rows.append({
                "variable": col,
                "value": val,
                "count": int(count),
                "pct": round(count / len(df), 4)
            })
    
    disc_df = pd.DataFrame(all_rows)
    
    fig = visualization._build_discrete_freq_grid(disc_df)
    
    return ToolbeltResult(
        data=disc_df,
        figure=fig,
        title="Distribución de variables discretas"
    )


def iqr_outlier_report(df: pd.DataFrame, cols: List[str]) -> ToolbeltResult:
    """Reporte de outliers usando el método IQR.
    
    Calcula los límites inferior y superior usando 1.5*IQR y cuenta
    los valores que caen fuera de estos límites. La figura muestra
    boxplots con outliers marcados en rojo.
    
    Args:
        df: DataFrame a analizar.
        cols: Lista de columnas a verificar para outliers.
        
    Returns:
        ToolbeltResult con DataFrame de outliers y figura de boxplots.
    """
    if not cols:
        empty_df = pd.DataFrame(
            {"column": [], "q1": [], "q3": [], "iqr": [], 
             "lower_bound": [], "upper_bound": [], "outlier_count": [], "outlier_pct": []}
        )
        fig = visualization._create_figure_with_text("Sin variables para analizar")
        return ToolbeltResult(
            data=empty_df,
            figure=fig,
            title="Reporte de outliers (IQR)"
        )
    
    rows = []
    for col in cols:
        s = df[col].dropna()
        if len(s) < 5:
            continue
        
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        outliers = ((df[col] < lo) | (df[col] > hi)).sum()
        
        rows.append({
            "column": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lo,
            "upper_bound": hi,
            "outlier_count": int(outliers),
            "outlier_pct": round(outliers / len(df), 4)
        })
    
    if not rows:
        outlier_df = pd.DataFrame(
            {"column": [], "q1": [], "q3": [], "iqr": [],
             "lower_bound": [], "upper_bound": [], "outlier_count": [], "outlier_pct": []}
        )
    else:
        outlier_df = pd.DataFrame(rows).sort_values("outlier_pct", ascending=False)
    
    fig = visualization._build_outlier_boxplot(outlier_df, df)
    
    return ToolbeltResult(
        data=outlier_df,
        figure=fig,
        title="Reporte de outliers (IQR)"
    )