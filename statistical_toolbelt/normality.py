"""
Módulo de pruebas de normalidad.

Este módulo contiene funciones para evaluar la distribución de las
variables continuas usando pruebas estadísticas (Shapiro-Wilk y
D'Agostino-Pearson) y visualizaciones (histograma y Q-Q plot).
"""

from typing import List

import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from scipy import stats
from scipy.stats import shapiro, normaltest

from .types import ToolbeltResult
from . import visualization


def normality_tests(
    df: pd.DataFrame, 
    cols: List[str], 
    max_shapiro_n: int = 5000,
    random_state: int = 42
) -> ToolbeltResult:
    """Pruebas de normalidad para variables continuas.
    
    Aplica las pruebas de Shapiro-Wilk y D'Agostino-Pearson a cada
    columna especificada. Devuelve estadísticas descriptivas (skew,
    kurtosis) y los p-valores de ambas pruebas. La figura muestra
    histogramas y Q-Q plots para las primeras 6 columnas, resaltando
    en rojo aquellas que no pasan la prueba de Shapiro al 5%.
    
    Args:
        df: DataFrame a analizar.
        cols: Lista de columnas continuas a probar.
        max_shapiro_n: Máximo de observaciones para Shapiro (si hay más,
            se muestrea).
        random_state: Semilla para reproducibilidad del muestreo.
        
    Returns:
        ToolbeltResult con DataFrame de pruebas y figura de histogramas + QQ-plots.
    """
    if not cols:
        empty_df = pd.DataFrame(
            {"column": [], "n": [], "skew": [], "kurtosis": [],
             "shapiro_p": [], "dagostino_p": [],
             "normal_by_shapiro_0.05": [], "normal_by_dagostino_0.05": []}
        )
        fig = visualization._create_figure_with_text("Sin variables para probar")
        return ToolbeltResult(
            data=empty_df,
            figure=fig,
            title="Pruebas de normalidad"
        )
    
    rows = []
    for col in cols:
        s = df[col].dropna()
        if len(s) < 8:
            continue
        
        # Shapiro-Wilk (con muestreo si es necesario)
        if len(s) <= max_shapiro_n:
            sh_stat, sh_p = shapiro(s.sample(len(s), random_state=random_state))
        else:
            sample = s.sample(max_shapiro_n, random_state=random_state)
            sh_stat, sh_p = shapiro(sample)
        
        # D'Agostino-Pearson
        try:
            sample_dag = s if len(s) <= 10000 else s.sample(10000, random_state=random_state)
            dag_stat, dag_p = normaltest(sample_dag)
        except Exception:
            dag_stat, dag_p = np.nan, np.nan
        
        rows.append({
            "column": col,
            "n": len(s),
            "skew": float(s.skew()),
            "kurtosis": float(s.kurtosis()),
            "shapiro_p": sh_p,
            "dagostino_p": dag_p,
            "normal_by_shapiro_0.05": bool(sh_p > 0.05),
            "normal_by_dagostino_0.05": bool(dag_p > 0.05) if not pd.isna(dag_p) else np.nan
        })
    
    if not rows:
        norm_df = pd.DataFrame(
            {"column": [], "n": [], "skew": [], "kurtosis": [],
             "shapiro_p": [], "dagostino_p": [],
             "normal_by_shapiro_0.05": [], "normal_by_dagostino_0.05": []}
        )
    else:
        norm_df = pd.DataFrame(rows).sort_values(
            ["shapiro_p", "dagostino_p"], ascending=[True, True]
        )
    
    fig = visualization._build_normality_grid(df, cols, norm_df)
    
    return ToolbeltResult(
        data=norm_df,
        figure=fig,
        title="Pruebas de normalidad"
    )