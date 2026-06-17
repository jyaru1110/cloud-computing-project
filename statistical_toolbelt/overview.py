"""
Módulo de diagnóstico general del dataset.

Este módulo contiene funciones para obtener una vista general del dataset:
-overview, reporte de missing, reporte de duplicados y reporte de cardinalidad.
Cada función devuelve un ToolbeltResult con datos y figura asociada.
"""

from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .types import ToolbeltResult
from . import visualization


def dataset_overview(df: pd.DataFrame) -> ToolbeltResult:
    """Genera una vista general del dataset.
    
    Proporciona información básica sobre cada columna: nombre, tipo de dato,
    número de valores únicos, cantidad y porcentaje de valores faltantes,
    y valores de ejemplo.
    
    Args:
        df: DataFrame a analizar.
        
    Returns:
        ToolbeltResult con DataFrame de overview y figura de missing %.
    """
    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "n_unique": df[col].nunique(dropna=True),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": round(df[col].isna().mean(), 4),
            "sample_values": ", ".join(map(str, df[col].dropna().astype(str).head(3).tolist()))
        })
    
    if not rows:
        overview_df = pd.DataFrame({"column": [], "dtype": [], "n_unique": [], "missing_count": [], "missing_pct": [], "sample_values": []})
    else:
        overview_df = pd.DataFrame(rows).sort_values(
            ["missing_pct", "n_unique"], ascending=[False, False]
        )
    
    # Figura: barra horizontal de missing % solo para columnas con missing
    missing_only = overview_df[overview_df["missing_pct"] > 0].copy()
    if not missing_only.empty:
        plot_df = missing_only.sort_values("missing_pct", ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.4)))
        ax.barh(plot_df["column"], plot_df["missing_pct"], color="#3498db")
        ax.set_xlabel("Porcentaje de valores faltantes")
        ax.set_title("Vista general del dataset - Missing %")
        for i, pct in enumerate(plot_df["missing_pct"]):
            ax.text(pct + 0.3, i, f"{pct*100:.1f}%", va="center", fontsize=9)
        ax.set_xlim(0, max(plot_df["missing_pct"]) * 1.2)
        fig.tight_layout()
    else:
        fig = visualization._create_figure_with_text("Sin valores faltantes")
    
    return ToolbeltResult(
        data=overview_df,
        figure=fig,
        title="Vista general del dataset"
    )


def missing_report(df: pd.DataFrame) -> ToolbeltResult:
    """Reporte detallado de valores faltantes por columna.
    
    Muestra la cantidad y porcentaje de valores nulos por cada columna,
    ordenado de mayor a menor porcentaje. La figura colorea por severidad:
    verde < 5%, amarillo 5-20%, rojo > 20%.
    
    Args:
        df: DataFrame a analizar.
        
    Returns:
        ToolbeltResult con DataFrame de missing y figura de barras por severidad.
    """
    missing_df = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_pct": (df.isna().mean().values * 100).round(2)
    }).sort_values("missing_pct", ascending=False)
    
    fig = visualization._build_missing_barplot(missing_df)
    
    return ToolbeltResult(
        data=missing_df,
        figure=fig,
        title="Reporte de valores faltantes"
    )


def duplicate_report(df: pd.DataFrame) -> ToolbeltResult:
    """Reporte de filas duplicadas en el dataset.
    
    Muestra el número total de filas, columnas, filas duplicadas y el
    porcentaje de duplicados respecto al total.
    
    Args:
        df: DataFrame a analizar.
        
    Returns:
        ToolbeltResult con DataFrame de duplicados y figura de barras.
    """
    total_dup = int(df.duplicated().sum())
    dup_df = pd.DataFrame([{
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "duplicate_rows": total_dup,
        "duplicate_pct": round(total_dup / len(df), 4) if len(df) else 0
    }])
    
    fig = visualization._build_duplicate_barplot(dup_df)
    
    return ToolbeltResult(
        data=dup_df,
        figure=fig,
        title="Reporte de duplicados"
    )


def cardinality_report(df: pd.DataFrame, cat_cols: List[str]) -> ToolbeltResult:
    """Reporte de cardinalidad de variables categóricas.
    
    Muestra el número de valores únicos por variable categórica y las
    frecuencias de los valores más comunes. Útil para identificar
    variables de alta cardinalidad que pueden requerir tratamiento especial.
    
    Args:
        df: DataFrame a analizar.
        cat_cols: Lista de columnas categóricas a analizar.
        
    Returns:
        ToolbeltResult con DataFrame de cardinalidad y figura de barras.
    """
    if not cat_cols:
        empty_df = pd.DataFrame({"column": [], "n_unique": [], "top_5_freq": []})
        fig = visualization._create_figure_with_text("Sin variables categóricas")
        return ToolbeltResult(
            data=empty_df,
            figure=fig,
            title="Cardinalidad de variables categóricas"
        )
    
    rows = []
    for col in cat_cols:
        rows.append({
            "column": col,
            "n_unique": int(df[col].nunique(dropna=True)),
            "top_5_freq": str(df[col].value_counts(dropna=False).head(5).to_dict())
        })
    
    if not rows:
        card_df = pd.DataFrame({"column": [], "n_unique": [], "top_5_freq": []})
    else:
        card_df = pd.DataFrame(rows).sort_values("n_unique", ascending=False)
    
    fig = visualization._build_cardinality_barplot(card_df)
    
    return ToolbeltResult(
        data=card_df,
        figure=fig,
        title="Cardinalidad de variables categóricas"
    )