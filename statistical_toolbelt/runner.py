"""
Runner de conveniencia para el Statistical Toolbelt.

Este módulo contiene la función run_full_diagnostic que ejecuta
todos los diagnósticos en orden y devuelve un diccionario de
ToolbeltResults.
"""

from typing import Dict

import pandas as pd

from .config import ToolbeltConfig
from .types import ToolbeltResult
from .inference import infer_column_types, infer_task_type
from .overview import dataset_overview, missing_report, duplicate_report, cardinality_report
from .descriptive import continuous_summary, discrete_summary, iqr_outlier_report
from .normality import normality_tests
from .correlation import (
    pearson_correlation, spearman_correlation, categorical_association_matrix,
    target_association, compare_groups_by_target, compute_vif
)
from .preprocessing import suggest_imputation, suggest_transformations
from .readiness import ml_readiness_check
from .fitness import evaluate_dataset_ml_fitness_advanced, recommend_next_actions_from_fitness
from .ml_recommendations import recommend_models_by_data_structure


def run_full_diagnostic(
    df: pd.DataFrame,
    config: ToolbeltConfig,
) -> Dict[str, ToolbeltResult]:
    """Ejecuta el diagnóstico completo del dataset.
    
    Este runner es un atajo de conveniencia que ejecuta todas las funciones
    de diagnóstico en orden y devuelve un diccionario con los resultados.
    Cada clave es el nombre del diagnóstico y cada valor es un ToolbeltResult
    que puede renderizarse en Jupyter o imprimirse en consola.
    
    El flujo interno es:
    1. Inferencia de tipos de columnas y tipo de tarea
    2. Diagnósticos generales (overview, missing, duplicados, cardinalidad)
    3. Estadística descriptiva (continuas, discretas, outliers)
    4. Pruebas de normalidad
    5. Correlaciones y asociación
    6. Sugerencias de preprocesamiento
    7. Checklist de readiness
    8. Fitness avanzado y acciones recomendadas
    9. Recomendaciones de modelos
    
    Args:
        df: DataFrame a analizar.
        config: Configuración con parámetros del análisis.
        
    Returns:
        Diccionario con nombres de diagnóstico como claves y ToolbeltResults
        como valores.
    """
    results = {}
    
    # === 1) Inferencia de tipos ===
    inferred = infer_column_types(
        df=df,
        target_col=config.target_col,
        id_cols=config.id_cols,
        date_cols=config.date_cols,
        high_card_threshold=config.high_cardinality_threshold
    )
    
    continuous_cols = inferred["continuous_numeric"]
    discrete_cols = inferred["discrete_numeric"]
    low_card_cat_cols = inferred["low_cardinality_categorical"]
    high_card_cat_cols = inferred["high_cardinality_categorical"]
    cat_cols = low_card_cat_cols + high_card_cat_cols
    
    # Inferir tipo de tarea
    inferred_task = infer_task_type(df, config.target_col, config.task_type)
    
    # Almacenar resultados intermedios para fitness avanzado
    intermediate_results = {
        "task_type": inferred_task,
        "column_types": inferred,
    }
    
    # === 2) Diagnósticos generales ===
    results["dataset_overview"] = dataset_overview(df)
    results["missing_report"] = missing_report(df)
    results["duplicate_report"] = duplicate_report(df)
    
    if cat_cols:
        results["cardinality_report"] = cardinality_report(df, cat_cols)
    else:
        results["cardinality_report"] = ToolbeltResult(
            data=pd.DataFrame({"column": [], "n_unique": [], "top_5_freq": []}),
            figure=None,
            title="Cardinalidad de variables categóricas"
        )
    
    # === 3) Estadística descriptiva ===
    if continuous_cols:
        results["continuous_summary"] = continuous_summary(df, continuous_cols)
    else:
        results["continuous_summary"] = ToolbeltResult(
            data=pd.DataFrame(),
            figure=None,
            title="Resumen de variables continuas"
        )
    
    if discrete_cols:
        results["discrete_summary"] = discrete_summary(df, discrete_cols)
    else:
        results["discrete_summary"] = ToolbeltResult(
            data=pd.DataFrame({"variable": [], "value": [], "count": [], "pct": []}),
            figure=None,
            title="Distribución de variables discretas"
        )
    
    if continuous_cols:
        results["outlier_report"] = iqr_outlier_report(df, continuous_cols)
    else:
        results["outlier_report"] = ToolbeltResult(
            data=pd.DataFrame(),
            figure=None,
            title="Reporte de outliers (IQR)"
        )
    
    # === 4) Normalidad ===
    if continuous_cols:
        results["normality"] = normality_tests(df, continuous_cols)
    else:
        results["normality"] = ToolbeltResult(
            data=pd.DataFrame(),
            figure=None,
            title="Pruebas de normalidad"
        )
    
    # === 5) Correlaciones ===
    if len(continuous_cols) >= 2:
        results["pearson_correlation"] = pearson_correlation(df, continuous_cols)
        results["spearman_correlation"] = spearman_correlation(df, continuous_cols)
        intermediate_results["correlations"] = {
            "pearson": results["pearson_correlation"].data,
            "spearman": results["spearman_correlation"].data
        }
    else:
        results["pearson_correlation"] = ToolbeltResult(
            data=pd.DataFrame(),
            figure=None,
            title="Correlación Pearson"
        )
        results["spearman_correlation"] = ToolbeltResult(
            data=pd.DataFrame(),
            figure=None,
            title="Correlación Spearman"
        )
        intermediate_results["correlations"] = {"pearson": pd.DataFrame(), "spearman": pd.DataFrame()}
    
    if len(cat_cols) >= 2:
        results["categorical_association"] = categorical_association_matrix(df, cat_cols)
    else:
        results["categorical_association"] = ToolbeltResult(
            data=pd.DataFrame(),
            figure=None,
            title="Asociación categórica (Cramer V)"
        )
    
    if config.target_col and len(continuous_cols) >= 2:
        results["vif"] = compute_vif(df, continuous_cols[:20])
    else:
        results["vif"] = ToolbeltResult(
            data=pd.DataFrame(),
            figure=None,
            title="Factor de Inflación de Varianza (VIF)"
        )
    
    # === 6) Asociación con target ===
    if config.target_col:
        results["target_association"] = target_association(
            df=df,
            target_col=config.target_col,
            task_type=inferred_task,
            continuous_cols=continuous_cols,
            cat_cols=cat_cols
        )
        
        # Comparación de grupos para clasificación
        if inferred_task == "classification" and continuous_cols:
            results["group_comparison"] = compare_groups_by_target(
                df=df,
                target_col=config.target_col,
                feature_cols=continuous_cols
            )
        else:
            results["group_comparison"] = ToolbeltResult(
                data=pd.DataFrame(),
                figure=None,
                title="Comparación de grupos por target"
            )
    else:
        results["target_association"] = ToolbeltResult(
            data=pd.DataFrame(),
            figure=None,
            title="Asociación con variable objetivo"
        )
        results["group_comparison"] = ToolbeltResult(
            data=pd.DataFrame(),
            figure=None,
            title="Comparación de grupos por target"
        )
    
    # === 7) Sugerencias de preprocesamiento ===
    results["imputation_suggestions"] = suggest_imputation(
        df=df,
        continuous_cols=continuous_cols,
        discrete_cols=discrete_cols,
        cat_cols=cat_cols
    )
    
    if continuous_cols:
        results["transformation_suggestions"] = suggest_transformations(df, continuous_cols)
    else:
        results["transformation_suggestions"] = ToolbeltResult(
            data=pd.DataFrame(),
            figure=None,
            title="Transformaciones sugeridas"
        )
    
    # === 8) Checklist de readiness ===
    results["ml_readiness"] = ml_readiness_check(
        df=df,
        target_col=config.target_col,
        task_type=inferred_task,
        continuous_cols=continuous_cols,
        discrete_cols=discrete_cols,
        cat_cols=cat_cols
    )
    intermediate_results["ml_readiness"] = results["ml_readiness"].data
    intermediate_results["outlier_report"] = results["outlier_report"].data
    intermediate_results["normality_report"] = results["normality"].data
    intermediate_results["vif_report"] = results["vif"].data
    intermediate_results["missing_report"] = results["missing_report"].data
    intermediate_results["target_association"] = results["target_association"].data
    
    # === 9) Fitness avanzado ===
    results["fitness"] = evaluate_dataset_ml_fitness_advanced(
        df=df,
        results=intermediate_results,
        target_col=config.target_col,
        task_type=inferred_task,
        date_cols=config.date_cols
    )
    
    results["recommended_actions"] = recommend_next_actions_from_fitness(
        results["fitness"].data
    )
    
    # === 10) Recomendaciones de modelos ===
    # Agregar metadata al intermediate_results
    intermediate_results["metadata"] = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_continuous_features": len(continuous_cols),
        "n_discrete_features": len(discrete_cols),
        "n_low_card_cat_features": len(low_card_cat_cols),
        "n_high_card_cat_features": len(high_card_cat_cols),
        "n_total_features_excluding_target": len([c for c in df.columns if c != config.target_col]),
        "date_cols_used": config.date_cols,
        "id_cols_used": config.id_cols,
        "high_cardinality_threshold": config.high_cardinality_threshold
    }
    intermediate_results["final_verdict"] = results["fitness"].data[
        results["fitness"].data["check"] == "FINAL_VERDICT"
    ]["status"].iloc[0] if not results["fitness"].data.empty else "UNKNOWN"
    
    results["model_recommendations"] = recommend_models_by_data_structure(intermediate_results)
    
    return results