"""
Statistical Toolbelt - Biblioteca de diagnóstico estadístico para datasets tabulares.

Este paquete proporciona funciones para diagnosticar la calidad de datasets
tabulares supervisados, verificar supuestos estadísticos, evaluar aptitud
para ML y recomendar familias de modelos.

Usage:
    from statistical_toolbelt import load_data, ToolbeltConfig, run_full_diagnostic
    
    df = load_data("dataset.csv")
    config = ToolbeltConfig(target_col="target")
    results = run_full_diagnostic(df, config)
    
    # En Jupyter: los resultados se renderizan automáticamente
    results["normality"]  # muestra tabla + figura
    
    # En consola: usar print()
    print(results["normality"])
"""

from .config import ToolbeltConfig
from .types import ToolbeltResult
from .io_utils import load_data
from .inference import infer_column_types, infer_task_type
from .overview import dataset_overview, missing_report, duplicate_report, cardinality_report
from .descriptive import continuous_summary, discrete_summary, iqr_outlier_report
from .normality import normality_tests
from .correlation import (
    pearson_correlation,
    spearman_correlation,
    cramers_v,
    categorical_association_matrix,
    target_association,
    compare_groups_by_target,
    compute_vif
)
from .preprocessing import suggest_imputation, suggest_transformations
from .readiness import ml_readiness_check
from .fitness import evaluate_dataset_ml_fitness_advanced, recommend_next_actions_from_fitness
from .ml_recommendations import recommend_models_by_data_structure
from .runner import run_full_diagnostic

__version__ = "1.0.0"

__all__ = [
    # Config
    "ToolbeltConfig",
    # Types
    "ToolbeltResult",
    # IO
    "load_data",
    # Inference
    "infer_column_types",
    "infer_task_type",
    # Overview
    "dataset_overview",
    "missing_report",
    "duplicate_report",
    "cardinality_report",
    # Descriptive
    "continuous_summary",
    "discrete_summary",
    "iqr_outlier_report",
    # Normality
    "normality_tests",
    # Correlation
    "pearson_correlation",
    "spearman_correlation",
    "cramers_v",
    "categorical_association_matrix",
    "target_association",
    "compare_groups_by_target",
    "compute_vif",
    # Preprocessing
    "suggest_imputation",
    "suggest_transformations",
    # Readiness
    "ml_readiness_check",
    # Fitness
    "evaluate_dataset_ml_fitness_advanced",
    "recommend_next_actions_from_fitness",
    # ML Recommendations
    "recommend_models_by_data_structure",
    # Runner
    "run_full_diagnostic",
]