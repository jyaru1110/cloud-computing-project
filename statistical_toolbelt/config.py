"""
Configuración centralizada para el Statistical Toolbelt.

Este módulo define la clase de configuración que reemplaza las variables globales
hardcodeadas del notebook original.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ToolbeltConfig:
    """Configuración para el análisis de diagnóstico estadístico.
    
    Reemplaza las variables globales DATA_PATH, TARGET_COL, ID_COLS, etc.
    del notebook original con una estructura de datos centralizada.
    """
    
    target_col: Optional[str] = None
    task_type: str = "auto"  # auto | classification | regression
    id_cols: List[str] = field(default_factory=list)
    date_cols: List[str] = field(default_factory=list)
    high_cardinality_threshold: int = 50
    missing_warn_threshold: float = 0.20
    outlier_warn_threshold: float = 0.05
    random_state: int = 42
    
    def __post_init__(self):
        """Validación básica de los parámetros."""
        if self.task_type not in ("auto", "classification", "regression"):
            raise ValueError(
                "task_type debe ser 'auto', 'classification' o 'regression', no '{}'".format(self.task_type)
            )
        if not 0 <= self.missing_warn_threshold <= 1:
            raise ValueError("missing_warn_threshold debe estar entre 0 y 1")
        if not 0 <= self.outlier_warn_threshold <= 1:
            raise ValueError("outlier_warn_threshold debe estar entre 0 y 1")