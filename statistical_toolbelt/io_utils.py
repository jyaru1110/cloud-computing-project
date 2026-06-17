"""
Utilidades de entrada/salida para el Statistical Toolbelt.

Este módulo contiene la función load_data que maneja la carga de datasets
desde diferentes formatos de archivo.
"""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Carga un dataset desde un archivo.
    
    Soporta los formatos más comunes: CSV, Parquet y Excel.
    
    Args:
        path: Ruta al archivo del dataset.
        
    Returns:
        DataFrame con los datos cargados.
        
    Raises:
        ValueError: Si el formato del archivo no es soportado.
        
    Example:
        >>> df = load_data("dataset.csv")
        >>> df = load_data("data.parquet")
    """
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(
            "Formato no soportado. Usa CSV, Parquet o Excel."
        )