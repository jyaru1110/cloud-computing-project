"""
Módulo de evaluación avanzada de aptitud ML.

Este módulo contiene funciones para evaluar la aptitud del dataset
para machine learning de forma avanzada, incluyendo detección de
leakage, desbalance, heterocedasticidad y drift temporal.
"""

from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from .types import ToolbeltResult
from . import visualization


# Intentar importar sklearn avanzado con try/except
try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    SKLEARN_ADVANCED_AVAILABLE = True
except ImportError:
    SKLEARN_ADVANCED_AVAILABLE = False


# =========================================================
# Funciones helper privadas
# =========================================================

def _safe_numeric_series(s: pd.Series) -> pd.Series:
    """Convierte una serie a numérico, reemplazando errores con NaN."""
    return pd.to_numeric(s, errors="coerce")


def _is_probable_id_column(s: pd.Series, uniqueness_threshold: float = 0.95) -> bool:
    """Detecta si una columna es probablemente un ID por su unicidad."""
    non_null = s.dropna()
    if len(non_null) == 0:
        return False
    unique_ratio = non_null.nunique() / len(non_null)
    return unique_ratio >= uniqueness_threshold


def _is_id_like_name(col: str) -> bool:
    """Detecta nombres de columnas que parecen IDs."""
    col_low = col.lower()
    patterns = ["id", "_id", "uuid", "folio", "customer_id", "user_id", "account_id", "case_id"]
    return any(p in col_low for p in patterns)


def _count_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.95) -> int:
    """Cuenta pares de features con correlación absoluta mayor al umbral."""
    if corr_matrix is None or corr_matrix.empty:
        return 0
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return int((upper.abs() > threshold).sum().sum())


def _classification_majority_ratio(y: pd.Series) -> float:
    """Calcula la proporción de la clase mayoritaria."""
    vc = y.value_counts(normalize=True, dropna=True)
    return float(vc.iloc[0]) if len(vc) else 1.0


def _classification_min_class_count(y: pd.Series) -> int:
    """Cuenta el número mínimo de muestras en cualquier clase."""
    vc = y.value_counts(dropna=True)
    return int(vc.min()) if len(vc) else 0


def _target_encoded_in_feature(df: pd.DataFrame, target_col: str, feature_col: str) -> bool:
    """Detecta si una feature es el target recodificado."""
    tmp = df[[target_col, feature_col]].dropna()
    if tmp.empty:
        return False
    
    # Caso exacto
    if tmp[target_col].astype(str).equals(tmp[feature_col].astype(str)):
        return True
    
    # Misma cardinalidad + mapping 1:1 perfecto
    mapping_counts = tmp.groupby(feature_col)[target_col].nunique(dropna=True)
    reverse_counts = tmp.groupby(target_col)[feature_col].nunique(dropna=True)
    
    if len(mapping_counts) > 0 and len(reverse_counts) > 0:
        if (mapping_counts.max() == 1) and (reverse_counts.max() == 1):
            if tmp[feature_col].nunique() == tmp[target_col].nunique():
                return True
    
    return False


def _perfect_or_suspicious_classification_leakage(
    df: pd.DataFrame, 
    target_col: str, 
    feature_col: str
) -> dict:
    """Detecta leakage fuerte cuando una feature predice casi perfecto el target."""
    tmp = df[[target_col, feature_col]].dropna()
    if tmp.empty:
        return {"flag": False, "reason": "empty"}
    
    if tmp[target_col].nunique() < 2:
        return {"flag": False, "reason": "target_constant"}
    
    # Si es exactamente el target recodificado
    if _target_encoded_in_feature(tmp, target_col, feature_col):
        return {"flag": True, "reason": "target_encoded_in_feature"}
    
    # Pureza de grupos
    purity = tmp.groupby(feature_col)[target_col].agg(
        lambda x: x.value_counts(normalize=True).iloc[0]
    )
    weighted_purity = np.average(
        purity.values,
        weights=tmp[feature_col].value_counts().loc[purity.index].values
    ) if len(purity) else 0.0
    
    if weighted_purity >= 0.98 and tmp[feature_col].nunique() > 1:
        return {"flag": True, "reason": f"near_perfect_group_purity={weighted_purity:.4f}"}
    
    return {"flag": False, "reason": f"group_purity={weighted_purity:.4f}"}


def _regression_target_outlier_pct(y: pd.Series) -> float:
    """Calcula el porcentaje de outliers en el target de regresión."""
    s = y.dropna()
    if len(s) < 5:
        return 0.0
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return float(((s < lo) | (s > hi)).mean())


def _signal_to_noise_ratio_proxy_regression(
    df: pd.DataFrame, 
    target_col: str, 
    numeric_features: list
) -> float:
    """Proxy de señal a ruido: R² holdout de regresión lineal simple."""
    if not SKLEARN_ADVANCED_AVAILABLE:
        return np.nan
    
    cols = [c for c in numeric_features if c != target_col]
    if len(cols) == 0:
        return np.nan
    
    tmp = df[cols + [target_col]].copy()
    tmp = tmp.apply(pd.to_numeric, errors="coerce")
    tmp = tmp.dropna()
    
    if len(tmp) < 30:
        return np.nan
    
    X = tmp[cols]
    y = tmp[target_col]
    
    if X.shape[1] == 0 or y.nunique() <= 1:
        return np.nan
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return float(r2_score(y_test, preds))


def _heteroscedasticity_proxy(
    df: pd.DataFrame, 
    target_col: str, 
    numeric_features: list
) -> dict:
    """Proxy de heterocedasticidad: correlación de |residual| con predicción y target."""
    if not SKLEARN_ADVANCED_AVAILABLE:
        return {"flag": False, "corr_abs_resid_pred": np.nan, "corr_abs_resid_target": np.nan}
    
    cols = [c for c in numeric_features if c != target_col]
    if len(cols) == 0:
        return {"flag": False, "corr_abs_resid_pred": np.nan, "corr_abs_resid_target": np.nan}
    
    tmp = df[cols + [target_col]].copy()
    tmp = tmp.apply(pd.to_numeric, errors="coerce").dropna()
    
    if len(tmp) < 30 or tmp[target_col].nunique() <= 1:
        return {"flag": False, "corr_abs_resid_pred": np.nan, "corr_abs_resid_target": np.nan}
    
    X = tmp[cols]
    y = tmp[target_col]
    
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    resid = y - preds
    abs_resid = np.abs(resid)
    
    corr1 = np.corrcoef(abs_resid, preds)[0, 1] if len(abs_resid) > 1 else np.nan
    corr2 = np.corrcoef(abs_resid, y)[0, 1] if len(abs_resid) > 1 else np.nan
    
    corr1 = 0.0 if pd.isna(corr1) else float(corr1)
    corr2 = 0.0 if pd.isna(corr2) else float(corr2)
    
    flag = (abs(corr1) >= 0.30) or (abs(corr2) >= 0.30)
    
    return {
        "flag": flag,
        "corr_abs_resid_pred": corr1,
        "corr_abs_resid_target": corr2
    }


def _temporal_drift_proxy(
    df: pd.DataFrame, 
    date_col: str, 
    target_col: str = None, 
    n_splits: int = 4
) -> dict:
    """Proxy de drift temporal: cambio en distribución del target por tiempo."""
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col]).sort_values(date_col)
    
    if len(tmp) < max(40, n_splits * 10):
        return {"flag": False, "reason": "not_enough_rows"}
    
    tmp["_time_bin"] = pd.qcut(
        np.arange(len(tmp)), q=n_splits, labels=False, duplicates="drop"
    )
    
    if target_col is None or target_col not in tmp.columns:
        return {"flag": False, "reason": "no_target"}
    
    y = tmp[target_col]
    
    if pd.api.types.is_numeric_dtype(y):
        grp = tmp.groupby("_time_bin")[target_col].mean()
        relative_shift = float((grp.max() - grp.min()) / (abs(grp.mean()) + 1e-9))
        return {
            "flag": relative_shift > 0.30,
            "reason": "regression_target_time_shift",
            "relative_shift": relative_shift
        }
    else:
        dist = pd.crosstab(tmp["_time_bin"], tmp[target_col], normalize="index")
        mean_dist = dist.mean(axis=0)
        max_l1 = float(dist.apply(
            lambda row: np.abs(row - mean_dist).sum(), axis=1
        ).max())
        return {
            "flag": max_l1 > 0.40,
            "reason": "classification_target_time_shift",
            "max_l1_shift": max_l1
        }


# =========================================================
# Función principal de fitness avanzado
# =========================================================

def evaluate_dataset_ml_fitness_advanced(
    df: pd.DataFrame,
    results: dict,
    target_col: str = None,
    task_type: str = "auto",
    date_cols: list = None,
    high_cardinality_ratio_warn: float = 0.30,
    id_like_ratio_warn: float = 0.20,
    correlation_threshold: float = 0.95,
    max_high_corr_pairs_warn: int = 5,
    min_rows_per_feature_ratio: float = 10.0,
    class_imbalance_warn_ratio: float = 0.90,
    min_class_count_warn: int = 20,
    target_skew_warn: float = 2.0,
    target_outlier_warn_pct: float = 0.10,
    min_regression_signal_r2: float = 0.05
) -> ToolbeltResult:
    """Evaluación avanzada de aptitud ML.
    
    Verifica múltiples dimensiones:
    - Generales: cardinalidad, IDs, correlaciones, ratio filas/features
    - Temporal: drift si hay columnas de fecha
    - Clasificación: desbalance, clases mínimas, leakage
    - Regresión: skew del target, outliers, heterocedasticidad, señal
    
    Args:
        df: DataFrame a analizar.
        results: Diccionario con resultados del runner básico (column_types, etc).
        target_col: Nombre de la columna objetivo.
        task_type: Tipo de tarea (classification/regression).
        date_cols: Lista de columnas de fecha.
        Parámetros de umbral adicionales para customize warnings.
        
    Returns:
        ToolbeltResult con DataFrame de checks y figura de estado.
    """
    checks = []
    date_cols = date_cols or []
    
    inferred_task = results.get("task_type", task_type)
    col_types = results.get("column_types", {})
    
    continuous_cols = col_types.get("continuous_numeric", [])
    discrete_cols = col_types.get("discrete_numeric", [])
    cat_low = col_types.get("low_cardinality_categorical", [])
    cat_high = col_types.get("high_cardinality_categorical", [])
    all_features = [c for c in df.columns if c != target_col]
    
    # === GENERALES ===
    total_features = len(all_features)
    high_card_ratio = (len(cat_high) / total_features) if total_features > 0 else 0.0
    checks.append({
        "category": "general",
        "check": "high_cardinality_ratio",
        "status": "PASS" if high_card_ratio <= high_cardinality_ratio_warn else "WARN",
        "value": round(high_card_ratio, 4),
        "rule": f"<= {high_cardinality_ratio_warn}",
        "comment": f"{len(cat_high)} high-card categorical cols de {total_features} features"
    })
    
    # IDs probables
    probable_id_cols = []
    for c in all_features:
        if _is_id_like_name(c) or _is_probable_id_column(df[c]):
            probable_id_cols.append(c)
    
    id_like_ratio = (len(probable_id_cols) / total_features) if total_features > 0 else 0.0
    checks.append({
        "category": "general",
        "check": "id_like_columns_ratio",
        "status": "PASS" if id_like_ratio <= id_like_ratio_warn else "WARN",
        "value": round(id_like_ratio, 4),
        "rule": f"<= {id_like_ratio_warn}",
        "comment": f"ID-like columns: {probable_id_cols[:10]}"
    })
    
    # Altas correlaciones
    corr_df = results.get("correlations", {})
    if isinstance(corr_df, dict):
        corr_df = corr_df.get("pearson", pd.DataFrame())
    else:
        corr_df = pd.DataFrame()
    
    high_corr_pairs = _count_high_correlations(corr_df, threshold=correlation_threshold)
    checks.append({
        "category": "general",
        "check": "high_feature_correlations",
        "status": "PASS" if high_corr_pairs <= max_high_corr_pairs_warn else "WARN",
        "value": int(high_corr_pairs),
        "rule": f"<= {max_high_corr_pairs_warn} pares con |r| > {correlation_threshold}",
        "comment": "Correlación extrema entre features"
    })
    
    # Ratio filas/features
    rows_per_feature = (len(df) / total_features) if total_features > 0 else 0.0
    checks.append({
        "category": "general",
        "check": "rows_per_feature_ratio",
        "status": "PASS" if rows_per_feature >= min_rows_per_feature_ratio else "WARN",
        "value": round(rows_per_feature, 4),
        "rule": f">= {min_rows_per_feature_ratio}",
        "comment": "Muy pocas filas por feature incrementa riesgo de sobreajuste"
    })
    
    # === TEMPORAL ===
    if len(date_cols) > 0:
        for dc in date_cols:
            if dc in df.columns:
                drift_res = _temporal_drift_proxy(df, dc, target_col=target_col)
                checks.append({
                    "category": "temporal",
                    "check": f"time_drift_{dc}",
                    "status": "WARN" if drift_res.get("flag", False) else "PASS",
                    "value": str({k: v for k, v in drift_res.items() if k != "flag"}),
                    "rule": "sin drift fuerte",
                    "comment": "Cambio de distribución del target a lo largo del tiempo"
                })
    
    # === CLASIFICACIÓN ===
    if inferred_task == "classification" and target_col is not None and target_col in df.columns:
        y = df[target_col].dropna()
        
        # Desbalance
        maj_ratio = _classification_majority_ratio(y)
        checks.append({
            "category": "classification",
            "check": "class_imbalance",
            "status": "WARN" if maj_ratio >= class_imbalance_warn_ratio else "PASS",
            "value": round(maj_ratio, 4),
            "rule": f"< {class_imbalance_warn_ratio}",
            "comment": "Proporción de la clase mayoritaria"
        })
        
        # Clase mínima
        min_class_count = _classification_min_class_count(y)
        checks.append({
            "category": "classification",
            "check": "minimum_class_count",
            "status": "WARN" if min_class_count < min_class_count_warn else "PASS",
            "value": int(min_class_count),
            "rule": f">= {min_class_count_warn}",
            "comment": "Muestras mínimas en la clase más pequeña"
        })
        
        # Leakage
        leakage_flags = []
        for c in all_features:
            leak = _perfect_or_suspicious_classification_leakage(df, target_col, c)
            if leak["flag"]:
                leakage_flags.append((c, leak["reason"]))
        
        checks.append({
            "category": "classification",
            "check": "feature_target_leakage",
            "status": "FAIL" if len(leakage_flags) > 0 else "PASS",
            "value": len(leakage_flags),
            "rule": "0 features con señal casi perfecta sospechosa",
            "comment": str(leakage_flags[:10])
        })
        
        # Target encoded
        encoded_target_cols = []
        for c in all_features:
            if _target_encoded_in_feature(df, target_col, c):
                encoded_target_cols.append(c)
        
        checks.append({
            "category": "classification",
            "check": "target_encoded_in_feature",
            "status": "FAIL" if len(encoded_target_cols) > 0 else "PASS",
            "value": len(encoded_target_cols),
            "rule": "0 features equivalentes al target",
            "comment": str(encoded_target_cols[:10])
        })
    
    # === REGRESIÓN ===
    if inferred_task == "regression" and target_col is not None and target_col in df.columns:
        y = _safe_numeric_series(df[target_col]).dropna()
        
        if len(y) > 0:
            # Skew del target
            target_skew = float(y.skew()) if len(y) > 2 else 0.0
            checks.append({
                "category": "regression",
                "check": "target_skewness",
                "status": "WARN" if abs(target_skew) >= target_skew_warn else "PASS",
                "value": round(target_skew, 4),
                "rule": f"|skew| < {target_skew_warn}",
                "comment": "Target muy sesgado puede requerir transformación"
            })
            
            # Outliers del target
            target_outlier_pct = _regression_target_outlier_pct(y)
            checks.append({
                "category": "regression",
                "check": "target_outlier_pct",
                "status": "WARN" if target_outlier_pct >= target_outlier_warn_pct else "PASS",
                "value": round(target_outlier_pct, 4),
                "rule": f"< {target_outlier_warn_pct}",
                "comment": "Outliers excesivos en target pueden desestabilizar regresión"
            })
        
        # Heterocedasticidad
        numeric_features = continuous_cols + discrete_cols
        het = _heteroscedasticity_proxy(df, target_col, numeric_features)
        checks.append({
            "category": "regression",
            "check": "heteroscedasticity_proxy",
            "status": "WARN" if het["flag"] else "PASS",
            "value": f"corr_abs_resid_pred={round(het['corr_abs_resid_pred'],4)}, corr_abs_resid_target={round(het['corr_abs_resid_target'],4)}",
            "rule": "correlaciones bajas entre |residual| y predicción/target",
            "comment": "Posible heterocedasticidad"
        })
        
        # Señal/ruido proxy
        signal_r2 = _signal_to_noise_ratio_proxy_regression(df, target_col, numeric_features)
        if pd.isna(signal_r2):
            sig_status = "WARN"
            sig_value = np.nan
            sig_comment = "No fue posible estimar señal/ruido proxy"
        else:
            sig_status = "PASS" if signal_r2 >= min_regression_signal_r2 else "WARN"
            sig_value = round(signal_r2, 4)
            sig_comment = "R² holdout lineal como proxy de señal total"
        
        checks.append({
            "category": "regression",
            "check": "signal_to_noise_proxy",
            "status": sig_status,
            "value": sig_value,
            "rule": f"R² proxy >= {min_regression_signal_r2}",
            "comment": sig_comment
        })
    
    # === BASE CHECKS ===
    base_mlr = results.get("ml_readiness", pd.DataFrame())
    if not base_mlr.empty:
        fail_count = int((base_mlr["status"] == "FAIL").sum())
        warn_count = int((base_mlr["status"] == "WARN").sum())
        
        checks.append({
            "category": "base",
            "check": "base_fail_count",
            "status": "FAIL" if fail_count > 0 else "PASS",
            "value": fail_count,
            "rule": "0 FAIL",
            "comment": "Checklist base heredado"
        })
        
        checks.append({
            "category": "base",
            "check": "base_warn_count",
            "status": "WARN" if warn_count > 2 else "PASS",
            "value": warn_count,
            "rule": "<= 2 WARN",
            "comment": "Warnings acumulados del checklist base"
        })
    
    fitness_df = pd.DataFrame(checks)
    
    # Veredicto final
    fail_total = int((fitness_df["status"] == "FAIL").sum())
    warn_total = int((fitness_df["status"] == "WARN").sum())
    
    if fail_total > 0:
        final_verdict = "NOT READY"
    elif warn_total >= 3:
        final_verdict = "READY WITH WARNINGS"
    else:
        final_verdict = "READY"
    
    summary_row = pd.DataFrame([{
        "category": "final",
        "check": "FINAL_VERDICT",
        "status": final_verdict,
        "value": "",
        "rule": "",
        "comment": f"FAIL={fail_total}, WARN={warn_total}"
    }])
    
    full_df = pd.concat([fitness_df, summary_row], ignore_index=True)
    
    fig = visualization._build_fitness_grouped_barplot(full_df)
    
    return ToolbeltResult(
        data=full_df,
        figure=fig,
        title="Evaluación de aptitud ML avanzada"
    )


def recommend_next_actions_from_fitness(fitness_df: pd.DataFrame) -> ToolbeltResult:
    """Sugiere acciones basadas en los checks de fitness que fallaron.
    
    Args:
        fitness_df: DataFrame con resultados de evaluate_dataset_ml_fitness_advanced.
        
    Returns:
        ToolbeltResult con DataFrame de acciones y sin figura.
    """
    actions = []
    
    for _, row in fitness_df.iterrows():
        check = row["check"]
        status = row["status"]
        
        if status == "PASS" or check == "FINAL_VERDICT":
            continue
        
        if check == "class_imbalance":
            actions.append({
                "issue": check,
                "action": "Usar stratified split, class weights, focal loss, oversampling o threshold tuning."
            })
        elif check == "minimum_class_count":
            actions.append({
                "issue": check,
                "action": "Revisar rare classes, agrupar clases o recolectar más muestras."
            })
        elif check == "feature_target_leakage":
            actions.append({
                "issue": check,
                "action": "Eliminar features con predicción casi perfecta sospechosa y revisar si contienen información posterior al evento."
            })
        elif check == "target_encoded_in_feature":
            actions.append({
                "issue": check,
                "action": "Eliminar inmediatamente la feature que codifica el target."
            })
        elif check == "target_skewness":
            actions.append({
                "issue": check,
                "action": "Considerar log transform, Yeo-Johnson o métricas robustas."
            })
        elif check == "target_outlier_pct":
            actions.append({
                "issue": check,
                "action": "Winsorización, clipping informado, robust regression o revisión de calidad del target."
            })
        elif check == "heteroscedasticity_proxy":
            actions.append({
                "issue": check,
                "action": "Probar transformación del target, weighted regression o modelos basados en árboles."
            })
        elif check == "signal_to_noise_proxy":
            actions.append({
                "issue": check,
                "action": "Explorar mejores features, interacciones, señales temporales o revisar si el problema es realmente modelable."
            })
        elif check == "high_cardinality_ratio":
            actions.append({
                "issue": check,
                "action": "Usar target encoding con cuidado, frequency encoding, hashing o CatBoost."
            })
        elif check == "id_like_columns_ratio":
            actions.append({
                "issue": check,
                "action": "Eliminar columnas ID-like salvo que tengan valor semántico real."
            })
        elif check == "high_feature_correlations":
            actions.append({
                "issue": check,
                "action": "Eliminar variables redundantes o usar regularización / PCA según el caso."
            })
        elif check == "rows_per_feature_ratio":
            actions.append({
                "issue": check,
                "action": "Reducir dimensionalidad, seleccionar features o conseguir más datos."
            })
        elif "time_drift_" in check:
            actions.append({
                "issue": check,
                "action": "Usar split temporal, monitoreo de drift y posiblemente reentrenamiento periódico."
            })
        elif check == "base_fail_count":
            actions.append({
                "issue": check,
                "action": "Resolver primero los FAIL del checklist base antes de modelar."
            })
        elif check == "base_warn_count":
            actions.append({
                "issue": check,
                "action": "Atender los WARN del checklist base para un pipeline más estable."
            })
    
    if not actions:
        actions.append({
            "issue": "none",
            "action": "No se detectaron acciones urgentes. El dataset luce apto para modelado."
        })
    
    actions_df = pd.DataFrame(actions).drop_duplicates()
    
    return ToolbeltResult(
        data=actions_df,
        figure=None,
        title="Acciones recomendadas"
    )