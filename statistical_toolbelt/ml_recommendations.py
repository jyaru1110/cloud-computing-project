"""
Módulo de recomendaciones de modelos ML.

Este módulo contiene la función que recomienda familias de modelos
basándose en la estructura del dataset y los resultados del diagnóstico.
"""

from typing import Dict

import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from .types import ToolbeltResult
from . import visualization


def recommend_models_by_data_structure(results: Dict) -> ToolbeltResult:
    """Recomienda familias de modelos según la estructura del dataset.
    
    Utiliza un sistema de scoring que combina:
    - Historical prior score (45%): conocimiento histórico de qué modelos
      funcionan bien en tabular
    - Structural fit score (55%): adaptación a las características específicas
      del dataset actual
    
    Args:
        results: Diccionario con resultados del runner avanzado, incluyendo
            task_type, column_types, metadata, fitness_report_advanced, etc.
            
    Returns:
        ToolbeltResult con DataFrame de recomendaciones y figura de barras.
    """
    task_type = results.get("task_type", "unknown")
    col_types = results.get("column_types", {})
    metadata = results.get("metadata", {})
    fitness = results.get("fitness_report_advanced", pd.DataFrame())
    final_verdict = results.get("final_verdict", "UNKNOWN")
    
    continuous_cols = col_types.get("continuous_numeric", [])
    discrete_cols = col_types.get("discrete_numeric", [])
    low_cat = col_types.get("low_cardinality_categorical", [])
    high_cat = col_types.get("high_cardinality_categorical", [])
    
    n_rows = metadata.get("n_rows", 0)
    n_features = metadata.get("n_total_features_excluding_target", 0)
    rows_per_feature = (n_rows / max(n_features, 1)) if n_features > 0 else 0.0
    
    outlier_report = results.get("outlier_report", pd.DataFrame())
    normality_report = results.get("normality_report", pd.DataFrame())
    vif_report = results.get("vif_report", pd.DataFrame())
    missing_report_df = results.get("missing_report", pd.DataFrame())
    
    # Flags de condición del dataset
    has_categorical = (len(low_cat) + len(high_cat)) > 0
    has_high_cardinality = len(high_cat) > 0
    many_categorical = (len(low_cat) + len(high_cat)) >= max(3, int(0.25 * max(n_features, 1)))
    mostly_numeric = (len(continuous_cols) + len(discrete_cols)) >= max(3, int(0.7 * max(n_features, 1)))
    
    # Outliers
    has_many_outliers = False
    severe_outliers = False
    if not outlier_report.empty and "outlier_pct" in outlier_report.columns:
        has_many_outliers = bool((outlier_report["outlier_pct"] > 0.05).any())
        severe_outliers = bool((outlier_report["outlier_pct"] > 0.20).any())
    
    # Normalidad
    many_non_normal = False
    if not normality_report.empty and "normal_by_shapiro_0.05" in normality_report.columns:
        many_non_normal = ((~normality_report["normal_by_shapiro_0.05"]).mean() > 0.5)
    
    # Multicolinealidad
    high_multicollinearity = False
    very_high_multicollinearity = False
    if not vif_report.empty and "VIF" in vif_report.columns:
        vif_clean = vif_report["VIF"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vif_clean) > 0:
            high_multicollinearity = bool((vif_clean > 10).any())
            very_high_multicollinearity = bool((vif_clean > 20).any())
    
    # Missing global
    global_missing = 0.0
    if not missing_report_df.empty and "missing_pct" in missing_report_df.columns:
        global_missing = float(missing_report_df["missing_pct"].mean() / 100.0)
    
    # Flags derivados del fitness
    warn_checks = set()
    fail_checks = set()
    if isinstance(fitness, pd.DataFrame) and not fitness.empty:
        if "status" in fitness.columns and "check" in fitness.columns:
            warn_checks = set(fitness.loc[fitness["status"] == "WARN", "check"].astype(str).tolist())
            fail_checks = set(fitness.loc[fitness["status"] == "FAIL", "check"].astype(str).tolist())
    
    has_class_imbalance = "class_imbalance" in warn_checks
    has_small_classes = "minimum_class_count" in warn_checks
    has_time_drift = any(str(chk).startswith("time_drift_") for chk in warn_checks)
    has_leakage_fail = ("feature_target_leakage" in fail_checks) or ("target_encoded_in_feature" in fail_checks)
    low_signal_regression = "signal_to_noise_proxy" in warn_checks
    hetero_warn = "heteroscedasticity_proxy" in warn_checks
    target_skew_warn = "target_skewness" in warn_checks
    target_outlier_warn = "target_outlier_pct" in warn_checks
    low_rows_feature_warn = "rows_per_feature_ratio" in warn_checks
    high_corr_warn = "high_feature_correlations" in warn_checks
    high_card_warn = "high_cardinality_ratio" in warn_checks
    
    # Helper para clasificar en buckets
    def bucket_from_score(score: float, interpretable: bool = False) -> str:
        if score >= 8.8:
            return "primary_candidate"
        elif score >= 7.6:
            return "strong_baseline"
        elif interpretable and score >= 6.5:
            return "interpretable_baseline"
        elif score >= 5.5:
            return "conditional_candidate"
        else:
            return "avoid_for_now"
    
    recommendations = []
    
    def add_model(
        model_name: str,
        historical_prior_score: float,
        structural_fit_score: float,
        why: list,
        strengths: list,
        risks: list,
        interpretable: bool = False
    ):
        final_score = round((0.45 * historical_prior_score) + (0.55 * structural_fit_score), 3)
        
        if final_verdict == "NOT READY":
            risks = list(risks) + ["Dataset con veredicto NOT READY; corregir problemas antes de entrenar."]
        if has_leakage_fail:
            risks = list(risks) + ["Hay señales de leakage; cualquier benchmark estaría contaminado hasta corregirlo."]
        
        recommendations.append({
            "model": model_name,
            "historical_prior_score": round(historical_prior_score, 3),
            "structural_fit_score": round(structural_fit_score, 3),
            "final_score": final_score,
            "recommended_as": bucket_from_score(final_score, interpretable=interpretable),
            "why": " | ".join(dict.fromkeys(why)),
            "strengths": " | ".join(dict.fromkeys(strengths)),
            "risks": " | ".join(dict.fromkeys(risks)),
        })
    
    # ========================
    # CLASIFICACIÓN
    # ========================
    if task_type == "classification":
        # 1) CatBoostClassifier
        score = 8.8
        why = ["Históricamente muy fuerte en datos tabulares de clasificación."]
        strengths = ["Buen desempeño en tabular.", "Maneja categóricas muy bien.", "Reduce trabajo manual de encoding."]
        risks = []
        
        if has_categorical:
            score += 0.8
            why.append("El dataset tiene variables categóricas.")
        if has_high_cardinality:
            score += 0.5
            why.append("La presencia de alta cardinalidad favorece enfoques tipo CatBoost.")
        if has_many_outliers or many_non_normal:
            score += 0.2
            strengths.append("Tolera razonablemente no linealidad y datos no normales.")
        if low_rows_feature_warn:
            score -= 0.2
            risks.append("Si el dataset es pequeño, vigilar sobreajuste.")
        if has_time_drift:
            risks.append("Si hay drift temporal, usar split temporal.")
        if has_class_imbalance:
            risks.append("Con desbalance, ajustar weights o threshold tuning.")
        
        add_model("CatBoostClassifier", 9.2, min(score, 10.0), why, strengths, risks)
        
        # 2) LightGBM / XGBoost
        score = 8.7
        why = ["Históricamente de los mejores candidatos para clasificación tabular."]
        strengths = ["Captura no linealidad.", "Captura interacciones.", "Muy competitivo en tabular real."]
        risks = []
        
        if mostly_numeric:
            score += 0.5
            why.append("El dataset parece mayormente numérico.")
        if has_many_outliers:
            score += 0.2
        if many_non_normal:
            score += 0.2
        if has_high_cardinality:
            score -= 0.2
            risks.append("Puede requerir encoding cuidadoso en categóricas de alta cardinalidad.")
        if low_rows_feature_warn:
            score -= 0.3
            risks.append("Con pocas filas por feature, vigilar overfitting.")
        if has_class_imbalance:
            risks.append("Usar class weights, scale_pos_weight o tuning de threshold.")
        if has_time_drift:
            risks.append("Preferir validación temporal si hay fecha.")
        
        add_model("LightGBMClassifier / XGBoostClassifier", 9.4, min(score, 10.0), why, strengths, risks)
        
        # 3) RandomForestClassifier
        score = 7.3
        why = ["Históricamente es un baseline robusto en clasificación tabular."]
        strengths = ["Robusto.", "Simple de usar.", "Tolera no linealidad moderada."]
        risks = []
        
        if has_many_outliers or many_non_normal:
            score += 0.3
        if n_rows > 100000:
            score -= 0.3
            risks.append("Puede volverse más pesado o menos competitivo en datasets muy grandes.")
        if has_high_cardinality:
            score -= 0.2
            risks.append("No suele ser la mejor primera opción con categóricas complejas.")
        if high_corr_warn:
            strengths.append("Suele sufrir menos que modelos lineales ante colinealidad moderada.")
        if has_class_imbalance:
            risks.append("Revisar balance de clases y métricas adecuadas.")
        
        add_model("RandomForestClassifier", 7.8, min(score, 10.0), why, strengths, risks)
        
        # 4) LogisticRegression
        score = 6.8
        why = ["Históricamente es el baseline interpretable por excelencia en clasificación."]
        strengths = ["Alta interpretabilidad.", "Entrena rápido.", "Muy útil como baseline."]
        risks = []
        
        if mostly_numeric:
            score += 0.3
        if high_multicollinearity:
            score -= 0.3
            why.append("Hay colinealidad; preferir regularización.")
            strengths.append("Con Ridge/L2 o Elastic Net puede estabilizarse.")
        if very_high_multicollinearity:
            score -= 0.3
        if has_many_outliers or many_non_normal:
            score -= 0.4
            risks.append("Puede quedarse corto si la estructura es muy no lineal.")
        if has_categorical:
            risks.append("Requiere encoding de categóricas.")
        if has_class_imbalance:
            risks.append("Usar class_weight='balanced' o calibrar threshold.")
        if rows_per_feature >= 15:
            score += 0.2
        
        add_model("LogisticRegression", 7.6, min(score, 10.0), why, strengths, risks, interpretable=True)
        
        # 5) LinearSVC / SVC
        score = 5.8
        why = ["Puede funcionar bien en ciertos datasets de clasificación con frontera compleja."]
        strengths = ["Capta fronteras complejas.", "Útil en datasets medianos y bien preparados."]
        risks = []
        
        if n_rows > 50000:
            score -= 1.0
            risks.append("Escala mal en datasets grandes.")
        if has_categorical:
            score -= 0.4
            risks.append("Requiere encoding y escalado.")
        if has_many_outliers:
            score -= 0.3
            risks.append("Sensible a escala y ruido.")
        if rows_per_feature >= 20 and n_rows < 30000:
            score += 0.3
        
        add_model("LinearSVC / SVC", 5.9, min(max(score, 0.0), 10.0), why, strengths, risks)
        
        # 6) KNeighborsClassifier
        score = 3.8
        why = ["Modelo situacional; históricamente no suele ser el primer candidato en tabular moderno."]
        strengths = ["Simple.", "Puede servir en datasets pequeños y bien escalados."]
        risks = ["Muy sensible a escala.", "Sufre con ruido.", "Sufre con alta dimensión."]
        
        if n_rows > 20000:
            score -= 0.8
            risks.append("Costoso con muchas filas.")
        if n_features > 25:
            score -= 0.6
        if has_categorical:
            score -= 0.5
        if rows_per_feature >= 20 and n_features <= 15:
            score += 0.4
        
        add_model("KNeighborsClassifier", 4.2, min(max(score, 0.0), 10.0), why, strengths, risks)
        
        # 7) Neural tabular / MLP
        score = 4.8
        why = ["En tabular clásico, históricamente no suele superar a boosting como primera apuesta."]
        strengths = ["Puede capturar relaciones complejas.", "Útil si hay mucha data y buen tuning."]
        risks = ["Mayor sensibilidad a tuning.", "Requiere pipeline más fino.", "En tabular suele perder contra boosting como baseline."]
        
        if n_rows > 100000 and n_features > 20:
            score += 0.8
            why.append("El volumen de datos podría justificar explorarlo.")
        if has_categorical:
            risks.append("Requiere encoding o embeddings.")
        if global_missing > 0:
            risks.append("Necesita imputación previa consistente.")
        
        add_model("MLPClassifier / Tabular NN", 4.8, min(max(score, 0.0), 10.0), why, strengths, risks)
    
    # ========================
    # REGRESIÓN
    # ========================
    elif task_type == "regression":
        # 1) LightGBM / XGBoost Regressor
        score = 8.8
        why = ["Históricamente de los candidatos más fuertes en regresión tabular."]
        strengths = ["Captura no linealidad.", "Captura interacciones.", "Muy competitivo en tabular."]
        risks = []
        
        if has_many_outliers:
            score += 0.2
        if many_non_normal:
            score += 0.2
        if target_skew_warn or target_outlier_warn:
            score += 0.2
            strengths.append("Suele tolerar targets complejos mejor que modelos lineales puros.")
        if has_high_cardinality:
            score -= 0.2
            risks.append("Con categóricas complejas, CatBoost puede ser mejor.")
        if low_rows_feature_warn:
            score -= 0.3
            risks.append("Con pocas filas por feature, vigilar overfitting.")
        if has_time_drift:
            risks.append("Si hay drift temporal, usar validación temporal.")
        
        add_model("LightGBMRegressor / XGBoostRegressor", 9.4, min(score, 10.0), why, strengths, risks)
        
        # 2) CatBoostRegressor
        score = 8.3
        why = ["Históricamente muy competitivo en regresión tabular, sobre todo con categóricas."]
        strengths = ["Muy útil con categóricas.", "Reduce trabajo de encoding.", "Fuerte baseline tabular."]
        risks = []
        
        if has_categorical:
            score += 0.8
            why.append("El dataset tiene categóricas relevantes.")
        if has_high_cardinality:
            score += 0.4
            why.append("La alta cardinalidad favorece un enfoque tipo CatBoost.")
        if low_rows_feature_warn:
            score -= 0.2
            risks.append("Con pocos datos, monitorear sobreajuste.")
        if has_time_drift:
            risks.append("Usar split temporal si hay drift.")
        
        add_model("CatBoostRegressor", 8.9, min(score, 10.0), why, strengths, risks)
        
        # 3) RandomForestRegressor
        score = 7.0
        why = ["Históricamente es un baseline robusto para regresión tabular."]
        strengths = ["Robusto.", "Simple.", "Tolera no linealidad moderada."]
        risks = []
        
        if has_many_outliers:
            score += 0.2
        if n_rows > 100000:
            score -= 0.3
            risks.append("Puede quedarse atrás frente a boosting en datasets grandes.")
        if target_outlier_warn:
            risks.append("Outliers fuertes en target pueden afectar estabilidad.")
        if has_high_cardinality:
            score -= 0.2
        
        add_model("RandomForestRegressor", 7.5, min(score, 10.0), why, strengths, risks)
        
        # 4) LinearRegression / Ridge / ElasticNet
        score = 6.8
        why = ["Históricamente son baselines interpretables muy valiosos en regresión."]
        strengths = ["Interpretabilidad.", "Rapidez.", "Muy útiles como baseline."]
        risks = []
        
        if high_multicollinearity:
            strengths.append("Ridge / ElasticNet ayudan bastante con colinealidad.")
            why.append("Hay colinealidad; regularización puede ser especialmente útil.")
            score += 0.2
        if target_skew_warn or target_outlier_warn:
            score -= 0.5
            risks.append("Target sesgado o con outliers puede perjudicar modelos lineales puros.")
        if hetero_warn:
            score -= 0.3
            risks.append("Posible heterocedasticidad; revisar transformación del target o modelos alternativos.")
        if low_signal_regression:
            risks.append("Señal lineal aparente baja; podría quedarse corto.")
        if has_categorical:
            risks.append("Requiere encoding de categóricas.")
        if rows_per_feature >= 15:
            score += 0.2
        
        add_model("LinearRegression / Ridge / ElasticNet", 7.7, min(score, 10.0), why, strengths, risks, interpretable=True)
        
        # 5) Huber / Robust Regression
        score = 6.2
        why = ["Históricamente son una buena alternativa cuando hay ruido u outliers."]
        strengths = ["Robustez a outliers.", "Útil en regresión tabular ruidosa."]
        risks = []
        
        if target_outlier_warn or severe_outliers:
            score += 1.0
            why.append("El target o varias features muestran outliers importantes.")
        if target_skew_warn:
            score += 0.2
        if has_categorical:
            risks.append("Requiere encoding si hay categóricas.")
        if low_signal_regression:
            risks.append("Si la señal es muy baja, incluso modelos robustos pueden rendir poco.")
        
        add_model("HuberRegressor / Robust Regression", 6.5, min(score, 10.0), why, strengths, risks)
        
        # 6) SVR
        score = 4.8
        why = ["Puede funcionar en regresión pequeña/mediana bien escalada."]
        strengths = ["Puede captar no linealidad.", "Útil en escenarios específicos."]
        risks = ["Requiere escalado.", "No suele ser primera apuesta en tabular grande.", "Mayor costo computacional."]
        
        if n_rows > 30000:
            score -= 1.0
        if has_categorical:
            score -= 0.4
            risks.append("Requiere encoding.")
        if rows_per_feature >= 20 and n_rows < 15000:
            score += 0.3
        
        add_model("SVR", 5.0, min(max(score, 0.0), 10.0), why, strengths, risks)
        
        # 7) MLP / Tabular NN
        score = 4.8
        why = ["En tabular clásico, históricamente boosting suele dominar como primera elección."]
        strengths = ["Puede capturar relaciones complejas.", "Puede valer la pena con mucha data."]
        risks = ["Más tuning.", "Más sensible a preparación.", "No suele ser baseline principal en tabular clásico."]
        
        if n_rows > 100000 and n_features > 20:
            score += 0.8
            why.append("El tamaño del dataset podría justificar explorarlo.")
        if global_missing > 0:
            risks.append("Necesita imputación previa consistente.")
        if has_categorical:
            risks.append("Requiere encoding o embeddings.")
        
        add_model("MLPRegressor / Tabular NN", 4.8, min(max(score, 0.0), 10.0), why, strengths, risks)
    
    else:
        models_df = pd.DataFrame([{
            "model": "unknown",
            "historical_prior_score": 0.0,
            "structural_fit_score": 0.0,
            "final_score": 0.0,
            "recommended_as": "avoid_for_now",
            "why": "No se pudo inferir correctamente el tipo de tarea.",
            "strengths": "Define target_col y task_type correctamente.",
            "risks": "No se puede recomendar familia de modelos sin target o task_type válido."
        }])
        fig = visualization._create_figure_with_text("Tipo de tarea desconocido")
        return ToolbeltResult(data=models_df, figure=fig, title="Recomendación de modelos por estructura de datos")
    
    # Ordenar por puntuación final
    models_df = pd.DataFrame(recommendations).sort_values(
        by=["final_score", "historical_prior_score", "structural_fit_score"],
        ascending=False
    ).reset_index(drop=True)
    
    fig = visualization._build_model_recommendations_barplot(models_df)
    
    return ToolbeltResult(
        data=models_df,
        figure=fig,
        title="Recomendación de modelos por estructura de datos"
    )