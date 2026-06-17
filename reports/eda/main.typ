// Reporte de EDA - Jigsaw Toxic Comment Classification Challenge
// Compilar con: typst compile reports/eda/main.typ

#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
  lang: "es",
)

#set heading(numbering: "1.1")

#set par(justify: true, leading: 0.75em)

#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  it
}

// Titulo
#align(center)[
  #text(size: 22pt, weight: "bold")[
    Análisis Exploratorio del Dataset Jigsaw\
    Toxic Comment Classification Challenge
  ]
  #v(1em)
  #text(size: 12pt)[
    Dataset de comentarios tóxicos de Wikipedia\

    159,571 comentarios etiquetados con seis categorías de toxicidad\
  ]
  #v(2em)
]

#pagebreak()

// ============================================================
// 1. HIPÓTESIS
// ============================================================
= Hipótesis

Antes de cualquier modelado se formularon cinco hipótesis explícitas que guían el análisis y definen qué constituye evidencia a favor o en contra.

*H1.* Las etiquetas de toxicidad no son independientes. Existe una estructura de co-ocurrencia donde algunas etiquetas aparecen juntas con frecuencia mucho mayor que la esperada por azar (por ejemplo, *obscene* e *insult*). Si esto se confirma, modelar cada etiqueta por separado pierde información y un modelo multietiqueta es preferible.

*H2.* Los comentarios tóxicos tienen distribuciones de longitud y proporción de mayúsculas distintas a los no tóxicos. Los comentarios tóxicos serían más cortos y con más mayúsculas. Esta hipótesis se basa en que la agresividad tiende a expresarse en fragmentos cortos y enfáticos.

*H3.* El desbalance extremo de clases (la clase mayoritaria supera el 89%) hace que la exactitud sea una métrica engañosa. Métricas como F1 por etiqueta y AUC-ROC son necesarias para evaluar correctamente.

*H4.* Los features textuales simples (longitud, mayúsculas, signos de exclamación) tienen poder discriminativo limitado pero no trivial. Un modelo lineal que use solo estos features logrará AUC-ROC por encima de 0.5 pero lejos de lo que un modelo con representación textual rica (TF-IDF, embeddings) alcanzaría.

*H5.* Las etiquetas *threat* e *identity_hate* son las más difíciles de predecir debido a su escasez (0.30% y 0.88%) y a su relativamente baja correlación con las demás etiquetas.

// ============================================================
// 2. EDA - VISTA GENERAL
// ============================================================
= Exploración de datos

== Vista general del dataset

El dataset contiene 159,571 comentarios provenientes de páginas de discusión de Wikipedia, etiquetados por anotadores humanos con seis categorías de toxicidad. No hay valores faltantes ni filas duplicadas. La integridad de los datos es completa.

#figure(
  image("imgs/01_overview.png", width: 100%),
  caption: [Vista general del dataset. Sin valores faltantes en ninguna columna.],
) <overview>

#figure(
  image("imgs/02_missing.png", width: 100%),
  caption: [Reporte de valores faltantes. Todas las columnas tienen 0% de missing.],
) <missing>

#figure(
  image("imgs/03_duplicates.png", width: 60%),
  caption: [Reporte de duplicados. Cero filas duplicadas en el dataset.],
) <duplicates>

// ============================================================
// 3. DISTRIBUCIÓN DE ETIQUETAS
// ============================================================
= Distribución de etiquetas

La distribución de etiquetas revela un desbalance extremo. La clase "comentario limpio" (sin ninguna etiqueta) representa 89.83% del dataset. Las seis categorías de toxicidad tienen prevalencias que van de 0.30% (*threat*) a 9.58% (*toxic*).

#figure(
  image("imgs/04_label_distribution.png", width: 100%),
  caption: [Distribución de etiquetas. Barras de conteo (izquierda) y número de etiquetas por comentario (derecha).],
) <label_dist>

Se observan tres niveles de prevalencia:

- *Extrema* (menor a 1%): *severe_toxic* (1.00%), *threat* (0.30%), *identity_hate* (0.88%). Estas etiquetas tienen muy pocos ejemplos positivos, lo que incrementa la varianza del estimador y dificulta la separación entre señal y ruido.
- *Severa* (1--5%): *insult* (4.94%).
- *Moderada* (5--10%): *toxic* (9.58%), *obscene* (5.29%).

Solo 10.17% de los comentarios tienen al menos una etiqueta tóxica. El 6.18% son multietiqueta (dos o más etiquetas simultáneamente). Esto confirma que la mayoría de los comentarios son limpios y que la clasificación enfrenta un problema de desbalance severo.

// ============================================================
// 4. CO-OCURRENCIA
// ============================================================
= Co-ocurrencia entre etiquetas

Las etiquetas no son independientes. La co-ocurrencia entre pares de etiquetas revela una estructura jerárquica fuerte.

#figure(
  image("imgs/05_label_cooccurrence.png", width: 100%),
  caption: [Porcentaje condicional P(etiqueta_col | etiqueta_row) y coeficiente Phi entre etiquetas.],
) <cooccurrence>

Los hallazgos clave son:

- *severe_toxic* es un subconjunto perfecto de *toxic*. El 100% de los comentarios *severe_toxic* también están etiquetados como *toxic*. No es una categoría independiente sino una intensificación dentro de *toxic*.
- *obscene* e *insult* co-ocurren en 72.8% de los casos de *obscene* y en 78.1% de los casos de *insult*. Su coeficiente Phi es 1.06, el par más alto del dataset. Estas dos etiquetas capturan un fenómeno compartido (el lenguaje ofensivo-dirigido) y rara vez aparecen aisladas.
- *threat* es la etiqueta más independiente. Su co-ocurrencia más alta es con *obscene* (63.0%) y su Phi promedio con las demás es 0.04. Captura un patrón conductual distinto (la amenaza explícita) que no se solapa fuertemente con el resto.
- Todos los pares de etiquetas fallan la prueba de independencia Chi-cuadrada (p < 0.001). Ningún par es estadísticamente independiente.

Esta estructura tiene implicaciones directas para el modelado. Un enfoque que entrene clasificadores binarios independientes ignora la covarianza entre etiquetas y pierde información. Un modelo multietiqueta que capture estas dependencias (por ejemplo, una red con salida sigmoide y pérdida binaria por etiqueta, o classifier chains) es preferible desde el punto de vista estadístico.

// ============================================================
// 5. FEATURES DE TEXTO
// ============================================================
= Features textuales derivados

Dado que el predictor principal es texto libre, se derivaron seis features numéricos simples a partir de cada comentario: longitud del texto (`text_len`), conteo de palabras (`word_count`), proporción de mayúsculas (`caps_ratio`), proporción de signos de exclamación (`exclaim_ratio`), proporción de signos de interrogación (`question_ratio`) y proporción de palabras únicas (`unique_word_ratio`).

#figure(
  image("imgs/06_text_features_summary.png", width: 100%),
  caption: [Resumen estadístico de variables continuas. Distribuciones con sesgo positivo extremo.],
) <feat_summary>

#figure(
  image("imgs/07_text_features_outliers.png", width: 100%),
  caption: [Reporte de outliers por IQR. Las variables ratio tienen hasta 23% de outliers.],
) <feat_outliers>

Todas las variables tienen distribuciones fuertemente sesgadas y colas pesadas. Los ratios de exclamación e interrogación tienen más de 14% de outliers por el método IQR. Esto confirma que los features no son normales y que los métodos no paramétricos son necesarios para las comparaciones de grupos.

#figure(
  image("imgs/08_text_features_by_toxicity.png", width: 100%),
  caption: [Distribuciones de features textuales comparando comentarios tóxicos vs limpios (percentil 99).],
) <feat_by_tox>

Las distribuciones muestran diferencias visibles entre grupos. Los comentarios tóxicos tienden a ser más cortos y a usar más mayúsculas y exclamaciones. Sin embargo, las distribuciones se solapan sustancialmente, lo que indica que estos features simples no separan los grupos por sí solos.

// ============================================================
// 6. PRUEBAS ESTADÍSTICAS
// ============================================================
= Pruebas estadísticas

== Correlación entre etiquetas

#figure(
  image("imgs/09_label_pearson.png", width: 80%),
  caption: [Matriz de correlación Pearson entre etiquetas.],
) <pearson>

#figure(
  image("imgs/10_label_spearman.png", width: 80%),
  caption: [Matriz de correlación Spearman entre etiquetas. Idéntica a Pearson por ser variables binarias.],
) <spearman>

Las correlaciones Pearson y Spearman coinciden porque las variables son binarias. Los pares con correlación más alta son *obscene*--*insult* (r = 0.74) y *toxic*--*obscene* (r = 0.68). Los pares con correlación más baja son *threat*--*identity_hate* (r = 0.12) y *threat*--*severe_toxic* (r = 0.12). La etiqueta *threat* es consistentemente la menos correlacionada con las demás, lo que refuerza su relativa independencia.

#figure(
  image("imgs/11_label_vif.png", width: 60%),
  caption: [Factor de Inflación de Varianza entre etiquetas. Todos los valores son menores a 3.],
) <vif>

El VIF máximo es 2.69 (*obscene*). No hay multicolinealidad problemática (el umbral convencional es VIF > 10). Esto significa que cada etiqueta aporta información distinta aunque estén correlacionadas. Un modelo puede usar las seis etiquetas como predictores sin que la colinealidad distorsione los coeficientes.

== Asociación entre features textuales y etiquetas

#figure(
  image("imgs/12_pointbiserial_text_vs_labels.png", width: 100%),
  caption: [Correlación point-biserial entre features textuales y cada etiqueta de toxicidad.],
) <pointbiserial>

El feature con mayor poder discriminativo es `caps_ratio`, que alcanza r = 0.22 con *toxic* y r = 0.17 con *severe_toxic* y *insult*. Le sigue `exclaim_ratio` con r entre 0.09 y 0.13. Los features de longitud (`text_len`, `word_count`) tienen correlaciones débiles y negativas (r alrededor de -0.05). `question_ratio` es esencialmente ruido para la mayoría de etiquetas (r < 0.03, no significativo para *severe_toxic* e *identity_hate*).

La interpretación es que los features de "énfasis" (mayúsculas, exclamaciones) capturan parcialmente la agresividad del comentario, mientras que la longitud apenas discrimina y los signos de interrogación no discriminan en absoluto.

== Normalidad

#figure(
  image("imgs/13_normality_text_features.png", width: 100%),
  caption: [Pruebas de normalidad con histogramas y Q-Q plots. Ningún feature es normal.],
) <normality>

Las pruebas de Shapiro-Wilk y D'Agostino-Pearson rechazan la normalidad para todos los features (p < 0.001). Los sesgos extremos (exclaim_ratio: 25.4, question_ratio: 23.1) y la kurtosis (hasta 1131) confirman distribuciones con colas pesadas y concentración masiva cerca de cero. Este resultado invalida el uso de t-test y ANOVA para comparar grupos, y justifica el uso de Mann-Whitney U y Kruskal-Wallis como pruebas no paramétricas.

== Comparación de grupos

#figure(
  image("imgs/14_group_comparison_effect.png", width: 100%),
  caption: [Tamaño del efecto (rank-biserial correlation) de cada feature al comparar tóxicos vs limpios.],
) <group_comp>

Las diferencias entre grupos son estadísticamente significativas (Mann-Whitney U, p < 0.05) para todos los features excepto `question_ratio` en etiquetas raras. Sin embargo, los tamaños de efecto son pequeños a moderados. `caps_ratio` tiene el efecto más grande (r_rb = 0.46 con *any_toxic*), seguido de `exclaim_ratio` (r_rb = 0.31). `text_len` y `word_count` tienen efectos débiles (r_rb entre 0.19 y 0.24).

Esto significa que las diferencias son reales pero modestas. Los features simples capturan solo una fracción de la señal discriminativa. La representación textual rica (TF-IDF, n-gramas, embeddings) es necesaria para alcanzar un rendimiento competitivo.

// ============================================================
// 7. COMPARACIÓN DE MODELOS
// ============================================================
= Comparación de modelos baseline

Se compararon LogisticRegression (con `class_weight="balanced"`) y RandomForestClassifier (100 árboles, `max_depth=10`, `class_weight="balanced"`) usando validación cruzada estratificada de 5 pliegues sobre una submuestra de 30,000 comentarios. La submuestra preserva la distribución de la variable `any_toxic`.

#figure(
  image("imgs/15_model_comparison.png", width: 100%),
  caption: [F1-score y AUC-ROC por etiqueta para LogisticRegression y RandomForest con features textuales simples.],
) <model_comp>

Los resultados confirman la hipótesis H4. Los AUC-ROC oscilan entre 0.56 (*identity_hate* con RF) y 0.79 (*severe_toxic* con RF), lo que está por encima del azar (0.5) pero lejos del rendimiento reportado por soluciones líderes en la competencia original (AUC > 0.95 con TF-IDF y modelos de deep learning). Random Forest supera a LogisticRegression en F1 para las etiquetas más prevalentes, pero LogisticRegression mantiene AUC competitivos y ofrece interpretabilidad directa de los coeficientes.

Los F1-scores son bajos en todas las etiquetas (entre 0.02 y 0.30), lo cual es esperado con features tan limitados y desbalance severo. Las etiquetas con menor prevalencia (*threat* e *identity_hate*) obtienen los F1 más bajos, confirmando la hipótesis H5.

La elección de `class_weight="balanced"` se justifica porque ajusta la función de pérdida para compensar el desbalance, evitando que el modelo ignore la clase minoritaria. La elección de 5-fold CV estratificado garantiza que cada pliegue preserve la distribución de clases. El `max_depth=10` en Random Forest previene el sobreajuste en la submuestra, aunque sacrifica capacidad de capturar interacciones profundas.

// ============================================================
// 8. DIAGNÓSTICO TOOLBELT
// ============================================================
= Diagnóstico Toolbelt

Se ejecutó el diagnóstico completo del paquete `statistical_toolbelt` sobre los features derivados con la variable `any_toxic` como target.

#figure(
  image("imgs/16_toolbelt_dataset_overview.png", width: 100%),
  caption: [Vista general del dataset de features.],
) <tb_overview>

#figure(
  image("imgs/16_toolbelt_fitness.png", width: 100%),
  caption: [Evaluación de aptitud ML. Veredicto: READY.],
) <tb_fitness>

El diagnóstico de aptitud arrojó veredicto READY sin advertencias críticas. El desbalance de clases (89.83% clase mayoritaria) está por debajo del umbral de alerta (90%). No se detectó leakage entre features y target. El ratio filas/features (5000:1) es más que suficiente.

#figure(
  image("imgs/16_toolbelt_model_recommendations.png", width: 100%),
  caption: [Recomendaciones de modelos del toolbelt. LightGBM/XGBoost como candidato principal.],
) <tb_models>

El toolbelt recomienda LightGBM/XGBoost como candidato principal (score 9.51), seguido de CatBoost (9.09). LogisticRegression aparece como baseline interpretable (7.22). Estas recomendaciones coinciden con la práctica actual en clasificación tabular, aunque para este problema específico la representación textual es más determinante que la elección del algoritmo.

// ============================================================
// 9. ESTRUCTURA JERÁRQUICA DE ETIQUETAS (Pregunta 2)
// ============================================================
= Estructura jerárquica de etiquetas

El análisis de co-ocurrencia reveló una posible estructura jerárquica entre etiquetas. La pregunta central es si las etiquetas forman un orden parcial donde la presencia de una etiqueta implica la presencia de otra.

#figure(
  image("imgs/17_label_hierarchy.png", width: 100%),
  caption: [Tabla de implicación P(B|A) y grafo de jerarquía aproximada. Aristas donde P > 0.50.],
) <hierarchy>

La única implicación estricta es *severe_toxic* → *toxic*, con cero violaciones en 159,571 comentarios. Todas las demás sub-etiquetas (*obscene*, *insult*, *threat*, *identity_hate*) co-ocurren con *toxic* entre 92.7% y 93.9% de los casos, pero cada una tiene entre 6% y 7.3% de violaciones donde la sub-etiqueta está presente sin que *toxic* lo esté.

#figure(
  image("imgs/18_implication_violations.png", width: 100%),
  caption: [Violaciones de implicaciones fuertes (P(B|A) ≥ 0.90). Barras rojas para violaciones > 5%.],
) <violations>

Las violaciones más notables son:

- *obscene*=1 y *toxic*=0: 523 casos (6.19% de los comentarios *obscene*). De estos, 317 (60.6%) solo tienen la etiqueta *obscene* sin ninguna otra. Los ejemplos muestran lenguaje vulgar ("Wanker", "damn") en contextos donde los anotadores no consideraron el comentario genéricamente tóxico.
- *insult*=1 y *toxic*=0: 533 casos (6.77%). De estos, 301 (56.5%) son etiqueta única.
- *threat*=1 y *toxic*=0: 29 casos (6.07%). De estos, 22 (75.9%) son etiqueta única. Los ejemplos incluyen advertencias de bloqueo administrativo en Wikipedia ("you will be blocked") que fueron etiquetadas como amenazas pero no como toxicidad general.

== ¿Forman un orden parcial?

Las etiquetas no forman un orden parcial estricto. La única relación de subconjunto perfecta es *severe_toxic* ⊂ *toxic*. Las demás sub-etiquetas son subconjuntos aproximados de *toxic* con tasas de violación entre 6% y 7.3%. Estas violaciones no son errores de etiquetado sino que reflejan una diferencia de interpretación entre anotadores. Algunos anotadores consideran que un comentario puede ser obsceno o amenazante sin ser genéricamente tóxico. Esto significa que la etiqueta *toxic* no es un prerrequisito formal sino una categoría de severidad que la mayoría de anotadores activa junto con las sub-etiquetas, pero no todos.

La estructura observada tiene la forma de una jerarquía aproximada donde *toxic* es la raíz, *severe_toxic* es un subconjunto perfecto, y las cuatro etiquetas restantes son hijos aproximados con ~93% de adherencia. Esta estructura puede explotarse en el modelado con dos enfoques complementarios. Primero, entrenar un modelo de detección de *toxic* como primer filtro y luego clasificar el sub-tipo entre los detectados. Segundo, usar una arquitectura con pérdida jerárquica que penalice más las predicciones que violen la estructura observada (por ejemplo, predecir *severe_toxic*=1 y *toxic*=0).

// ============================================================
// 10. ESTABILIDAD Y INTERVALOS DE CONFIANZA (Pregunta 4)
// ============================================================
= Estabilidad de los resultados

Los resultados del modelado se reportaron como medias de validación cruzada sin intervalos de confianza. Esto impide determinar si las diferencias entre LogisticRegression y Random Forest son estadísticamente significativas o si son ruido del muestreo. Para resolver esto se ejecutó el mismo pipeline de modelado con 10 semillas diferentes, cada una con su propia submuestra estratificada de 30,000 comentarios y su propia partición de 5-fold CV. Los intervalos de confianza al 95% se calcularon sobre la distribución de las 10 medias.

#figure(
  image("imgs/19_confidence_intervals.png", width: 100%),
  caption: [F1-score y AUC-ROC con intervalos de confianza al 95% sobre 10 semillas. LR en azul, RF en rojo.],
) <ci>

#figure(
  image("imgs/20_stability_cv.png", width: 100%),
  caption: [Coeficiente de variación del F1-score entre 10 semillas. Valores altos indican métrica inestable.],
) <stability>

== Resultados de estabilidad

Para las etiquetas prevalentes (*toxic*, *obscene*, *insult*), los F1-scores son estables con coeficientes de variación menores a 4%. Los AUC-ROC son aún más estables (CV < 2%). Esto significa que las medias reportadas son estimaciones confiables para estas etiquetas.

Para *threat* e *identity_hate*, los F1-scores son altamente inestables. El F1 de *threat* con Random Forest tiene un CV de 33.9%, lo que significa que la métrica puede oscilar entre 0.005 y 0.024 solo por cambiar la semilla. El AUC de *threat* con LogisticRegression tiene un IC 95% de [0.62, 0.80], un rango de 18 puntos porcentuales. Esta inestabilidad refleja que con solo 93 ejemplos positivos en la submuestra, la varianza del estimador es intrínsecamente alta.

== ¿LR vs RF es significativamente diferente?

La comparación de intervalos de confianza revela que la diferencia entre LR y RF es significativa (IC no se solapan) en F1 para *toxic*, *obscene* e *insult*, donde Random Forest supera consistentemente a LogisticRegression. En AUC-ROC, la diferencia es significativa para las mismas tres etiquetas más *severe_toxic*.

Sin embargo, para las etiquetas raras (*threat*, *identity_hate*), los IC se solapan ampliamente. La diferencia entre LR y RF en *threat* no es estadísticamente significativa ni en F1 ni en AUC. Lo mismo ocurre con *identity_hate*. Esto significa que no se puede afirmar que un modelo sea superior al otro para estas etiquetas con los datos disponibles. La recomendación práctica es no seleccionar el modelo basándose en las métricas de *threat* e *identity_hate* con features simples, porque la señal es insuficiente para distinguir entre algoritmos.

// ============================================================
// 11. ANALISIS DE SENTIMIENTO (VADER)
// ============================================================
= Analisis de sentimiento (VADER)

Se aplico VADER (Valence Aware Dictionary and sEntiment Reasoner) a los 159,571 comentarios para evaluar si la polaridad del sentimiento predice la toxicidad.

#figure(
  image("imgs/21_sentiment_distribution.png", width: 100%),
  caption: [Distribucion del compound score por grupo (toxico vs limpio) y prevalencia de toxicidad por categoria de sentimiento.],
) <sent_dist>

Los comentarios con sentimiento negativo (compound < -0.05) tienen 22.1% de prevalencia de toxicidad, frente a 5.3% en neutros y 3.9% en positivos. La asociacion es real pero imperfecta. La mayoria de los comentarios negativos son limpios (critica constructiva, frustracion legitima), y 3,075 comentarios toxicos tienen sentimiento positivo (sarcasmo hostil, amenazas encubiertas).

#figure(
  image("imgs/24_sentiment_correlation_heatmap.png", width: 100%),
  caption: [Correlacion point-biserial entre features de sentimiento y etiquetas de toxicidad.],
) <sent_corr>

La correlacion mas fuerte es entre sent_neg y toxic (r = 0.47). Las correlaciones mas debiles son entre compound y threat (r = -0.07) y entre compound e identity_hate (r = -0.10). Esto confirma que el sentimiento captura valencia pero no intencion hostil.

== Efecto de VADER en el modelo

Agregar features de VADER al baseline de texto simple mejora el AUC-ROC de LogisticRegression entre +0.15 y +0.25 segun la etiqueta. La mayor mejora se observa en severe_toxic (+0.25 AUC) y la menor en identity_hate (+0.15). La combinacion texto+VADER+EMPATH supera sistematicamente a texto+VADER sola.

#figure(
  image("imgs/27_sentiment_delta_auc.png", width: 100%),
  caption: [Incremento de AUC-ROC al agregar VADER y EMPATH al baseline de texto simple.],
) <sent_delta>

#figure(
  image("imgs/28_sentiment_confusion.png", width: 80%),
  caption: [Relacion entre sentimiento y toxicidad. Los falsos negativos (positivos y toxicos) incluyen sarcasmo y amenazas encubiertas.],
) <sent_confusion>

// ============================================================
// 12. ANALISIS DE CATEGORIAS TEMATICAS (EMPATH)
// ============================================================
= Analisis de categorias tematicas (EMPATH)

EMPATH clasifica texto en ~194 categorias tematicas (hate, aggression, violence, swearing, ridicule, etc.). A diferencia de VADER que mide valencia, EMPATH captura tema e intencion, lo cual deberia alinearse mejor con las etiquetas de toxicidad.

#figure(
  image("imgs/29_empath_top_differences.png", width: 100%),
  caption: [Categorias EMPATH con mayor diferencia de media entre comentarios toxicos y limpios.],
) <empath_top>

Las categorias mas activas en comentarios toxicos son swearing_terms (diferencia = 0.014), negative_emotion (0.012), ridicule (0.004) y hate (0.004). Esto confirma que la toxicidad se expresa predominantemente con lenguaje soez, emocion negativa, ridiculo y odio, no solo con valencia negativa.

#figure(
  image("imgs/30_empath_by_label.png", width: 100%),
  caption: [Top 5 categorias EMPATH mas discriminativas por etiqueta de toxicidad.],
) <empath_by_label>

La estructura semantica por etiqueta muestra patrones coherentes. swearing_terms domina en toxic, obscene e insult. kill y weapon son las mas discriminativas para threat. Para identity_hate, swearing_terms lidera pero hate tiene un r bajo (0.04), lo que sugiere que el odio identitario en Wikipedia se expresa mas con vulgaridad que con lexico explicito de odio.

== EMPATH vs VADER

#figure(
  image("imgs/31_empath_vs_vader.png", width: 100%),
  caption: [Poder discriminativo del mejor feature EMPATH vs el mejor feature VADER por etiqueta.],
) <empath_vs_vader>

VADER sent_neg tiene correlaciones individuales mas altas que cualquier categoria EMPATH en 5 de 6 etiquetas porque sent_neg agrega toda la valencia negativa. Sin embargo, EMPATH supera a VADER en threat (kill r=0.18 vs sent_neg r=0.12), donde el lexico tematico es mas informativo que la valencia. La combinacion de ambos conjuntos supera a cualquiera por separado.

== Efecto de EMPATH en el modelo

#figure(
  image("imgs/32_empath_model_comparison.png", width: 100%),
  caption: [AUC-ROC y F1-score por configuracion de features (texto, texto+VADER, texto+EMPATH, texto+VADER+EMPATH).],
) <empath_model>

#figure(
  image("imgs/33_empath_delta_auc.png", width: 100%),
  caption: [Incremento de AUC-ROC al agregar VADER, EMPATH o ambos al baseline de texto simple.],
) <empath_delta>

Los deltas de AUC respecto al baseline de solo texto son notablemente grandes. La combinacion texto+VADER+EMPATH alcanza AUC de 0.84 a 0.94 segun la etiqueta, un salto de +0.17 a +0.31 sobre el baseline de solo texto. EMPATH como conjunto es inferior a VADER como conjunto en 5 de 6 etiquetas, pero la combinacion de ambos supera sistematicamente a cualquiera por separado, confirmando que capturan dimensiones complementarias (tema, valencia, enfasis).

// ============================================================
// 13. CONCLUSIONES ACTUALIZADAS
// ============================================================
= Conclusiones

*H1 confirmada.* Ningún par de etiquetas es independiente (Chi-cuadrada p < 0.001 en todos los casos). *severe_toxic* es un subconjunto perfecto de *toxic*. *obscene* e *insult* co-ocurren en más del 70% de los casos. Esta estructura invalida el supuesto de independencia de clasificadores binarios separados y favorece modelos multietiqueta que capturen la covarianza entre etiquetas.

*H2 parcialmente confirmada.* Los comentarios tóxicos son más cortos (media 303 vs 404 caracteres) y usan más mayúsculas (ratio 0.111 vs 0.045). Los features de énfasis (`caps_ratio`, `exclaim_ratio`) tienen el mayor tamaño de efecto, pero las distribuciones se solapan sustancialmente. Los features simples capturan solo una fracción de la señal discriminativa.

*H3 confirmada.* La clase mayoritaria (comentarios limpios) es 89.83% del dataset. La exactitud trivialmente alcanza ~90% prediciendo siempre "no tóxico". F1-score y AUC-ROC son necesarias como métricas primarias porque son robustas al desbalance, aunque ninguna es suficiente sola. F1 depende del umbral de decisión y AUC no refleja el costo asimétrico de errores en moderación real.

*H4 confirmada con matiz.* Los baselines con features simples alcanzan AUC-ROC entre 0.56 y 0.79, lejos del rendimiento competitivo (AUC > 0.95). Los intervalos de confianza sobre 10 semillas confirman que Random Forest supera significativamente a LogisticRegression en F1 y AUC para las etiquetas prevalentes (*toxic*, *obscene*, *insult*). Sin embargo, para las etiquetas raras (*threat*, *identity_hate*) los intervalos se solapan y la diferencia no es estadísticamente significativa. El coeficiente de variación del F1 para *threat* con Random Forest es 33.9%, lo que indica que la métrica es intrínsecamente inestable con tan pocos positivos.

*H5 confirmada y cuantificada.* *threat* (0.30%) e *identity_hate* (0.88%) obtienen los F1 más bajos y tienen la menor co-ocurrencia con las demás etiquetas. El análisis de estabilidad muestra que sus métricas son tan ruidosas que no permiten distinguir entre algoritmos. La escasez extrema de ejemplos positivos (478 para *threat* en el dataset completo, 93 en la submuestra) hace que la varianza del estimador domine la señal. Soluciones como oversampling dirigido, data augmentation textual o transfer learning son necesarias para estas etiquetas.

*Estructura jerárquica.* Las etiquetas no forman un orden parcial estricto. La única implicación perfecta es *severe_toxic* → *toxic*. Las demás sub-etiquetas son subconjuntos aproximados de *toxic* con tasas de violación entre 6% y 7.3%. Las 523 excepciones de *obscene*=1 sin *toxic*=1 incluyen comentarios con lenguaje vulgar que los anotadores no consideraron genéricamente tóxicos. Esta estructura puede explotarse con una arquitectura jerárquica (detectar *toxic* primero, luego sub-tipo) o con una pérdida que penalice las violaciones de la jerarquía.

*Defensa del modelo baseline.* LogisticRegression con `class_weight="balanced"` es preferible a Random Forest como baseline inicial para las etiquetas prevalentes porque ofrece interpretabilidad directa, entrenamiento rapido y AUC comparable. Para las etiquetas raras, ningun modelo con features simples es confiable (IC amplios, CV > 10%). El modelo productivo debe ser multietiqueta con representacion textual rica.

*Sentimiento y categorias tematicas.* VADER (valencia) y EMPATH (tema) capturan dimensiones parcialmente independientes que mejoran significativamente el baseline. La combinacion texto+VADER+EMPATH alcanza AUC de 0.84 a 0.94, un salto de +0.17 a +0.31 sobre solo texto. VADER sent_neg es el feature individual mas potente para la mayoria de etiquetas, pero EMPATH kill supera a VADER en threat. Los 3,075 comentarios toxicos con sentimiento positivo ilustran la limitacion de los modelos lexicos. La combinacion completa (texto + VADER + EMPATH) sistematicamente supera a cualquier subconjunto, confirmando que las tres dimensiones son complementarias.

*Supuestos estadisticos verificados.* Las pruebas de normalidad rechazan la normalidad para todos los features (p < 0.001), lo que justifica el uso de Mann-Whitney U en lugar de t-test. Las correlaciones Pearson entre etiquetas son validas porque las variables son binarias y el tamano de muestra (N = 159,571) garantiza convergencia asintotica. Las pruebas de Chi-cuadrada son validas porque todas las celdas de las tablas de contingencia tienen frecuencias esperadas mayores a 5. El VIF maximo (2.69) indica que no hay multicolinealidad problematica entre etiquetas. Las correlaciones Spearman entre features de VADER y EMPATH son debiles (r < 0.23), confirmando que capturan dimensiones independientes.

*Limitaciones.* Este analisis usa features lexicos (texto simple, VADER, EMPATH) que no capturan contexto ni intencion. Las conclusiones sobre poder predictivo son un limite inferior del rendimiento alcanzable. El dataset proviene de paginas de discusion de Wikipedia en ingles, por lo que los patrones detectados no generalizan necesariamente a otras plataformas, idiomas o contextos culturales. La seleccion de categorias EMPATH basada en correlacion con any_toxic introduce sesgo de seleccion. El modelado de EMPATH uso una submuestra de 10,000 con 2 semillas, lo que produce intervalos mas amplios que los analisis anteriores.

#v(1em)
#text(size: 9pt, fill: luma(100))[
  Herramientas de IA utilizadas: Claude (generación de código y estructura del análisis), Typst (composición del reporte).
]
