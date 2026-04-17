# Reporte: Predicción de Churn en Banco de Tarjetas de Crédito

## Resumen

Este proyecto analiza la deserción de clientes en una cartera de tarjetas de crédito y desarrolla un modelo para identificar con anticipación a los clientes con mayor riesgo de salida. La meta no fue solo mejorar una métrica, sino convertir el análisis en una herramienta útil para priorizar acciones de retención.

## Qué se hizo

Partimos del archivo `BankChurners.csv`, limpiamos variables redundantes y revisamos el comportamiento de los clientes desde dos frentes: análisis descriptivo y validación estadística. Después preparamos la base procesada para modelado, comparamos una regresión logística como referencia y LightGBM como modelo principal, y finalmente validamos el desempeño en entrenamiento y en holdout externo.

## Hallazgos principales

La señal más consistente no está en la demografía sino en el uso real del producto. Los clientes que se van tienden a gastar menos, hacer menos transacciones y pasar más tiempo inactivos. Esa relación aparece tanto en el análisis exploratorio como en la importancia de variables del modelo.

La clase está desbalanceada, así que no bastaba con mirar exactitud. Por eso se priorizaron métricas como recall, precisión, F1 y AUC. En esa comparación, LightGBM quedó por encima de la regresión logística y se mantuvo como la alternativa más sólida para el caso de negocio.

## Resultado del modelado

El modelo principal muestra un desempeño fuerte en el conjunto de prueba. En el holdout externo obtiene una accuracy cercana a 96.5%, precision de 89.9%, recall de 88.0% y F1 de 88.96%. La brecha entre entrenamiento y prueba indica un sobreajuste leve, pero no suficiente como para invalidar el modelo. La lectura correcta es que el modelo sí aprende patrones reales y generaliza de forma razonable.

## Variables que más pesan

1. `Total_Trans_Amt`: cuánto gasta el cliente.
2. `Total_Trans_Ct`: cuántas transacciones hace.
3. `Months_Inactive_12_mon`: cuántos meses permanece inactivo.

El mensaje es claro: el comportamiento transaccional explica mejor la deserción que las variables puramente demográficas.

## Validación

La comparación train vs test y la curva de aprendizaje apuntan en la misma dirección: el modelo mejora con más datos y la separación entre entrenamiento y validación no se dispara. Eso sugiere que el modelo aprende patrones útiles, aunque todavía conserva un margen de mejora en generalización.

## Dashboard

El dashboard reúne el trabajo en una sola vista. Está organizado para revisar el contexto, explorar el comportamiento de los clientes, comparar modelos y simular casos concretos. La sección de campaña mensual no calcula una nueva predicción puntual; estima el impacto comercial esperado según el tamaño de la campaña y su efectividad.

## Conclusiones

1. El modelo sirve para priorizar clientes con riesgo de salida.
2. Lo que más importa es el uso del producto, no solo el perfil del cliente.
3. Hay una señal de sobreajuste leve, pero el desempeño en prueba sigue siendo bueno.
4. El dashboard convierte el resultado técnico en una lectura útil para negocio.

## Qué sigue

1. Ajustar el threshold según el costo real de perder un cliente vs contactar de más.
2. Probar una regularización más fina si se quiere reducir todavía más la brecha train-test.
3. Reentrenar periódicamente con datos nuevos.
4. Usar el dashboard como apoyo para campañas de retención más dirigidas.

## Herramientas que usamos

- Python con pandas, numpy, scikit-learn, lightgbm
- scipy.stats para las pruebas estadísticas
- matplotlib y seaborn para visualización
- PyCaret para comparar y ajustar modelos rápidamente

## Archivos generados

- Modelos: `lgbm_tuned_model.pkl`, `logistic_regression_baseline.pkl`
- Datos procesados: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- Predicciones: `predictions.csv`

En conjunto, el proyecto muestra que con datos bien preparados y una validación correcta sí se puede anticipar churn y tomar decisiones de retención más informadas.