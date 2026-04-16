# Reporte final

## Resumen
Este proyecto analiza la deserción de clientes en una cartera de tarjetas de crédito y propone un modelo para anticipar qué clientes tienen mayor riesgo de irse. El objetivo no era solo obtener una buena métrica, sino entender qué señales del comportamiento del cliente ayudan a tomar decisiones a tiempo.

## Qué se revisó
Se trabajó con el archivo original `BankChurners.csv`, se limpiaron variables que no aportaban al análisis y se preparó una base procesada para modelado. También se revisaron diferencias entre clientes que permanecen y clientes que se retiran, con especial atención a gasto, número de transacciones e inactividad.

## Hallazgos principales
La deserción no depende tanto del perfil personal como del uso del producto. Los clientes que se van tienden a gastar menos, hacer menos transacciones y pasar más tiempo inactivos. Esa es la señal que mejor se repite en el análisis.

La clase está desbalanceada, así que no bastaba con mirar exactitud. Por eso se compararon métricas como recall, precisión y AUC. En esa comparación, LightGBM quedó por encima de la regresión logística y se mantuvo como el modelo principal del proyecto.

## Resultado del modelado
El modelo principal logra detectar una parte importante de los clientes en riesgo y mantiene una lectura consistente entre entrenamiento y prueba. Eso hace que sea útil para priorizar campañas de retención, aunque todavía existen casos que no se detectan y que conviene estudiar más adelante.

## Dashboard
El dashboard reúne el resultado del trabajo en una sola vista. Está dividido en secciones para revisar el contexto, explorar el comportamiento de los clientes, comparar modelos y probar casos concretos. La idea es que el tablero sirva tanto para exponer el proyecto como para entenderlo sin depender de los notebooks.

## Cierre
En conjunto, el trabajo muestra que la deserción sí puede anticiparse con datos de comportamiento bien preparados. La parte más valiosa no es solo el modelo, sino la forma en que conecta los datos con una decisión práctica: a qué clientes conviene prestarles atención antes de que se vayan.
# Reporte Final: Predicción de Churn en Banco de Tarjetas de Crédito

## Lo que hicimos

Armamos un modelo para predecir qué clientes de un banco van a dejar de usar sus tarjetas. Usamos datos de 10,127 clientes y terminamos con un modelo que acierta el 96% de las veces y detecta el 85% de los que se van.

## Los datos

Trabajamos con BankChurners.csv:
- 10,127 registros
- 23 variables (cosas como edad, ingresos, cuánto gastan, etc.)
- El 16% de los clientes se van (churn)

## Lo que encontramos

1. **Hay más clientes que se quedan**: 84% vs 16% que se van. Eso hace que sea difícil predecir.
2. **Los que se van gastan menos**: Hacen menos transacciones y mueven menos dinero.
3. **Las variables importantes son las transaccionales**: No importa tanto si eres hombre o mujer, sino cuánto usas la tarjeta.

## Cómo lo hicimos

1. **Miramos los datos**: Vimos qué había, limpiamos lo que no servía.
2. **Hicimos pruebas estadísticas**: Confirmamos que las diferencias eran reales (no por suerte).
3. **Preparamos los datos**: Codificamos variables, eliminamos redundancias, balanceamos con SMOTE.
4. **Probamos modelos**: Regresión logística como base, LightGBM como el bueno.

## Resultados

### El modelo final (LightGBM)
- Acierta 96% de las veces
- Detecta 85% de los que se van (mejoró 80% respecto del baseline)
- Precisión 87%

### Comparado con el baseline
- El simple (regresión logística) solo detectaba 47% de los que se van.
- LightGBM es mucho mejor.

## Variables que más pesan

1. **Total_Trans_Amt**: Cuánto gasta el cliente
2. **Total_Trans_Ct**: Cuántas transacciones hace
3. **Months_Inactive_12_mon**: Meses sin usar la tarjeta

El mensaje es claro: si usas poco la tarjeta, hay riesgo de que te vayas.

## Validación

No hay overfitting. El modelo funciona igual de bien en datos nuevos que en los de entrenamiento.

## Conclusiones

1. **Funciona**: Detectamos 85% de los clientes en riesgo.
2. **Lo que importa es el uso**: No los datos personales.
3. **Se puede usar**: Para identificar clientes y hacer campañas de retención.
4. **Falta mejorar**: El 15% que no detectamos necesita más trabajo.

## Qué sigue

1. Poner el modelo en producción para puntuar clientes nuevos.
2. Revisar cada 3-6 meses y reentrenar.
3. Enfocarse en clientes con baja actividad.
4. Investigar por qué algunos se nos escapan.

## Herramientas que usamos

- Python con pandas, numpy, scikit-learn, lightgbm
- scipy.stats para estadística
- matplotlib y seaborn para gráficos
- PyCaret para probar modelos rápido

## Archivos que generamos

- Modelos: lgbm_tuned_model.pkl, logistic_regression_baseline.pkl
- Datos listos: X_train.csv, X_test.csv, etc.
- Predicciones: predictions.csv

Al final, mostramos que con datos buenos y un modelo bien elegido, se puede predecir churn y ayudar a retener clientes.