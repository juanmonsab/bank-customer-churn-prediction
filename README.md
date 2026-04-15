# Bank Customer Churn Prediction

## El problema
El banco pierde clientes y dinero. Con 10,127 clientes en los datos, el 16% se van (1,627 personas). Necesitamos saber quiénes van a irse antes de que lo hagan.

El objetivo era armar un modelo que identifique clientes en riesgo. Así se pueden hacer acciones para retenerlos.

## Los datos
Usamos `BankChurners.csv` con:
- 10,127 registros
- 23 variables (datos personales, límites de crédito, movimientos, relación con el banco)
- Target: `Attrition_Flag` (Existing Customer: 8,500 / Attrited Customer: 1,627)

### Cómo se distribuyen los clientes

**Tarjetas**: Blue 93%, Silver 5.5%, Gold 1.1%, Platinum 0.2%

**Ingresos**: 
- Menos de 40K: 35%
- 40K-60K: 18%
- 80K-120K: 15%
- 60K-80K: 14%
- Unknown: 11%
- 120K+: 7%

**Educación**: Graduate 31%, High School 20%, Unknown 15%, Uneducated 15%, College 10%

**Estado civil**: Married 46%, Single 39%, Unknown 7%, Divorced 7%


### Números importantes sobre los clientes

- **Edad**: Promedio 46 años (26-73)
- **Meses como cliente**: Promedio 36 meses (13-56)
- **Límite de crédito**: Promedio $8,632 ($1,438-$34,516)
- **Gasto total**: Promedio $4,404 ($510-$18,484)
- **Transacciones**: Promedio 64.86 (10-139)

## Qué encontramos

**Las clases están desbalanceadas**: 16% churn, 84% no churn. Esto significa que un modelo "tonto" que diga "nadie se va" tendría 84% de accuracy. Necesitamos técnicas especiales.

**Hay variables que significan lo mismo**: Credit_Limit y Avg_Open_To_Buy están casi perfectamente correlacionadas (>0.99). Hay que eliminar una.

**Los clientes que se van son diferentes**: Gastan menos dinero, hacen menos transacciones, y están más inactivos. Eso son pistas importante.

**La mayoría usa tarjeta Blue**: El 93% usa Blue. Esto reduce variedad en esa variable.

## Análisis estadístico

Hicimos pruebas para confirmar que las diferencias entre clientes churn y no-churn son reales (no por suerte).

**Normalidad**: Los montos gastados no siguen distribución normal. Por eso usamos Mann-Whitney U en lugar de t-test.

**Diferencias entre grupos**: Mann-Whitney U confirmó que clientes que se van gastan menos (p < 0.001).

**Colinealidad**: Avg_Open_To_Buy tiene VIF infinito. Es redundante con Credit_Limit. Se eliminó.

## Preparación de datos

1. **Sacamos lo que no sirve**: CLIENTNUM es solo un identificador, nada más.
2. **Codificamos variables categóricas**: Gender con LabelEncoder, el resto (estado civil, educación, ingresos, tipo de tarjeta) con one-hot encoding.
3. **Eliminamos redundancia**: Sacamos Avg_Open_To_Buy porque es simplemente Credit_Limit - Balance.
4. **Dividimos los datos**: 80% para entrenar, 20% para test. Mantuvimos las proporciones de churn.
5. **Balanceamos con SMOTE**: El conjunto de entrenamiento tenía 16% churn. SMOTE generó clientes sintéticos hasta llegar a 50%. Así el modelo entendía ambas clases.

Resultado final:
- X_train: 16,408 muestras (después de SMOTE)
- X_test: 2,025 muestras
- 40 variables en total

## Modelos

### Regresión Logística (baseline)

Algo simple para comparar. 
- Accuracy: 89%
- Recall (detecta churn): 47%
- Precision: 62%
- Conclusión: Solo atrapa menos de la mitad de los que se van. No es suficiente.

### LightGBM (modelo principal)

Los árboles manejan mejor las categorías y el desbalance.
- Accuracy: 96%
- Recall: 85%
- Precision: 87%
- Mejora: Detecta 1,382 de 1,627 clientes que se van. Es un 80% mejor que el baseline.

### Variables que más importan

1. **Total_Trans_Amt** (0.18): Cuánto gasta el cliente
2. **Total_Trans_Ct** (0.16): Cuántas transacciones hace
3. **Months_Inactive_12_mon** (0.12): Meses sin actividad en el último año

El mensaje es claro: lo que importa es el comportamiento transaccional, no si es hombre/mujer o cuánto gana.

## Validación

No hay overfitting. Train accuracy = Test accuracy = 96%.

Lo que el modelo se equivoca:
- ~1,382 churn detectados correctamente
- ~230 false positives (dice que se van pero no)
- ~245 no detectados (se van pero el modelo no los ve)

## Qué funciona y qué no

El modelo detecta bien quién se va. Una mejora del 80% respecto del baseline.

¿Limitaciones? Sí:
- El 15% de clientes que se van, no los atrapa (falsos negativos).
- Necesita reentrenamiento cada 3-6 meses con datos nuevos.
- El 93% usa tarjeta Blue. Menos variedad para esa variable.

## Código y datos

Se usó:
- Python con pandas, numpy, scikit-learn, lightgbm
- scipy.stats y statsmodels para las pruebas estadísticas
- matplotlib y seaborn para gráficos

Archivos generados:
- `X_train.csv`, `X_test.csv` (datos procesados)
- `y_train.csv`, `y_test.csv` (targets)
- Modelos entrenados en formato pkl

## Cómo replicar esto

1. `.\.venv\Scripts\Activate.ps1` (activar entorno)
2. `pip install -r requirements.txt`
3. Ejecutar los notebooks en orden:
   - `01_Data_Understanding.ipynb`: Cómo se ven los datos
   - `02_Statistical_Analysis.ipynb`: Pruebas para confirmar diferencias
   - `03_Preprocessing.ipynb`: Limpiar y preparar
   - `04_Modeling.ipynb`: Entrenar y evaluar modelos
