# Bank Customer Churn Prediction

**Autores:** Juan Montes Sabogal y Nicolás Almonacid Muñoz

## El problema
El banco pierde clientes y eso se traduce en pérdida de ingresos. En este caso trabajamos con 10,127 clientes y vimos que el 16% se va, es decir, 1,627 personas. La idea del proyecto fue detectar esas señales de salida con tiempo para poder actuar antes de que el cliente abandone la entidad.

## Estructura del proyecto

- `data/raw/`: archivo original del banco.
- `data/processed/`: datos listos para modelar y para probar.
- `models/`: artefactos entrenados.
- `notebooks/`: desarrollo paso a paso del análisis.
- `src/`: scripts del proyecto y dashboard.
- `reports/`: carpeta reservada para reportes finales si se necesitan.

La idea del flujo es simple: primero se revisan los datos, luego se preparan, después se entrenan los modelos y al final se muestra todo en el dashboard.

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

**Los clientes que se van son diferentes**: Gastan menos dinero, hacen menos transacciones y pasan más tiempo inactivos. Esa fue una de las señales más claras.

**La mayoría usa tarjeta Blue**: El 93% usa Blue. Esto reduce variedad en esa variable.

## Análisis estadístico

Hicimos pruebas para confirmar que las diferencias entre clientes churn y no-churn eran reales y no simples coincidencias.

**Normalidad**: Los montos gastados no siguen distribución normal. Por eso usamos Mann-Whitney U en lugar de t-test.

**Diferencias entre grupos**: Mann-Whitney U confirmó que clientes que se van gastan menos (p < 0.001).

**Colinealidad**: Avg_Open_To_Buy tiene VIF infinito. Es redundante con Credit_Limit. Se eliminó.

## Preparación de datos

1. **Sacamos lo que no servía**: `CLIENTNUM` era solo un identificador.
2. **Codificamos las variables categóricas**: `Gender` con `LabelEncoder` y el resto de categorías con one-hot encoding.
3. **Quitamos la redundancia**: `Avg_Open_To_Buy` se eliminó porque básicamente era `Credit_Limit - Balance`.
4. **Dividimos los datos**: 80% para entrenar y 20% para test, cuidando que la proporción de churn se mantuviera.
5. **Balanceamos con SMOTE**: El entrenamiento venía con 16% churn, así que se generaron ejemplos sintéticos hasta llegar a 50%.

Resultado final:
- X_train: 16,408 muestras (después de SMOTE)
- X_test: 2,025 muestras
- 40 variables en total

## Modelos

### Regresión Logística (baseline)

Se usó como punto de comparación.
- Accuracy: 89%
- Recall (detecta churn): 47%
- Precision: 62%
- Conclusión: detecta menos de la mitad de los clientes que se van, así que no alcanza por sí solo.

### LightGBM (modelo principal)

Este modelo funcionó mejor porque maneja bien las variables categóricas y el desbalance de clases.
- Accuracy: 96%
- Recall: 85%
- Precision: 87%
- Mejora: detecta 1,382 de 1,627 clientes que se van, bastante más que el baseline.

### Variables que más importan

1. **Total_Trans_Amt** (0.18): Cuánto gasta el cliente
2. **Total_Trans_Ct** (0.16): Cuántas transacciones hace
3. **Months_Inactive_12_mon** (0.12): Meses sin actividad en el último año

La lectura final fue bastante clara: pesa más el comportamiento transaccional que variables como género o ingreso.

## Validación

No vimos señales de overfitting: train accuracy y test accuracy quedaron en 96%.

Lo que el modelo se equivoca:
- ~1,382 churn detectados correctamente
- ~230 false positives (dice que se van pero no)
- ~245 no detectados (se van pero el modelo no los ve)

## Qué funciona y qué no

El modelo detecta bastante bien quién se va y mejora claramente frente al baseline.

¿Limitaciones? Sí:
- El 15% de clientes que se van, no los atrapa (falsos negativos).
- Necesita reentrenamiento cada 3-6 meses con datos nuevos.
- El 93% usa tarjeta Blue. Menos variedad para esa variable.

## Código y datos

Se trabajó con:
- Python con pandas, numpy, scikit-learn, lightgbm
- scipy.stats y statsmodels para las pruebas estadísticas
- matplotlib y seaborn para gráficos

Archivos generados:
- `X_train.csv`, `X_test.csv` (datos procesados)
- `y_train.csv`, `y_test.csv` (targets)
- Modelos entrenados en formato pkl

## Cómo replicar el proyecto

### 1. Clonar el repositorio

```powershell
git clone <URL-del-repositorio>
cd bank-customer-churn-prediction
```

Si ya tienes la carpeta descargada, entra directo al proyecto con `cd`.

### 2. Crear y activar el entorno virtual

```powershell
python -m venv venv_clean
.\venv_clean\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Si PowerShell bloquea la activación, ejecuta antes:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### 3. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 4. Ejecutar el proceso completo

Si quieres generar de nuevo los datos procesados y los modelos, corre:

```powershell
python src/main.py
```

### 5. Abrir el dashboard

Cuando ya existan los archivos en `data/processed/` y `models/`, abre la app con:

```powershell
streamlit run src/dashboard.py
```

El dashboard se abre en `http://localhost:8501`.

## Cómo usar el dashboard

El panel se organiza en estas secciones:

- Sala gerencial: resumen del problema y lectura para tomar decisiones.
- Panorama general: comportamiento de clientes activos y clientes que se fueron.
- Cómo se hizo: explicación breve del proceso seguido en los notebooks.
- Qué muestran los clientes: patrones, segmentos y señales de riesgo.
- Qué tan bien funciona el modelo: comparación entre regresión logística y LightGBM.
- Simular un cliente: evaluación de un caso concreto y lectura del riesgo estimado.

Para presentarlo, conviene empezar por el problema, seguir con los patrones de comportamiento, comparar modelos y cerrar con un caso práctico.

## Resumen del panel

En pocas palabras, el panel resume el problema, la exploración de datos, la comparación de modelos y la simulación de casos concretos.
