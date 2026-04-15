# Carpeta src/

Aquí está todo el código del proyecto de churn.

## Estructura

```
src/
├── main.py                 # Corre todo el proceso
├── data/
│   └── load_data.py        # Cargar y ver datos
├── features/
│   └── preprocessing.py    # Limpiar y preparar datos
├── models/
│   └── train_model.py      # Entrenar modelos
└── visualization/
    └── plots.py            # Gráficos y visualizaciones
```

## Cómo usar

Para correr todo:
```bash
python src/main.py
```

Para usar partes:
```python
from src.data.load_data import load_data
from src.features.preprocessing import preprocess_data
from src.models.train_model import train_lightgbm
```

## Lo que necesitas

- pandas
- numpy
- scikit-learn
- lightgbm
- imblearn
- matplotlib
- seaborn
- statsmodels