"""
Funciones para entrenar modelos de churn.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib

def train_baseline(X_train, y_train, X_test, y_test):
    """
    Entrena modelo baseline (Regresión Logística).

    Args:
        X_train, y_train, X_test, y_test: Datos

    Returns:
        LogisticRegression: Modelo entrenado
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    y_pred = lr.predict(X_test_scaled)
    print("=== BASELINE: Regresión Logística ===")
    print(classification_report(y_test, y_pred))

    return lr, scaler

def train_lightgbm(X_train, y_train, X_test, y_test):
    """
    Entrena modelo LightGBM.

    Args:
        X_train, y_train, X_test, y_test: Datos

    Returns:
        LGBMClassifier: Modelo entrenado
    """
    # Parámetros
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }

    # Dataset LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Entrenar
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(10)]
    )

    # Predicciones
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    print("=== LIGHTGBM ===")
    print(classification_report(y_test, y_pred))

    return model

def save_models(lr_model, scaler, lgb_model, path='models/'):
    """
    Guarda los modelos.

    Args:
        lr_model: Modelo baseline
        scaler: Scaler
        lgb_model: Modelo LightGBM
        path: Donde guardar
    """
    joblib.dump(lr_model, f'{path}logistic_regression_baseline.pkl')
    joblib.dump(scaler, f'{path}scaler.pkl')
    joblib.dump(lgb_model, f'{path}lgbm_tuned_model.pkl')

    print("Modelos guardados en", path)

if __name__ == "__main__":
    # Cargar datos procesados
    X_train = pd.read_csv('../data/processed/X_train.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()

    # Entrenar
    lr_model, scaler = train_baseline(X_train, y_train, X_test, y_test)
    lgb_model = train_lightgbm(X_train, y_train, X_test, y_test)

    # Guardar
    save_models(lr_model, scaler, lgb_model)