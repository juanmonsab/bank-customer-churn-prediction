"""
Funciones para limpiar y preparar los datos.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor

def preprocess_data(df):
    """
    Limpia y transforma los datos básicos.

    Args:
        df (pd.DataFrame): Dataset raw

    Returns:
        pd.DataFrame: Dataset procesado
    """
    # Sacar ID del cliente
    df_proc = df.drop('CLIENTNUM', axis=1)

    # Convertir churn a números
    df_proc['Attrition_Flag'] = df_proc['Attrition_Flag'].map({
        'Existing Customer': 0,
        'Attrited Customer': 1
    })

    # Convertir categorías a números
    le = LabelEncoder()
    for col in ['Gender', 'Marital_Status', 'Card_Category']:
        df_proc[col] = le.fit_transform(df_proc[col])

    return df_proc

def remove_multicollinearity(df, threshold=10):
    """
    Saca variables que están muy correlacionadas.

    Args:
        df (pd.DataFrame): Dataset
        threshold (float): Umbral para VIF

    Returns:
        pd.DataFrame: Dataset sin multicolinealidad
    """
    numeric_cols = df.select_dtypes(include=np.number).drop('Attrition_Flag', axis=1)
    vif_data = pd.DataFrame()
    vif_data['feature'] = numeric_cols.columns
    vif_data['VIF'] = [variance_inflation_factor(numeric_cols.values, i)
                      for i in range(len(numeric_cols.columns))]

    high_vif = vif_data[vif_data['VIF'] > threshold]['feature'].tolist()
    df_clean = df.drop(high_vif, axis=1)

    print(f"Sacadas por VIF alto: {high_vif}")
    return df_clean

def split_and_balance(df, test_size=0.2, random_state=42):
    """
    Divide los datos y balancea con SMOTE.

    Args:
        df (pd.DataFrame): Dataset procesado
        test_size (float): Proporción de test
        random_state (int): Semilla

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # One-hot encoding para categorías restantes
    X = pd.get_dummies(df.drop('Attrition_Flag', axis=1), drop_first=True)
    y = df['Attrition_Flag']

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # SMOTE en training
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print(f"Training antes SMOTE: {y_train.value_counts().to_dict()}")
    print(f"Training después SMOTE: {pd.Series(y_train_bal).value_counts().to_dict()}")

    return X_train_bal, X_test, y_train_bal, y_test

if __name__ == "__main__":
    from load_data import load_data

    df = load_data()
    df_proc = preprocess_data(df)
    df_clean = remove_multicollinearity(df_proc)
    X_train, X_test, y_train, y_test = split_and_balance(df_clean)

    print(f"Dataset final: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"Features: {X_train.shape[1]}")