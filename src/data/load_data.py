"""
Funciones para cargar y ver los datos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath='data/raw/BankChurners.csv'):
    """
    Carga el csv y saca las últimas 2 columnas vacías.

    Args:
        filepath (str): Ruta al csv

    Returns:
        pd.DataFrame: Los datos
    """
    df = pd.read_csv(filepath).iloc[:, :-2]
    return df

def explore_data(df):
    """
    Mira qué hay en el dataset.

    Args:
        df (pd.DataFrame): Dataset

    Returns:
        dict: Info básica
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'nulls': df.isnull().sum().to_dict(),
        'target_distribution': df['Attrition_Flag'].value_counts().to_dict()
    }
    return info

def plot_target_distribution(df):
    """
    Gráfico del balance de clases.

    Args:
        df (pd.DataFrame): Dataset con Attrition_Flag
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Attrition_Flag', data=df)
    plt.title('Clientes Activos vs Churn')
    plt.show()

if __name__ == "__main__":
    df = load_data()
    info = explore_data(df)
    print("Info del dataset:")
    for key, value in info.items():
        print(f"{key}: {value}")

    plot_target_distribution(df)