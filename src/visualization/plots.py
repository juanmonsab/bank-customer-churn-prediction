"""
Funciones para hacer gráficos del proyecto.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def plot_target_balance(df):
    """
    Grafica cuántos clientes se van y cuántos se quedan.

    Args:
        df (pd.DataFrame): Dataset con Attrition_Flag
    """
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='Attrition_Flag', data=df)
    plt.title('Clientes Activos vs Churn')
    plt.xlabel('Estado del Cliente')
    plt.ylabel('Cantidad')

    # Agregar porcentajes
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.show()

def plot_transaction_analysis(df):
    """
    Grafica cuánto gastan y cuántas transacciones hacen los clientes.

    Args:
        df (pd.DataFrame): Dataset
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Monto transaccionado
    sns.boxplot(x='Attrition_Flag', y='Total_Trans_Amt', data=df, ax=axes[0])
    axes[0].set_title('Cuánto gastan vs Churn')
    axes[0].set_xlabel('Estado del Cliente')
    axes[0].set_ylabel('Monto Total ($)')

    # Cantidad de transacciones
    sns.boxplot(x='Attrition_Flag', y='Total_Trans_Ct', data=df, ax=axes[1])
    axes[1].set_title('Cuántas transacciones vs Churn')
    axes[1].set_xlabel('Estado del Cliente')
    axes[1].set_ylabel('Número de Transacciones')

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title="Matriz de Confusión"):
    """
    Grafica la matriz de confusión.

    Args:
        y_true: Valores reales
        y_pred: Predicciones
        title: Título del gráfico
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, title="Curva ROC"):
    """
    Grafica la curva ROC.

    Args:
        y_true: Valores reales
        y_pred_proba: Probabilidades predichas
        title: Título del gráfico
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_feature_importance(model, feature_names, top_n=10):
    """
    Grafica qué variables son más importantes para el modelo.

    Args:
        model: Modelo entrenado
        feature_names: Nombres de las features
        top_n: Número de features a mostrar
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        # Para LightGBM
        importance = model.feature_importance()

    feat_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feat_importance)
    plt.title(f'Top {top_n} Variables Más Importantes')
    plt.xlabel('Importancia')
    plt.ylabel('Variable')
    plt.show()

if __name__ == "__main__":
    # Ejemplo de uso
    from data.load_data import load_data

    df = load_data()
    plot_target_balance(df)
    plot_transaction_analysis(df)