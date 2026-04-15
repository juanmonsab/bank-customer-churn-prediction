"""
Script principal para ejecutar todo el proceso de churn prediction.
"""

from data.load_data import load_data, explore_data
from features.preprocessing import preprocess_data, remove_multicollinearity, split_and_balance
from models.train_model import train_baseline, train_lightgbm, save_models
from visualization.plots import plot_target_balance, plot_transaction_analysis

def main():
    """Corre todo el pipeline."""
    print("=== EMPEZANDO EL PIPELINE DE CHURN ===\n")

    # 1. Cargar datos
    print("1. Cargando datos...")
    df = load_data()
    info = explore_data(df)
    print(f"Datos: {info['shape'][0]} filas, {info['shape'][1]} columnas")
    print(f"Target: {info['target_distribution']}\n")

    # 2. Gráficos iniciales
    print("2. Haciendo gráficos iniciales...")
    plot_target_balance(df)
    plot_transaction_analysis(df)

    # 3. Limpiar datos
    print("3. Limpiando datos...")
    df_proc = preprocess_data(df)
    df_clean = remove_multicollinearity(df_proc)
    X_train, X_test, y_train, y_test = split_and_balance(df_clean)
    print(f"Datos listos: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} variables\n")

    # 4. Entrenar modelos
    print("4. Entrenando modelos...")
    lr_model, scaler = train_baseline(X_train, y_train, X_test, y_test)
    lgb_model = train_lightgbm(X_train, y_train, X_test, y_test)

    # 5. Guardar
    print("5. Guardando modelos...")
    save_models(lr_model, scaler, lgb_model)

    print("\n=== LISTO ===")
    print("Modelos en models/")
    print("Datos en data/processed/")

if __name__ == "__main__":
    main()