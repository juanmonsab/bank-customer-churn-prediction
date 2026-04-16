from pathlib import Path
import warnings
from contextlib import contextmanager

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)

try:
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:  # pragma: no cover
    InconsistentVersionWarning = Warning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def ensure_pycaret_compat() -> None:
    """Patch sklearn internals expected by older PyCaret pipelines."""
    try:
        import sklearn.utils
    except Exception:
        return

    if not hasattr(sklearn.utils, "_print_elapsed_time"):
        @contextmanager
        def _print_elapsed_time(*args, **kwargs):
            yield

        sklearn.utils._print_elapsed_time = _print_elapsed_time

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "BankChurners.csv"
X_TRAIN_PATH = ROOT / "data" / "processed" / "X_train.csv"
X_TEST_PATH = ROOT / "data" / "processed" / "X_test.csv"
Y_TRAIN_PATH = ROOT / "data" / "processed" / "y_train.csv"
Y_TEST_PATH = ROOT / "data" / "processed" / "y_test.csv"
LOGISTIC_PATH = ROOT / "models" / "logistic_regression_baseline.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
LGB_MODEL_PATH = ROOT / "models" / "lgbm_tuned_model.pkl"

CHURN_LABEL = "Attrited Customer"
ACTIVE_LABEL = "Existing Customer"
TARGET_COLUMN = "Attrition_Flag"

COLORS = {
    "bg": "#f5f7fb",
    "navy": "#0f172a",
    "slate": "#334155",
    "muted": "#64748b",
    "teal": "#15803d",
    "cyan": "#06b6d4",
    "amber": "#f59e0b",
    "coral": "#dc2626",
    "rose": "#b91c1c",
    "mint": "#16a34a",
    "line": "#d8e1ef",
}

NOTEBOOK_COMPARE_METRICS = {
    "Accuracy": 0.9831,
    "AUC": 0.9986,
    "Recall": 0.9884,
    "Precisión": 0.9780,
    "F1": 0.9832,
    "Kappa": 0.9662,
    "MCC": 0.9662,
    "Tiempo": 0.3820,
}

NOTEBOOK_TUNED_METRICS = {
    "Accuracy": 0.9826,
    "Precisión": 0.9754,
    "Recall": 0.9902,
    "F1": 0.9827,
}

sns.set_theme(style="whitegrid")

st.set_page_config(
    page_title="Análisis de Deserción de Clientes",
    layout="wide",
    initial_sidebar_state="expanded",
)

SECTION_ORDER = [
    "Sala gerencial",
    "Panorama general",
    "Cómo se hizo",
    "Qué muestran los clientes",
    "Qué tan bien funciona el modelo",
    "Simular un cliente",
    "Conclusiones finales",
]


def go_to_section(section_name: str) -> None:
    st.session_state["dashboard_section"] = section_name
    st.session_state["dashboard_section_radio"] = section_name


def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(6,182,212,0.12), transparent 26%),
                radial-gradient(circle at top right, rgba(249,115,22,0.10), transparent 24%),
                linear-gradient(180deg, #f8fbff 0%, {COLORS['bg']} 42%, #eef4fb 100%);
            color: {COLORS['navy']};
        }}
        [data-testid="stHeader"] {{ background: rgba(255,255,255,0); }}
        [data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0f172a 0%, #172554 100%); }}
        [data-testid="stSidebar"] * {{ color: #e2e8f0 !important; }}
        .hero {{
            padding: 2rem 2.2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(8,47,73,0.92));
            color: white;
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.20);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1.2rem;
        }}
        .hero h1 {{ margin: 0; font-size: 2.4rem; font-weight: 800; line-height: 1.05; }}
        .hero p {{ margin: 0.7rem 0 0 0; font-size: 1.01rem; line-height: 1.55; color: rgba(255,255,255,0.84); max-width: 920px; }}
        .hero-badges {{ display: flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 1rem; }}
        .hero-badge {{ display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.42rem 0.8rem; border-radius: 999px; background: rgba(255,255,255,0.12); color: rgba(255,255,255,0.95); border: 1px solid rgba(255,255,255,0.14); font-size: 0.86rem; font-weight: 700; }}
        .hero-actions {{ display: flex; flex-wrap: wrap; gap: 0.65rem; margin-top: 1.1rem; }}
        .hero-actions .stButton > button {{
            background: rgba(255,255,255,0.12) !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            color: white !important;
            border-radius: 999px !important;
            padding: 0.6rem 1rem !important;
        }}
        .hero-actions .stButton > button:hover {{
            background: rgba(255,255,255,0.18) !important;
        }}
        .section-title {{ font-size: 1.35rem; font-weight: 800; color: {COLORS['navy']}; margin: 0.35rem 0 0.8rem 0; }}
        .metric-card {{ background: rgba(255,255,255,0.92); border: 1px solid {COLORS['line']}; border-radius: 22px; padding: 1.05rem 1.1rem; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06); }}
        .metric-label {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: {COLORS['muted']}; margin-bottom: 0.3rem; font-weight: 700; }}
        .metric-value {{ font-size: 2rem; font-weight: 800; color: {COLORS['navy']}; line-height: 1; }}
        .metric-caption {{ margin-top: 0.55rem; font-size: 0.92rem; color: {COLORS['slate']}; }}
        .stat-strip {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.9rem; margin-top: 0.9rem; }}
        .glass-card {{ background: rgba(255,255,255,0.96); border: 1px solid {COLORS['line']}; border-radius: 22px; padding: 1rem 1.05rem; box-shadow: 0 12px 30px rgba(15,23,42,0.06); }}
        .glass-card .kpi-title {{ color: {COLORS['muted']}; font-size: 0.84rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; }}
        .glass-card .kpi-value {{ color: {COLORS['navy']}; font-size: 2.1rem; font-weight: 900; line-height: 1; margin-top: 0.3rem; }}
        .glass-card .kpi-note {{ color: {COLORS['slate']}; margin-top: 0.45rem; font-size: 0.93rem; line-height: 1.45; }}
        .insight-card {{ background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(255,255,255,0.92)); border: 1px solid {COLORS['line']}; border-radius: 22px; padding: 1.1rem 1.15rem; min-height: 170px; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05); }}
        .insight-card h4 {{ margin: 0 0 0.55rem 0; font-size: 1rem; color: {COLORS['navy']}; }}
        .insight-card p {{ margin: 0; color: {COLORS['slate']}; line-height: 1.5; font-size: 0.95rem; }}
        .risk-box {{ padding: 1rem 1.1rem; border-radius: 20px; color: white; box-shadow: 0 16px 35px rgba(15,23,42,0.10); }}
        .small-note {{ color: {COLORS['muted']}; font-size: 0.92rem; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 0.35rem; }}
        .stTabs [data-baseweb="tab"] {{
            background: rgba(255,255,255,0.92);
            border-radius: 999px;
            border: 1px solid {COLORS['line']};
            padding: 0.45rem 0.85rem;
            color: {COLORS['navy']} !important;
        }}
        .stTabs [data-baseweb="tab"] * {{ color: {COLORS['navy']} !important; }}
        .stTabs [aria-selected="true"] {{ background: {COLORS['navy']} !important; color: white !important; border-color: {COLORS['navy']} !important; }}
        .stTabs [aria-selected="true"] * {{ color: white !important; }}
        .story-card {{ background: rgba(255,255,255,0.9); border: 1px solid {COLORS['line']}; border-radius: 20px; padding: 1rem 1.1rem; }}
        .story-card h4 {{ margin: 0 0 0.45rem 0; color: {COLORS['navy']}; font-size: 1.02rem; }}
        .story-card p {{ margin: 0; color: {COLORS['slate']}; line-height: 1.55; font-size: 0.95rem; }}
        .big-panel {{ background: linear-gradient(180deg, rgba(255,255,255,1), rgba(248,250,252,0.98)); border: 1px solid {COLORS['line']}; border-radius: 26px; padding: 1.4rem 1.5rem; box-shadow: 0 12px 30px rgba(15,23,42,0.08); }}
        .big-panel h3 {{ margin: 0 0 0.65rem 0; color: {COLORS['navy']}; font-size: 1.55rem; }}
        .big-panel p {{ margin: 0; color: {COLORS['slate']}; line-height: 1.7; font-size: 1.08rem; }}
        .conclusion-card {{ background: linear-gradient(135deg, rgba(255,255,255,1), rgba(245,247,251,0.96)); border: 1px solid {COLORS['line']}; border-left: 8px solid {COLORS['teal']}; border-radius: 24px; padding: 1.3rem 1.35rem; box-shadow: 0 14px 35px rgba(15,23,42,0.08); min-height: 240px; }}
        .conclusion-card.red {{ border-left-color: {COLORS['coral']}; }}
        .conclusion-card h4 {{ margin: 0 0 0.7rem 0; font-size: 1.3rem; color: {COLORS['navy']}; }}
        .conclusion-card p {{ margin: 0; font-size: 1.08rem; line-height: 1.75; color: {COLORS['slate']}; }}
        .conclusion-number {{ font-size: 2.4rem; font-weight: 900; line-height: 1; margin-bottom: 0.7rem; color: {COLORS['navy']}; }}
        label[data-testid="stWidgetLabel"] p {{ color: {COLORS['navy']} !important; font-weight: 700 !important; font-size: 1.05rem !important; }}
        div[data-baseweb="select"] > div {{ background: #ffffff !important; color: {COLORS['navy']} !important; border: 1px solid {COLORS['line']} !important; }}
        div[data-baseweb="select"] * {{ color: {COLORS['navy']} !important; }}
        div[data-baseweb="base-input"] {{ background: #ffffff !important; border: 1px solid {COLORS['line']} !important; }}
        div[data-baseweb="base-input"] input {{ color: {COLORS['navy']} !important; }}
        .stSlider p {{ color: {COLORS['navy']} !important; }}
        .stCaption {{ color: {COLORS['slate']} !important; font-size: 0.98rem !important; }}
        .stButton > button, .stFormSubmitButton > button {{
            background: linear-gradient(135deg, {COLORS['navy']}, #1e3a8a) !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 0.7rem 1.2rem !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
        }}
        .stButton > button *, .stFormSubmitButton > button * {{ color: white !important; }}
        .stButton > button:hover, .stFormSubmitButton > button:hover {{
            background: linear-gradient(135deg, #1e3a8a, {COLORS['navy']}) !important;
        }}
        .stDataFrame, .stTable {{ background: white; border-radius: 18px; overflow: hidden; }}
        .stRadio label p, .stTabs [data-baseweb="tab"] p {{
            font-size: 1rem !important;
            font-weight: 700 !important;
        }}
        .section-title {{ font-size: 1.7rem; }}
        .metric-value {{ font-size: 2.35rem; }}
        .metric-caption {{ font-size: 1rem; }}
        .hero h1 {{ font-size: 2.8rem; }}
        .hero p {{ font-size: 1.08rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_raw_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH).iloc[:, :-2].copy()
    df["Attrition_Binary"] = (df[TARGET_COLUMN] == CHURN_LABEL).astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = sanitize_target(pd.read_csv(Y_TRAIN_PATH))
    y_test = sanitize_target(pd.read_csv(Y_TEST_PATH))
    return X_train, X_test, y_train, y_test


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[object, object, object]:
    lr_model = joblib.load(LOGISTIC_PATH)
    scaler = joblib.load(SCALER_PATH)
    ensure_pycaret_compat()
    lgb_model = joblib.load(LGB_MODEL_PATH)
    return lr_model, scaler, lgb_model


def label_encoder_mapping(df: pd.DataFrame, column: str) -> dict:
    unique_values = sorted(df[column].dropna().unique())
    return {value: index for index, value in enumerate(unique_values)}


def sanitize_target(values) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("El target debe tener una sola columna.")
        values = values.iloc[:, 0]
    series = pd.Series(np.ravel(values), name="target")
    series = pd.to_numeric(series, errors="coerce")
    if series.isna().any():
        raise ValueError("El target contiene valores no numericos.")
    return series.astype(int)


def canonicalize_feature_name(name: str) -> str:
    return name.replace(" - ", "_-_").replace(" ", "_")


def align_raw_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing = [feature for feature in feature_names if feature not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para el modelo: {missing}")
    return df[feature_names]

def get_lightgbm_feature_names(model, fallback_columns: list[str]) -> list[str]:
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        names = [name for name in list(model.feature_names_in_) if name != TARGET_COLUMN]
        if names:
            return names
    if hasattr(model, "steps") and model.steps:
        final_model = model.steps[-1][1]
        if hasattr(final_model, "feature_name_") and final_model.feature_name_ is not None:
            return list(final_model.feature_name_)
    return [col for col in fallback_columns if col != TARGET_COLUMN]


def preprocess_input(values: dict, model_columns: list[str], label_maps: dict, feature_means: pd.Series) -> pd.DataFrame:
    row = feature_means.copy().to_frame().T
    row = row.reindex(columns=model_columns, fill_value=0)

    row["Gender"] = label_maps["Gender"][values["Gender"]]
    row["Marital_Status"] = label_maps["Marital_Status"][values["Marital_Status"]]
    row["Card_Category"] = label_maps["Card_Category"][values["Card_Category"]]

    education_cols = [col for col in model_columns if col.startswith("Education_Level_")]
    income_cols = [col for col in model_columns if col.startswith("Income_Category_")]

    for col in education_cols:
        row[col] = 0
    education_key = f"Education_Level_{values['Education_Level']}"
    if education_key in row.columns:
        row[education_key] = 1

    for col in income_cols:
        row[col] = 0
    income_key = f"Income_Category_{values['Income_Category']}"
    if income_key in row.columns:
        row[income_key] = 1

    for numeric in [
        "Dependent_count",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Total_Trans_Amt",
        "Avg_Utilization_Ratio",
    ]:
        row[numeric] = values[numeric]

    return row.astype(float)


def format_int(value: float) -> str:
    return f"{int(round(value)):,}"


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def render_metric_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_strip(summary: dict) -> None:
    delta_behavior = summary["churn_trans_ct"] - summary["active_trans_ct"]
    delta_inactive = summary["churn_inactive_months"] - summary["active_inactive_months"]
    cols = st.columns(4)
    items = [
        ("Tasa de churn", f"{summary['churn_rate']:.1%}", "Base histórica de deserción"),
        ("Gap transaccional", f"{delta_behavior:.1f}", "Menos transacciones en clientes que se van"),
        ("Gap inactividad", f"{delta_inactive:.2f}", "Más meses sin movimiento"),
        ("Modelo principal", "LightGBM", "Ajustado y guardado en lgbm_tuned_model.pkl"),
    ]
    for column, (title, value, note) in zip(cols, items):
        with column:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="kpi-title">{title}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def build_portfolio_summary(df: pd.DataFrame) -> dict:
    churn_rate = df["Attrition_Binary"].mean()
    active_df = df[df[TARGET_COLUMN] == ACTIVE_LABEL]
    churn_df = df[df[TARGET_COLUMN] == CHURN_LABEL]
    return {
        "total_customers": len(df),
        "churn_count": int(df["Attrition_Binary"].sum()),
        "churn_rate": churn_rate,
        "total_transacted": float(df["Total_Trans_Amt"].sum()),
        "avg_tenure": float(df["Months_on_book"].mean()),
        "avg_relationships": float(df["Total_Relationship_Count"].mean()),
        "active_trans_amt": float(active_df["Total_Trans_Amt"].mean()),
        "churn_trans_amt": float(churn_df["Total_Trans_Amt"].mean()),
        "active_trans_ct": float(active_df["Total_Trans_Ct"].mean()),
        "churn_trans_ct": float(churn_df["Total_Trans_Ct"].mean()),
        "active_inactive_months": float(active_df["Months_Inactive_12_mon"].mean()),
        "churn_inactive_months": float(churn_df["Months_Inactive_12_mon"].mean()),
    }


def plot_class_balance(df: pd.DataFrame) -> plt.Figure:
    counts = df[TARGET_COLUMN].value_counts().reindex([ACTIVE_LABEL, CHURN_LABEL])
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.pie(
        counts.values,
        labels=["Permanecen", "Se retiran"],
        autopct="%1.1f%%",
        startangle=90,
        colors=[COLORS["mint"], COLORS["coral"]],
        wedgeprops={"width": 0.42, "edgecolor": "white"},
        textprops={"color": COLORS["navy"], "fontsize": 11, "fontweight": "bold"},
    )
    ax.set_title("Balance de la cartera", fontsize=14, fontweight="bold", color=COLORS["navy"])
    return fig


def plot_behavior_gap(df: pd.DataFrame) -> plt.Figure:
    grouped = (
        df.groupby(TARGET_COLUMN)[["Total_Trans_Amt", "Total_Trans_Ct", "Months_Inactive_12_mon", "Avg_Utilization_Ratio"]]
        .mean()
        .rename(
            columns={
                "Total_Trans_Amt": "Monto transado",
                "Total_Trans_Ct": "Cantidad transacciones",
                "Months_Inactive_12_mon": "Meses inactivo",
                "Avg_Utilization_Ratio": "Uso de credito",
            }
        )
    )
    normalized = grouped / grouped.max()
    plot_df = normalized.reset_index().melt(id_vars=TARGET_COLUMN, var_name="Indicador", value_name="Nivel")
    plot_df[TARGET_COLUMN] = plot_df[TARGET_COLUMN].replace(
        {ACTIVE_LABEL: "Permanecen", CHURN_LABEL: "Se retiran"}
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    sns.barplot(data=plot_df, x="Indicador", y="Nivel", hue=TARGET_COLUMN, palette=[COLORS["mint"], COLORS["coral"]], ax=ax)
    ax.set_title("Huella conductual: clientes que permanecen vs clientes que se retiran", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Nivel relativo")
    ax.tick_params(axis="x", rotation=10)
    ax.legend(title="")
    return fig


def plot_trans_scatter(df: pd.DataFrame) -> plt.Figure:
    sample_df = df.sample(min(len(df), 1800), random_state=42)
    sample_df[TARGET_COLUMN] = sample_df[TARGET_COLUMN].replace(
        {ACTIVE_LABEL: "Permanecen", CHURN_LABEL: "Se retiran"}
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    sns.scatterplot(
        data=sample_df,
        x="Total_Trans_Ct",
        y="Total_Trans_Amt",
        hue=TARGET_COLUMN,
        palette=[COLORS["mint"], COLORS["coral"]],
        alpha=0.68,
        s=58,
        edgecolor="none",
        ax=ax,
    )
    ax.set_title("Actividad transaccional por cliente", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total de transacciones")
    ax.set_ylabel("Monto total transado")
    ax.legend(title="")
    return fig


def plot_inactivity_heatmap(df: pd.DataFrame) -> plt.Figure:
    heatmap_df = (
        df.groupby(["Months_Inactive_12_mon", "Contacts_Count_12_mon"])["Attrition_Binary"]
        .mean()
        .mul(100)
        .round(1)
        .unstack()
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".1f",
        cmap=sns.light_palette(COLORS["rose"], as_cmap=True),
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "% de deserción"},
        ax=ax,
    )
    ax.set_title("Deserción según inactividad y número de contactos", fontsize=14, fontweight="bold")
    ax.set_xlabel("Contactos últimos 12 meses")
    ax.set_ylabel("Meses inactivo")
    return fig

def plot_segment_risk(df: pd.DataFrame, column: str, title: str) -> plt.Figure:
    segment_df = (
        df.groupby(column)
        .agg(Clientes=("Attrition_Binary", "size"), Churn=("Attrition_Binary", "mean"))
        .sort_values("Churn", ascending=False)
        .reset_index()
    )
    segment_df["Churn"] = segment_df["Churn"] * 100
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(data=segment_df, x="Churn", y=column, color=COLORS["coral"], ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("% de deserción")
    ax.set_ylabel("")
    return fig


def build_priority_segments(df: pd.DataFrame) -> pd.DataFrame:
    segment_df = (
        df.groupby(["Income_Category", "Card_Category"])
        .agg(
            Clientes=("Attrition_Binary", "size"),
            Churn_pct=("Attrition_Binary", lambda values: values.mean() * 100),
            Inactividad_prom=("Months_Inactive_12_mon", "mean"),
            Transacciones_prom=("Total_Trans_Ct", "mean"),
        )
        .reset_index()
    )
    segment_df = segment_df[segment_df["Clientes"] >= 80].sort_values(["Churn_pct", "Clientes"], ascending=[False, False])
    segment_df["Churn_pct"] = segment_df["Churn_pct"].round(2)
    segment_df["Inactividad_prom"] = segment_df["Inactividad_prom"].round(2)
    segment_df["Transacciones_prom"] = segment_df["Transacciones_prom"].round(1)
    return segment_df.head(8)


def build_action_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    rules = [
        {
            "Frente": "Activacion temprana",
            "Filtro": "4 o más meses inactivo y menos de 35 transacciones",
            "Clientes": int(
                df[
                    (df["Months_Inactive_12_mon"] >= 4)
                    & (df["Total_Trans_Ct"] < 35)
                ].shape[0]
            ),
            "Accion": "Campaña de uso con incentivo inmediato y seguimiento de 30 días.",
        },
        {
            "Frente": "Relaciones en riesgo",
            "Filtro": "Solo 1-2 relaciones activas",
            "Clientes": int(df[df["Total_Relationship_Count"] <= 2].shape[0]),
            "Accion": "Oferta de vinculación cruzada para aumentar permanencia.",
        },
        {
            "Frente": "Fricción comercial",
            "Filtro": "3 o más contactos y bajo monto transado",
            "Clientes": int(
                df[
                    (df["Contacts_Count_12_mon"] >= 3)
                    & (df["Total_Trans_Amt"] < df["Total_Trans_Amt"].median())
                ].shape[0]
            ),
            "Accion": "Revisar experiencia, soporte y razón de contacto para evitar desgaste.",
        },
    ]
    return pd.DataFrame(rules)


def build_manager_brief(summary: dict, artifacts: dict) -> dict:
    avg_value_per_customer = summary["total_transacted"] / max(summary["total_customers"], 1)
    monthly_churn_volume = summary["churn_count"] / 12
    model_recall = artifacts["report_lgb"]["1"]["recall"]
    detected_monthly = monthly_churn_volume * model_recall
    return {
        "avg_value_per_customer": avg_value_per_customer,
        "monthly_churn_volume": monthly_churn_volume,
        "model_recall": model_recall,
        "detected_monthly": detected_monthly,
    }


def render_executive_room(raw_df: pd.DataFrame, summary: dict, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, lr_model, scaler, lgb_model) -> None:
    artifacts = compute_model_artifacts(X_train, X_test, y_test, lr_model, scaler, lgb_model)
    brief = build_manager_brief(summary, artifacts)

    st.markdown('<div class="section-title">Sala gerencial</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="big-panel">
            <h3>Estado actual de la cartera</h3>
            <p>
                La cartera analizada presenta una deserción histórica de <strong>{summary['churn_rate']:.1%}</strong>.
                En términos operativos, esto equivale a aproximadamente <strong>{brief['monthly_churn_volume']:.0f}</strong> clientes en riesgo por mes.
                El modelo LightGBM identifica cerca del <strong>{brief['model_recall']:.1%}</strong> de esos casos con anticipación,
                lo que permite priorizar acciones de retención con mayor oportunidad.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_cols = st.columns(4)
    kpi_data = [
        ("Clientes en cartera", format_int(summary["total_customers"]), "Base total analizada"),
        ("Clientes en riesgo/mes", f"{brief['monthly_churn_volume']:.0f}", "Promedio mensual estimado"),
        ("Casos detectables/mes", f"{brief['detected_monthly']:.0f}", "Con el recall actual del modelo"),
        ("Valor medio por cliente", format_currency(brief["avg_value_per_customer"]), "Proxy para priorizar impacto"),
    ]
    for col, (title, value, note) in zip(kpi_cols, kpi_data):
        with col:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="kpi-title">{title}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Escenario de campaña mensual (negocio)</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="story-card">
            <h4>¿Para qué sirve esta parte?</h4>
            <p>
                Esta herramienta <strong>no</strong> calcula una nueva predicción del modelo para un cliente puntual.
                Su propósito es distinto: ayuda a estimar, a nivel gerencial, cuánto valor podría protegerse en un mes
                según el tamaño de la campaña y su efectividad esperada.
            </p>
            <p>
                En síntesis: el modelo identifica a quién priorizar y este bloque proyecta el impacto comercial de intervenir esos casos.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    targeted_customers = sim_col1.slider(
        "Clientes contactados al mes",
        min_value=20,
        max_value=max(200, int(brief["detected_monthly"] * 2)),
        value=max(20, int(brief["detected_monthly"])),
        step=5,
    )
    retention_effectiveness = sim_col2.slider(
        "Efectividad esperada de retención",
        min_value=0.05,
        max_value=0.70,
        value=0.25,
        step=0.01,
    )
    avg_saved_value = sim_col3.number_input(
        "Valor promedio preservado por cliente",
        min_value=50.0,
        max_value=50000.0,
        value=float(round(brief["avg_value_per_customer"], 2)),
        step=50.0,
    )

    customers_saved = targeted_customers * retention_effectiveness
    monthly_value_saved = customers_saved * avg_saved_value

    sim_res_1, sim_res_2, sim_res_3 = st.columns(3)
    with sim_res_1:
        render_metric_card("Clientes retenidos (estimado)", f"{customers_saved:.1f}", "Resultado esperado del esfuerzo mensual")
    with sim_res_2:
        render_metric_card("Valor mensual protegido", format_currency(monthly_value_saved), "Proyección de negocio")
    with sim_res_3:
        render_metric_card("Valor anual protegido", format_currency(monthly_value_saved * 12), "Escenario anual simple")

    st.caption("Lectura recomendada: utilice este bloque para comparar escenarios de campaña y definir metas realistas del equipo comercial.")

    left, right = st.columns([1.05, 0.95])
    with left:
        action_df = build_action_recommendations(raw_df)
        st.caption("Plan sugerido de acciones: cada fila representa un frente de trabajo para el equipo comercial y de fidelización.")
        st.dataframe(action_df, use_container_width=True, height=235)
    with right:
        st.markdown(
            """
            <div class="story-card">
                <h4>Cómo usar esta sala</h4>
                <p>
                    1) Defina cuántos clientes se pueden intervenir por mes. 2) Ajuste una efectividad de campaña realista.
                    3) Revise el valor protegido y priorice los frentes con mayor retorno esperado. Esta sección está diseñada
                    para apoyar decisiones, no solo para mostrar indicadores.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            """
            <div class="story-card">
                <h4>Mensaje clave</h4>
                <p>
                    Una intervención oportuna sobre clientes de alto riesgo puede generar un impacto económico relevante,
                    incluso con campañas de alcance moderado. El modelo permite priorizar con criterio a quién contactar primero.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Glosario</div>', unsafe_allow_html=True)
    gloss_cols = st.columns(3)
    glossary = [
        ("Recall", "De todos los clientes que realmente desertan, qué proporción logra detectar el modelo."),
        ("Precisión", "De los clientes clasificados como riesgo, qué proporción efectivamente era riesgo real."),
        ("AUC", "Capacidad global del modelo para separar clientes estables de clientes en riesgo."),
    ]
    for col, (term, definition) in zip(gloss_cols, glossary):
        with col:
            st.markdown(
                f"""
                <div class="story-card">
                    <h4>{term}</h4>
                    <p>{definition}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_methodology() -> None:
    st.markdown('<div class="section-title">Metodología del proyecto</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="big-panel">
                <h3>Ruta de trabajo</h3>
                <p>
                    El proyecto se desarrolló en cuatro etapas. Primero se revisó la estructura general del conjunto de datos.
                    Después se realizó el análisis estadístico para validar si las diferencias entre grupos eran significativas.
                    Luego se prepararon los datos mediante codificación, eliminación de variables redundantes, partición de entrenamiento y prueba,
                    y balanceo de clases con SMOTE. Finalmente, se compararon varios modelos y se seleccionó LightGBM como alternativa principal.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="big-panel">
                <h3>Lo que se hizo en los notebooks</h3>
                <p>
                    <strong>01_Data_Understanding:</strong> revisión de variables, distribución de clientes y descripción general.<br>
                    <strong>02_Statistical_Analysis:</strong> pruebas para validar diferencias entre grupos y análisis de colinealidad.<br>
                    <strong>03_Preprocessing:</strong> limpieza, codificación, división de datos y aplicación de SMOTE en entrenamiento.<br>
                    <strong>04_Modeling:</strong> comparación de modelos, ajuste de LightGBM, revisión de métricas y generación de predicciones.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        """
        <div class="story-card">
            <h4>Decisiones metodológicas importantes</h4>
            <p>
                Se retiró <strong>CLIENTNUM</strong> por ser un identificador, se eliminó <strong>Avg_Open_To_Buy</strong> por su redundancia con
                <strong>Credit_Limit</strong>, se mantuvo la partición estratificada 80/20 y el balanceo con SMOTE se aplicó solo sobre entrenamiento.
                Estas decisiones permitieron trabajar con una base más consistente y, al mismo tiempo, atender el desbalance del problema.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_project_conclusions(summary: dict) -> None:
    st.markdown('<div class="section-title">Conclusiones del proyecto</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="conclusion-card">
                <div class="conclusion-number">1</div>
                <h4>El riesgo está en el comportamiento</h4>
                <p>
                    La principal señal de deserción no está en el perfil demográfico sino en el uso del producto.
                    Los clientes que abandonan el banco transan menos, mueven menos dinero y muestran más meses de inactividad.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="conclusion-card red">
                <div class="conclusion-number">2</div>
                <h4>La deserción sí puede anticiparse</h4>
                <p>
                    En una cartera de {summary['total_customers']:,} registros, la pérdida histórica alcanza {summary['churn_rate']:.1%}.
                    Eso hace necesario un modelo que ayude a detectar a tiempo los casos más delicados para intervenir antes de la salida.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="conclusion-card">
                <div class="conclusion-number">3</div>
                <h4>LightGBM fue la mejor alternativa</h4>
                <p>
                    Frente a la línea base de regresión logística, LightGBM mostró un mejor equilibrio entre precisión y recall.
                    Por eso se tomó como modelo principal dentro del proyecto y como base para estimar riesgo de nuevos clientes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        """
        <div class="big-panel">
            <h3>Cierre general</h3>
            <p>
                En conjunto, el proyecto muestra que la deserción de clientes no ocurre de manera repentina, sino como un proceso gradual
                de pérdida de actividad. A partir del análisis exploratorio, la validación estadística, el preprocesamiento y el modelado,
                fue posible construir una solución coherente con el problema planteado. Más allá de la métrica final, el mayor valor del trabajo
                está en que conecta los datos con decisiones reales de retención y ofrece una base clara para futuras mejoras.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_classification_report(y_true, y_pred, output_dict=False):
    y_true = sanitize_target(y_true)
    y_pred = sanitize_target(y_pred)
    return classification_report(y_true, y_pred, output_dict=output_dict)


def compute_model_artifacts(X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, lr_model, scaler, lgb_model) -> dict:
    y_test = sanitize_target(y_test)
    lr_feature_names = list(scaler.feature_names_in_)
    X_test_lr = align_raw_features(X_test, lr_feature_names)
    X_test_scaled = scaler.transform(X_test_lr)
    y_pred_lr = sanitize_target(lr_model.predict(X_test_scaled))
    y_prob_lr = np.asarray(lr_model.predict_proba(X_test_scaled)[:, 1], dtype=float)

    lgb_feature_names = get_lightgbm_feature_names(lgb_model, list(X_test.columns))
    X_test_lgb = align_raw_features(X_test, lgb_feature_names)
    y_pred_lgb = sanitize_target(lgb_model.predict(X_test_lgb))
    y_prob_lgb = np.asarray(lgb_model.predict_proba(X_test_lgb)[:, 1], dtype=float)

    report_lr = safe_classification_report(y_test, y_pred_lr, output_dict=True)
    report_lgb = safe_classification_report(y_test, y_pred_lgb, output_dict=True)

    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_prob_lgb)
    model_table = pd.DataFrame(
        [
            {
                "Modelo": "Regresión logística",
                "Accuracy": report_lr["accuracy"],
                "Precisión deserción": report_lr["1"]["precision"],
                "Recall deserción": report_lr["1"]["recall"],
                "F1 deserción": report_lr["1"]["f1-score"],
                "AUC": auc(fpr_lr, tpr_lr),
            },
            {
                "Modelo": "LightGBM",
                "Accuracy": report_lgb["accuracy"],
                "Precisión deserción": report_lgb["1"]["precision"],
                "Recall deserción": report_lgb["1"]["recall"],
                "F1 deserción": report_lgb["1"]["f1-score"],
                "AUC": auc(fpr_lgb, tpr_lgb),
            },
        ]
    )

    risk_band_df = pd.DataFrame({"y_true": y_test, "prob": y_prob_lgb, "pred": y_pred_lgb})
    risk_band_df["Banda"] = pd.cut(
        risk_band_df["prob"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
        labels=["Muy bajo", "Bajo", "Medio", "Alto", "Crítico"],
        include_lowest=True,
    )
    risk_band_summary = (
        risk_band_df.groupby("Banda", observed=False)
        .agg(Clientes=("y_true", "size"), Churn_real=("y_true", "mean"), Casos_predichos=("pred", "mean"))
        .reset_index()
    )
    risk_band_summary["Churn_real"] = (risk_band_summary["Churn_real"] * 100).round(1)
    risk_band_summary["Casos_predichos"] = (risk_band_summary["Casos_predichos"] * 100).round(1)

    return {
        "lgb_feature_names": lgb_feature_names,
        "y_pred_lgb": y_pred_lgb,
        "y_prob_lgb": y_prob_lgb,
        "report_lr": report_lr,
        "report_lgb": report_lgb,
        "model_table": model_table,
        "risk_band_summary": risk_band_summary,
    }


def build_notebook_metrics_tables(artifacts: dict, y_test: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    notebook_df = pd.DataFrame(
        [
            {
                "Métrica": "Exactitud",
                "Valor": artifacts["report_lgb"]["accuracy"],
                "Qué significa": "Porcentaje de aciertos totales",
            },
            {
                "Métrica": "Recall",
                "Valor": artifacts["report_lgb"]["1"]["recall"],
                "Qué significa": "Clientes en riesgo que el modelo logra detectar",
            },
            {
                "Métrica": "Precisión",
                "Valor": artifacts["report_lgb"]["1"]["precision"],
                "Qué significa": "Qué tan confiables son las alertas de riesgo",
            },
            {
                "Métrica": "F1",
                "Valor": artifacts["report_lgb"]["1"]["f1-score"],
                "Qué significa": "Balance entre precisión y recall",
            },
            {
                "Métrica": "AUC",
                "Valor": auc(*roc_curve(sanitize_target(y_test), artifacts["y_prob_lgb"])[:2]),
                "Qué significa": "Capacidad del modelo para separar clientes estables y en riesgo",
            },
        ]
    )
    test_df = pd.DataFrame(
        [
            {
                "Etapa": "Prueba final sobre X_test del proyecto",
                "Accuracy": artifacts["report_lgb"]["accuracy"],
                "AUC": auc(*roc_curve(sanitize_target(y_test), artifacts["y_prob_lgb"])[:2]),
                "Recall": artifacts["report_lgb"]["1"]["recall"],
                "Precisión": artifacts["report_lgb"]["1"]["precision"],
                "F1": artifacts["report_lgb"]["1"]["f1-score"],
            }
        ]
    )
    return notebook_df, test_df


def plot_model_comparison(model_table: pd.DataFrame) -> plt.Figure:
    plot_df = model_table.melt(id_vars="Modelo", var_name="Metrica", value_name="Valor")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    sns.barplot(data=plot_df, x="Metrica", y="Valor", hue="Modelo", palette=[COLORS["amber"], COLORS["teal"]], ax=ax)
    ax.set_title("Comparación de modelos en prueba", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(title="")
    return fig


def plot_confusion(y_true, y_pred, title: str) -> plt.Figure:
    y_true = sanitize_target(y_true)
    y_pred = sanitize_target(y_pred)
    fig, ax = plt.subplots(figsize=(5.6, 4.9))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Permanece", "Deserción"])
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title, fontsize=14, fontweight="bold")
    return fig


def plot_roc(y_true, y_score, title: str) -> plt.Figure:
    y_true = sanitize_target(y_true)
    y_score = np.asarray(y_score, dtype=float).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6.2, 4.9))
    ax.plot(fpr, tpr, color=COLORS["teal"], lw=3, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=COLORS["muted"], lw=1.2, linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    return fig


def plot_feature_importance(model, feature_names: list[str], top_n: int = 12) -> plt.Figure:
    if hasattr(model, "steps") and model.steps:
        model = model.steps[-1][1]

    if hasattr(model, "feature_importance"):
        importance = model.feature_importance()
    else:
        importance = model.feature_importances_

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importancia": importance})
        .sort_values("Importancia", ascending=False)
        .head(top_n)
    )
    importance_df["Feature"] = importance_df["Feature"].str.replace("_-_", " - ", regex=False).str.replace("_", " ", regex=False)

    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    sns.barplot(data=importance_df, x="Importancia", y="Feature", color=COLORS["teal"], ax=ax)
    ax.set_title("Variables con mayor peso en LightGBM", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importancia relativa")
    ax.set_ylabel("")
    return fig


def render_hero(summary: dict) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>Análisis de Deserción de Clientes</h1>
            <p>
                Aquí se resume el comportamiento de la cartera, los patrones que aparecen antes de la deserción y el desempeño
                del modelo construido con los datos del proyecto. En la muestra estudiada, <strong>{summary['churn_count']:,}</strong>
                clientes abandonaron la entidad dentro de una cartera de <strong>{summary['total_customers']:,}</strong> registros.
            </p>
            <p><strong>Autores:</strong> Juan Montes Sabogal y Nicolás Almonacid Muñoz</p>
            <div class="hero-badges">
                <span class="hero-badge">Comportamiento de clientes</span>
                <span class="hero-badge">Comparación de modelos</span>
                <span class="hero-badge">Casos de prueba</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quick_actions() -> None:
    st.markdown('<div class="section-title">Atajos de navegación</div>', unsafe_allow_html=True)
    action_columns = st.columns(5)
    actions = [
        ("Sala gerencial", "Sala gerencial"),
        ("Ver datos", "Qué muestran los clientes"),
        ("Abrir modelo", "Qué tan bien funciona el modelo"),
        ("Probar cliente", "Simular un cliente"),
        ("Leer cierre", "Conclusiones finales"),
    ]
    for column, (label, target_section) in zip(action_columns, actions):
        with column:
            st.button(label, use_container_width=True, on_click=go_to_section, args=(target_section,))


def render_overview(df: pd.DataFrame, summary: dict) -> None:
    st.markdown('<div class="section-title">Resumen ejecutivo</div>', unsafe_allow_html=True)
    render_kpi_strip(summary)

    insight1, insight2, insight3 = st.columns(3)
    with insight1:
        st.markdown("""
            <div class="insight-card">
                <h4>Problema de negocio</h4>
                <p>La fuga de clientes es importante: uno de cada seis termina abandonando el banco. Esto afecta ingresos, oportunidades de venta cruzada y costos de adquisición futura.</p>
            </div>
            """, unsafe_allow_html=True)
    with insight2:
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>Señal más fuerte</h4>
                <p>Los clientes que se retiran realizan en promedio {summary['churn_trans_ct']:.1f} transacciones, frente a {summary['active_trans_ct']:.1f} en los clientes que permanecen. La actividad real pesa más que la demografía.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with insight3:
        st.markdown("""
            <div class="insight-card">
                <h4>Qué se puede hacer</h4>
                <p>La combinación de baja actividad, más meses inactivos y mayor fricción de contacto dibuja cohortes donde el banco puede actuar antes de perder valor.</p>
            </div>
            """, unsafe_allow_html=True)

    left, right = st.columns([0.95, 1.05])
    with left:
        st.caption("Distribución general de la cartera: porcentaje de clientes que permanecen y porcentaje de clientes que se van.")
        st.pyplot(plot_class_balance(df), use_container_width=True)
    with right:
        st.caption("Comparación simple de comportamiento entre clientes que permanecen y clientes que se van.")
        st.pyplot(plot_behavior_gap(df), use_container_width=True)

    st.markdown('<div class="section-title">Señales que importan</div>', unsafe_allow_html=True)
    key_cols = st.columns(3)
    key_cards = [
        ("Transacciones", "Los clientes que se van hacen menos transacciones y mueven menos dinero. Esa es la señal más estable del churn."),
        ("Inactividad", "La inactividad de 12 meses separa muy bien a quienes siguen activos de quienes se están desconectando."),
        ("Relación", "Cuando cae la relación con el banco, el riesgo sube. El modelo lo aprende con bastante claridad."),
    ]
    for column, (title, text) in zip(key_cols, key_cards):
        with column:
            st.markdown(
                f"""
                <div class="insight-card">
                    <h4>{title}</h4>
                    <p>{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div class="small-note">
            La deserción no aparece por edad o género de forma aislada.
            La diferencia más clara está en el uso real del producto. El cliente que se va mueve menos dinero
            ({format_currency(summary['churn_trans_amt'])} vs {format_currency(summary['active_trans_amt'])})
            y permanece más tiempo inactivo ({summary['churn_inactive_months']:.2f} vs {summary['active_inactive_months']:.2f} meses).
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_customer_story(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Radiografía de clientes</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Comportamiento", "Segmentos", "Conclusiones prácticas"])

    with tab1:
        left, right = st.columns(2)
        with left:
            st.caption("Cada punto es un cliente. Ayuda a ver cómo cambia el riesgo según nivel de uso y monto transado.")
            st.pyplot(plot_trans_scatter(df), use_container_width=True)
        with right:
            st.caption("Resumen promedio por grupo: sirve para comparar rápidamente a los clientes que permanecen vs los que se van.")
            compare_df = (
                df.groupby(TARGET_COLUMN)[["Total_Trans_Amt", "Total_Trans_Ct", "Months_Inactive_12_mon", "Avg_Utilization_Ratio", "Total_Relationship_Count"]]
                .mean()
                .round(2)
                .transpose()
                .rename(columns={ACTIVE_LABEL: "Permanecen", CHURN_LABEL: "Se retiran"})
            )
            st.dataframe(compare_df, use_container_width=True, height=310)
            st.caption("El patrón de deserción se parece a una desconexión progresiva: menos uso, menos transacciones y mayor inactividad.")

    with tab2:
        left, right = st.columns(2)
        with left:
            st.caption("Mapa de riesgo: entre más alto el porcentaje, mayor prioridad de intervención.")
            st.pyplot(plot_inactivity_heatmap(df), use_container_width=True)
        with right:
            segment_choice = st.selectbox("Ver deserción por segmento", ["Income_Category", "Card_Category", "Marital_Status", "Education_Level", "Gender"])
            titles = {
                "Income_Category": "Riesgo por nivel de ingreso",
                "Card_Category": "Riesgo por categoría de tarjeta",
                "Marital_Status": "Riesgo por estado civil",
                "Education_Level": "Riesgo por nivel educativo",
                "Gender": "Riesgo por género",
            }
            st.caption("Riesgo por segmento: útil para definir campañas por tipo de cliente.")
            st.pyplot(plot_segment_risk(df, segment_choice, titles[segment_choice]), use_container_width=True)
            st.caption("Las categorías pequeñas pueden verse más extremas, por eso conviene interpretar estos resultados junto con el tamaño de cada grupo.")

        st.markdown('<div class="section-title">Arquetipos útiles</div>', unsafe_allow_html=True)
        archetype_cols = st.columns(3)
        archetypes = [
            ("Cliente en riesgo", "Poco uso, más inactividad, pocas relaciones activas y un patrón parecido al churn histórico."),
            ("Cliente neutro", "Actividad intermedia. No es prioridad alta, pero vale la pena monitorear cambios en uso y contacto."),
            ("Cliente sano", "Alta actividad, varias relaciones y menor inactividad. Es el perfil más estable para retener."),
        ]
        for column, (title, text) in zip(archetype_cols, archetypes):
            with column:
                st.markdown(
                    f"""
                    <div class="story-card">
                        <h4>{title}</h4>
                        <p>{text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab3:
        priority_segments = build_priority_segments(df)
        left, right = st.columns([1.15, 0.85])
        with left:
            st.caption("Segmentos prioritarios: grupos con tamaño suficiente y mayor riesgo, recomendados para iniciar acciones.")
            st.dataframe(priority_segments, use_container_width=True, height=360)
        with right:
            st.markdown(
                f"""
                <div class="risk-box" style="background: linear-gradient(135deg, {COLORS['rose']}, {COLORS['coral']});">
                    <div class="metric-label" style="color: rgba(255,255,255,0.75);">Hallazgo central</div>
                    <div style="font-size: 1.45rem; font-weight: 800; line-height: 1.2;">Las alertas tempranas están en el comportamiento, no solo en el perfil.</div>
                    <div style="margin-top: 0.65rem; color: rgba(255,255,255,0.88); line-height: 1.5;">
                        Los grupos con más inactividad y menos transacciones concentran la mayor prioridad comercial.
                        Allí tiene más sentido actuar con campañas de retención, beneficios de uso y seguimiento preventivo.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown(
                """
                **Lectura de la tabla**

                - Muestra dónde se combina volumen suficiente con una tasa de deserción elevada.
                - Conviene no exagerar un resultado porcentual cuando la cohorte es demasiado pequeña.
                - Los segmentos ayudan a priorizar acciones, pero no reemplazan al modelo.
                """
            )

    st.markdown('<div class="section-title">Interpretación de resultados</div>', unsafe_allow_html=True)
    action_df = build_action_recommendations(df)
    left, right = st.columns([1.05, 0.95])
    with left:
        st.caption("Acciones propuestas por frente: muestra qué hacer, con qué tipo de cliente y cuántos casos aproximados abarca.")
        st.dataframe(action_df, use_container_width=True, height=230)
    with right:
        st.markdown(
            """
            <div class="story-card">
                <h4>Valor del análisis</h4>
                <p>El análisis no se limita a describir clientes: traduce los hallazgos en decisiones concretas de retención. Esto fortalece la interpretación del problema y su aplicación práctica.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            """
            <div class="story-card">
                <h4>Conclusión principal</h4>
                <p>La deserción se comporta como una desconexión progresiva. Primero cae el uso, luego aumenta la inactividad y finalmente aparece la salida del cliente. El modelo captura justamente esa secuencia.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_model_section(X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, lr_model, scaler, lgb_model) -> None:
    artifacts = compute_model_artifacts(X_train, X_test, y_test, lr_model, scaler, lgb_model)
    report_lr = artifacts["report_lr"]
    report_lgb = artifacts["report_lgb"]
    model_table = artifacts["model_table"].copy()
    model_table_display = model_table.copy()
    for column in ["Accuracy", "Precisión deserción", "Recall deserción", "F1 deserción", "AUC"]:
        model_table_display[column] = model_table_display[column].map(lambda value: f"{value:.3f}")
    notebook_df, test_df = build_notebook_metrics_tables(artifacts, y_test)
    notebook_df_display = notebook_df.copy()
    test_df_display = test_df.copy()
    for table in [notebook_df_display, test_df_display]:
        for column in table.columns:
            if column != "Etapa" and column != "Tiempo" and column != "Métrica" and column != "Qué significa":
                table[column] = table[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "—")
        if "Tiempo" in table.columns:
            table["Tiempo"] = table["Tiempo"].map(lambda value: f"{value:.4f} s" if pd.notna(value) else "—")

    st.markdown('<div class="section-title">Qué tan bien funciona el modelo</div>', unsafe_allow_html=True)

    compare_metric_cols = st.columns(3)
    comparison_summary = [
        ("Mejor recall", "LightGBM", "Detecta más clientes que se van"),
        ("Modelo base", "Logistic Regression", "Sirve para comparar, pero se queda corto"),
        ("Uso recomendado", "Retención", "Priorizar contacto en clientes con riesgo alto"),
    ]
    for column, (title, value, note) in zip(compare_metric_cols, comparison_summary):
        with column:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="kpi-title">{title}</div>
                    <div class="kpi-value" style="font-size: 1.5rem;">{value}</div>
                    <div class="kpi-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        render_metric_card("Exactitud", f"{report_lgb['accuracy']:.4f}", "Porcentaje de aciertos totales del modelo.")
    with top2:
        render_metric_card("Recall", f"{report_lgb['1']['recall']:.4f}", "Qué tanto detecta clientes que realmente se van.")
    with top3:
        render_metric_card("Precisión", f"{report_lgb['1']['precision']:.4f}", "Qué tan confiables son las alertas de riesgo.")
    with top4:
        render_metric_card("AUC", f"{model_table.loc[model_table['Modelo']=='LightGBM', 'AUC'].values[0]:.4f}", "Capacidad de separar clientes estables vs en riesgo.")

    st.markdown(
        """
        <div class="big-panel">
            <h3>Cómo leer estas métricas</h3>
            <p>
                En el proyecto se revisaron las métricas durante el proceso de modelado y ajuste de LightGBM.
                En este tablero también se muestra la prueba final sobre <strong>X_test</strong>, que es el conjunto separado del proyecto.
                Por eso aquí se presentan dos lecturas complementarias: una para entender cómo se eligió el modelo y otra para ver cómo responde
                cuando se evalúa sobre datos de prueba.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Resumen del modelo", "Prueba en datos nuevos", "Variables más importantes"])

    with tab1:
        left, right = st.columns([1.0, 1.0])
        with left:
            st.caption("Comparación de modelos: permite ver por qué LightGBM se elige como modelo principal.")
            st.dataframe(model_table_display, use_container_width=True, height=210)
        with right:
            st.markdown(
                """
                <div class="big-panel">
                    <h3>Qué significa en términos simples</h3>
                    <p>
                        LightGBM es el modelo que mejor equilibra detección de riesgo y confiabilidad.
                        En otras palabras: ayuda a encontrar a más clientes que se pueden perder,
                        sin disparar alertas innecesarias en exceso.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.caption("Lectura rápida de métricas del modelo principal en lenguaje simple.")
            st.dataframe(notebook_df_display, use_container_width=True, height=170)

    with tab2:
        left, right = st.columns([0.95, 1.05])
        with left:
            st.caption("Métricas finales del modelo principal sobre datos separados para prueba.")
            st.dataframe(test_df_display, use_container_width=True, height=120)
            st.caption("Matriz de confusión: muestra aciertos y errores del modelo por tipo de cliente.")
            st.pyplot(plot_confusion(y_test, artifacts["y_pred_lgb"], "Matriz de confusión sobre el conjunto de prueba"), use_container_width=True)
        with right:
            st.caption("Curva ROC: mientras más cerca de la esquina superior izquierda, mejor rendimiento general.")
            st.pyplot(plot_roc(y_test, artifacts["y_prob_lgb"], "Curva ROC sobre el conjunto de prueba"), use_container_width=True)
            st.caption("Bandas de riesgo: distribución de clientes por probabilidad de salida.")
            st.dataframe(artifacts["risk_band_summary"], use_container_width=True, height=210)
            st.caption("La tabla de bandas de riesgo resume cómo se distribuyen las probabilidades del modelo en los datos de prueba.")

        st.markdown('<div class="section-title">Lectura ejecutiva</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="big-panel">
                <h3>Qué se debe recordar</h3>
                <p>
                    El modelo no está prediciendo a partir de edad o género como señal principal. Lo importante es el comportamiento:
                    cuántas transacciones hace el cliente, cuánto mueve, cuán inactivo está y cómo se relaciona con el banco.
                    Por eso LightGBM funciona mejor y por eso el proyecto y este tablero cuentan la misma historia.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab3:
        left, right = st.columns([1.0, 1.0])
        with left:
            st.caption("Variables que más influyen en la predicción del modelo.")
            st.pyplot(plot_feature_importance(lgb_model, artifacts["lgb_feature_names"], top_n=12), use_container_width=True)
        with right:
            st.markdown(
                """
                <div class="big-panel">
                    <h3>Qué está aprendiendo el modelo</h3>
                    <p>
                        Las variables más importantes vuelven a señalar la misma idea del análisis exploratorio:
                        lo más importante no es quién es el cliente, sino cómo usa la tarjeta.
                        El monto transado, la cantidad de transacciones, la inactividad y la intensidad de la relación con el banco
                        concentran buena parte de la capacidad de predicción.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown(
                """
                <div class="story-card">
                    <h4>Lectura para negocio</h4>
                    <p>Cuando cae la actividad del cliente, el riesgo aumenta. Por eso el modelo resulta útil no solo para clasificar, sino también para orientar acciones de retención en los grupos con menor uso y mayor inactividad.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="big-panel">
            <h3>Conclusión del modelado</h3>
            <p>
                Tomando como referencia el proceso completo del proyecto, LightGBM fue la alternativa más sólida para identificar clientes con riesgo de salida.
                La comparación inicial, la validación del notebook y la prueba final coinciden en una idea central:
                el modelo funciona bien y además tiene sentido con el comportamiento real de los clientes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def build_reference_profiles(df: pd.DataFrame) -> dict:
    churn_df = df[df[TARGET_COLUMN] == CHURN_LABEL]

    def safe_mode(frame: pd.DataFrame, column: str, fallback: str) -> str:
        if frame.empty or frame[column].dropna().empty:
            return fallback
        return str(frame[column].mode().iloc[0])

    def bounded_int(value: float, min_value: int, max_value: int) -> int:
        return int(max(min_value, min(max_value, round(value))))

    def bounded_float(value: float, min_value: float, max_value: float) -> float:
        return float(max(min_value, min(max_value, value)))

    high_risk = {
        "Gender": safe_mode(churn_df, "Gender", str(df["Gender"].mode().iloc[0])),
        "Marital_Status": safe_mode(churn_df, "Marital_Status", str(df["Marital_Status"].mode().iloc[0])),
        "Card_Category": safe_mode(churn_df, "Card_Category", str(df["Card_Category"].mode().iloc[0])),
        "Education_Level": safe_mode(churn_df, "Education_Level", str(df["Education_Level"].mode().iloc[0])),
        "Income_Category": safe_mode(churn_df, "Income_Category", str(df["Income_Category"].mode().iloc[0])),
        "Dependent_count": bounded_int(df["Dependent_count"].quantile(0.40), 0, 6),
        "Total_Relationship_Count": bounded_int(df["Total_Relationship_Count"].quantile(0.15), 1, 6),
        "Months_Inactive_12_mon": bounded_int(df["Months_Inactive_12_mon"].quantile(0.90), 0, 6),
        "Contacts_Count_12_mon": bounded_int(df["Contacts_Count_12_mon"].quantile(0.85), 0, 6),
        "Total_Trans_Amt": bounded_float(df["Total_Trans_Amt"].quantile(0.10), 0.0, 25000.0),
        "Avg_Utilization_Ratio": bounded_float(df["Avg_Utilization_Ratio"].quantile(0.20), 0.0, 1.0),
    }

    return {
        "Cliente en alto riesgo": high_risk,
        "Promedio de la cartera": {
            "Gender": df["Gender"].mode()[0],
            "Marital_Status": df["Marital_Status"].mode()[0],
            "Card_Category": df["Card_Category"].mode()[0],
            "Education_Level": df["Education_Level"].mode()[0],
            "Income_Category": df["Income_Category"].mode()[0],
            "Dependent_count": int(df["Dependent_count"].median()),
            "Total_Relationship_Count": int(df["Total_Relationship_Count"].median()),
            "Months_Inactive_12_mon": int(df["Months_Inactive_12_mon"].median()),
            "Contacts_Count_12_mon": int(df["Contacts_Count_12_mon"].median()),
            "Total_Trans_Amt": float(df["Total_Trans_Amt"].median()),
            "Avg_Utilization_Ratio": float(df["Avg_Utilization_Ratio"].median()),
        },
        "Cliente fidelizado": {
            "Gender": "M",
            "Marital_Status": "Married",
            "Card_Category": "Blue",
            "Education_Level": "Graduate",
            "Income_Category": "$80K - $120K",
            "Dependent_count": 2,
            "Total_Relationship_Count": 5,
            "Months_Inactive_12_mon": 1,
            "Contacts_Count_12_mon": 1,
            "Total_Trans_Amt": 5200.0,
            "Avg_Utilization_Ratio": 0.42,
        },
    }


def get_risk_band(probability: float) -> tuple[str, str, str]:
    if probability < 0.20:
        return "Muy bajo", COLORS["mint"], "Cliente estable. Conviene mantener una buena experiencia y un monitoreo liviano."
    if probability < 0.40:
        return "Bajo", COLORS["teal"], "Riesgo contenido. Vale la pena revisar el uso y activar recordatorios de valor."
    if probability < 0.60:
        return "Medio", COLORS["amber"], "Cliente a vigilar. Conviene una acción comercial preventiva."
    if probability < 0.80:
        return "Alto", COLORS["coral"], "Riesgo alto. Es recomendable un contacto directo y una oferta de retención."
    return "Crítico", COLORS["rose"], "Prioridad inmediata. Conviene activar un plan de retención y seguimiento."


def render_prediction_tool(raw_df: pd.DataFrame, X_train: pd.DataFrame, label_maps: dict, lr_model, scaler, lgb_model) -> None:
    st.markdown('<div class="section-title">Evaluación de cliente</div>', unsafe_allow_html=True)
    st.caption("Usa este espacio para ver cómo cambia el riesgo cuando cambian la actividad, la relación con el banco y el comportamiento transaccional del cliente.")

    st.markdown(
        """
        <div class="big-panel">
            <h3>Casos de referencia</h3>
            <p>
                Elige un perfil de referencia y ajusta las variables más influyentes. El resultado se calcula con el modelo principal
                guardado en <strong>models/lgbm_tuned_model.pkl</strong>.
            </p>
            <p>
                El perfil <strong>Cliente en alto riesgo</strong> se construye con percentiles reales de la cartera, combinando baja actividad
                y alta inactividad para que el punto de partida sea más útil.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    profiles = build_reference_profiles(raw_df)
    selected_profile = st.selectbox("Perfil inicial", list(profiles.keys()))
    profile = profiles[selected_profile]

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        gender_options = list(label_maps["Gender"].keys())
        marital_options = list(label_maps["Marital_Status"].keys())
        card_options = list(label_maps["Card_Category"].keys())
        education_options = ["Graduate", "High School", "Uneducated", "Unknown", "College", "Doctorate", "Post-Graduate"]
        income_options = ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"]
        values = {
            "Gender": col1.selectbox("Género", gender_options, index=gender_options.index(profile["Gender"])),
            "Marital_Status": col1.selectbox("Estado civil", marital_options, index=marital_options.index(profile["Marital_Status"])),
            "Card_Category": col1.selectbox("Tarjeta", card_options, index=card_options.index(profile["Card_Category"])),
            "Education_Level": col2.selectbox("Nivel educativo", education_options, index=education_options.index(profile["Education_Level"])),
            "Income_Category": col2.selectbox("Rango de ingresos", income_options, index=income_options.index(profile["Income_Category"])),
            "Dependent_count": col1.slider("Dependientes", 0, 6, profile["Dependent_count"]),
            "Total_Relationship_Count": col1.slider("Relaciones activas", 1, 6, profile["Total_Relationship_Count"]),
            "Months_Inactive_12_mon": col1.slider("Meses inactivo", 0, 6, profile["Months_Inactive_12_mon"]),
            "Contacts_Count_12_mon": col1.slider("Contactos últimos 12 meses", 0, 6, profile["Contacts_Count_12_mon"]),
            "Total_Trans_Amt": col2.number_input("Monto total transado", min_value=0.0, max_value=25000.0, value=float(profile["Total_Trans_Amt"]), step=100.0),
            "Avg_Utilization_Ratio": col2.slider("Utilización promedio", 0.0, 1.0, float(profile["Avg_Utilization_Ratio"]), 0.01),
        }
        submitted = st.form_submit_button("Calcular riesgo del cliente")

    if not submitted:
        return

    model_columns = [column for column in X_train.columns if column != TARGET_COLUMN]
    feature_means = X_train[model_columns].mean()
    input_full = preprocess_input(values, model_columns, label_maps, feature_means)
    lr_columns = list(scaler.feature_names_in_)
    input_lr = input_full[lr_columns]
    input_scaled = scaler.transform(input_lr)
    prob_lr = float(lr_model.predict_proba(input_scaled)[0, 1])
    prob_lgb = float(lgb_model.predict_proba(input_full)[0, 1])
    band, band_color, recommendation = get_risk_band(prob_lgb)

    churn_profile = raw_df[raw_df[TARGET_COLUMN] == CHURN_LABEL]
    active_profile = raw_df[raw_df[TARGET_COLUMN] == ACTIVE_LABEL]

    top1, top2 = st.columns([0.9, 1.1])
    with top1:
        st.markdown(
            f"""
            <div class="risk-box" style="background: linear-gradient(135deg, {band_color}, {COLORS['navy']});">
                <div class="metric-label" style="color: rgba(255,255,255,0.76);">LightGBM</div>
                <div style="font-size: 2.25rem; font-weight: 800;">{prob_lgb:.1%}</div>
                <div style="font-size: 1rem; font-weight: 700; margin-top: 0.25rem;">Banda: {band}</div>
                <div style="margin-top: 0.7rem; line-height: 1.45; color: rgba(255,255,255,0.90);">{recommendation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        compare_boxes = st.columns(2)
        with compare_boxes[0]:
            render_metric_card("Probabilidad LightGBM", f"{prob_lgb:.1%}", "Modelo principal del proyecto")
        with compare_boxes[1]:
            render_metric_card("Probabilidad logística", f"{prob_lr:.1%}", f"Diferencia: {(prob_lgb - prob_lr):+.1%}")
        st.caption("La probabilidad de LightGBM es la referencia principal para decisión. La logística se deja solo como comparación histórica.")

    with top2:
        st.caption("Comparación del cliente ingresado contra promedios históricos de clientes que permanecen y clientes que se van.")
        compare_df = pd.DataFrame(
            {
                "Indicador": ["Monto transado", "Meses inactivo", "Relaciones activas", "Contactos", "Uso de crédito"],
                "Cliente": [
                    values["Total_Trans_Amt"],
                    values["Months_Inactive_12_mon"],
                    values["Total_Relationship_Count"],
                    values["Contacts_Count_12_mon"],
                    values["Avg_Utilization_Ratio"],
                ],
                "Promedio deserción": [
                    churn_profile["Total_Trans_Amt"].mean(),
                    churn_profile["Months_Inactive_12_mon"].mean(),
                    churn_profile["Total_Relationship_Count"].mean(),
                    churn_profile["Contacts_Count_12_mon"].mean(),
                    churn_profile["Avg_Utilization_Ratio"].mean(),
                ],
                "Promedio permanencia": [
                    active_profile["Total_Trans_Amt"].mean(),
                    active_profile["Months_Inactive_12_mon"].mean(),
                    active_profile["Total_Relationship_Count"].mean(),
                    active_profile["Contacts_Count_12_mon"].mean(),
                    active_profile["Avg_Utilization_Ratio"].mean(),
                ],
            }
        )
        compare_df[["Cliente", "Promedio deserción", "Promedio permanencia"]] = compare_df[["Cliente", "Promedio deserción", "Promedio permanencia"]].round(2)
        st.dataframe(compare_df, use_container_width=True, height=265)
        st.caption("Si el cliente se parece más al perfil de deserción que al perfil de permanencia, la probabilidad aumenta.")

    st.markdown(
        """
        <div class="story-card">
            <h4>Cómo interpretar este resultado</h4>
            <p>
                Una probabilidad alta no significa que el cliente se irá con certeza, pero sí indica que su comportamiento se parece al de quienes ya abandonaron el banco.
                Esto puede servir para priorizar seguimiento, campañas de retención o revisión de beneficios.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(summary: dict) -> str:
    st.sidebar.markdown("## Navegación")
    st.sidebar.caption("Usa las secciones o los atajos del panel principal.")
    current_section = st.session_state.get("dashboard_section", SECTION_ORDER[0])
    current_index = SECTION_ORDER.index(current_section) if current_section in SECTION_ORDER else 0
    section = st.sidebar.radio(
        "Sección",
        SECTION_ORDER,
        index=current_index,
        key="dashboard_section_radio",
    )
    st.session_state["dashboard_section"] = section
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Ficha del proyecto")
    st.sidebar.markdown(
        f"""
        - Tasa histórica de deserción: **{summary['churn_rate']:.1%}**
        - Señal más fuerte: **baja actividad transaccional**
        - Permanencia promedio: **{summary['avg_tenure']:.1f} meses**
        - Modelo recomendado: **LightGBM**
        - Autores: **Juan Montes Sabogal** y **Nicolás Almonacid Muñoz**
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Guía")
    st.sidebar.markdown(
        """
        - `Sala gerencial`: impacto y decisión.
        - `Qué muestran los clientes`: señales de riesgo.
        - `Qué tan bien funciona el modelo`: métricas explicadas con claridad.
        - `Simular un cliente`: evaluación de un caso puntual.
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Proyecto académico de análisis y predicción de deserción de clientes en banca.")
    return section


def main() -> None:
    inject_custom_css()
    st.session_state.setdefault("dashboard_section", SECTION_ORDER[0])

    raw_df = load_raw_dataset()
    X_train, X_test, y_train, y_test = load_processed_data()
    lr_model, scaler, lgb_model = load_models()

    label_maps = {
        "Gender": label_encoder_mapping(raw_df, "Gender"),
        "Marital_Status": label_encoder_mapping(raw_df, "Marital_Status"),
        "Card_Category": label_encoder_mapping(raw_df, "Card_Category"),
    }
    summary = build_portfolio_summary(raw_df)

    render_hero(summary)
    render_quick_actions()
    section = render_sidebar(summary)

    if section == "Sala gerencial":
        render_executive_room(raw_df, summary, X_train, X_test, y_test, lr_model, scaler, lgb_model)
    elif section == "Panorama general":
        render_overview(raw_df, summary)
    elif section == "Cómo se hizo":
        render_methodology()
    elif section == "Qué muestran los clientes":
        render_customer_story(raw_df)
    elif section == "Qué tan bien funciona el modelo":
        render_model_section(X_train, X_test, y_test, lr_model, scaler, lgb_model)
    elif section == "Conclusiones finales":
        render_project_conclusions(summary)
    else:
        render_prediction_tool(raw_df, X_train, label_maps, lr_model, scaler, lgb_model)

    st.markdown("---")
    st.caption("Trabajo realizado por Juan Montes Sabogal y Nicolás Almonacid Muñoz. El tablero reúne los resultados descritos a lo largo de los notebooks del proyecto y los organiza en una lectura clara para entender el problema, el modelo y sus conclusiones.")


if __name__ == "__main__":
    main()
