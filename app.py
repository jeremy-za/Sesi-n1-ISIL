import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configuración de la página
st.set_page_config(page_title="Predicción de Churn", page_icon="📉", layout="wide")

st.title("📉 Predicción de baja de clientes (Churn)")
st.write("Ejemplo educativo con Streamlit. Entrena un modelo simple y realiza predicciones.")

# Generar dataset sintético
def make_synthetic_data(n=500):
    rng = np.random.default_rng(42)
    tenure = rng.integers(1, 72, size=n)  # meses como cliente
    monthly_charges = rng.normal(60, 20, size=n).clip(5, 150)
    contract_type = rng.choice(["Mensual", "Anual", "Dos años"], size=n, p=[0.6, 0.25, 0.15])
    has_complaints = rng.choice(["Sí", "No"], size=n, p=[0.3, 0.7])

    churn_prob = (
        0.45 * (contract_type == "Mensual").astype(float)
        + 0.25 * (has_complaints == "Sí").astype(float)
        + 0.20 * (monthly_charges > 80).astype(float)
        - 0.25 * (tenure > 24).astype(float)
    )
    churn_prob = 1 / (1 + np.exp(-churn_prob))
    churn = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges.round(2),
        "contract_type": contract_type,
        "has_complaints": has_complaints,
        "churn": churn
    })
    return df

# Cargar datos
data = make_synthetic_data()

st.subheader("Vista previa de datos")
st.dataframe(data.head(10), use_container_width=True)

# Separar variables
X = data.drop(columns=["churn"])
y = data["churn"]

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Preprocesamiento
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Modelo
model = LogisticRegression(max_iter=500)
clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

# Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf.fit(X_train, y_train)

# Evaluación
y_pred = clf.predict(X_test)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
col4.metric("F1-score", f"{f1_score(y_test, y_pred):.3f}")

st.markdown("---")
st.subheader("Predicción para un cliente nuevo")

with st.form("predict_form"):
    tenure = st.number_input("Meses como cliente", min_value=1, max_value=72, value=12)
    monthly_charges = st.number_input("Cargo mensual", min_value=5.0, max_value=150.0, value=60.0)
    contract_type = st.selectbox("Tipo de contrato", ["Mensual", "Anual", "Dos años"])
    has_complaints = st.selectbox("¿Tiene reclamos?", ["Sí", "No"])
    submitted = st.form_submit_button("Predecir")

if submitted:
    new_data = pd.DataFrame([{
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "contract_type": contract_type,
        "has_complaints": has_complaints
    }])
    pred = clf.predict(new_data)[0]
    prob = clf.predict_proba(new_data)[0, 1]
    resultado = "⚠️ Riesgo de baja" if pred == 1 else "✅ Cliente retenido"
    st.success(f"{resultado} | Probabilidad de churn: {prob:.2%}")
