
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# -------------------------------
# 1. Page configuration
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a customer is likely to leave "
    "using Logistic Regression."
)

# -------------------------------
# 2. Generate sample dataset
# -------------------------------
@st.cache_data
def create_dataset():
    rng = np.random.default_rng(42)
    n = 1000

    tenure = rng.integers(1, 73, n)
    monthly_charges = rng.uniform(500, 5000, n)
    support_calls = rng.integers(0, 11, n)

    # Simulate churn probability
    score = (
        -1.0
        - 0.045 * tenure
        + 0.0005 * monthly_charges
        + 0.30 * support_calls
    )

    probability = 1 / (1 + np.exp(-score))
    churn = rng.binomial(1, probability)

    data = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges.round(2),
        "support_calls": support_calls,
        "churn": churn
    })

    data.to_csv("customer_churn.csv", index=False)
    return data


df = create_dataset()

# -------------------------------
# 3. Display dataset
# -------------------------------
st.subheader("Customer Dataset")
st.dataframe(df.head(10), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(df))
col2.metric("Customers Churned", int(df["churn"].sum()))
col3.metric(
    "Churn Rate",
    f"{df['churn'].mean() * 100:.1f}%"
)

# -------------------------------
# 4. Prepare features and target
# -------------------------------
X = df[["tenure", "monthly_charges", "support_calls"]]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scaling and model training
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

# -------------------------------
# 5. Evaluate the model
# -------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

st.subheader("Model Performance")

col1, col2 = st.columns(2)
col1.metric("Test Accuracy", f"{accuracy:.2%}")
col2.metric("ROC-AUC Score", f"{auc:.3f}")

# Confusion matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

fig, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stay", "Churn"],
    yticklabels=["Stay", "Churn"],
    ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
plt.close(fig)

# Classification report
st.subheader("Classification Report")
report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Stay", "Churn"],
    output_dict=True,
    zero_division=0
)
st.dataframe(pd.DataFrame(report).transpose())

# -------------------------------
# 6. Interactive prediction
# -------------------------------
st.subheader("Predict for a New Customer")

with st.form("prediction_form"):
    tenure_input = st.slider(
        "Customer tenure (months)", 1, 72, 24
    )
    charges_input = st.number_input(
        "Monthly charges", 500.0, 5000.0, 2500.0
    )
    calls_input = st.slider(
        "Customer support calls", 0, 10, 3
    )
    submitted = st.form_submit_button("Predict Churn")

if submitted:
    customer = pd.DataFrame([{
        "tenure": tenure_input,
        "monthly_charges": charges_input,
        "support_calls": calls_input
    }])

    churn_probability = model.predict_proba(customer)[0, 1]
    prediction = int(model.predict(customer)[0])

    st.metric(
        "Churn Probability",
        f"{churn_probability:.1%}"
    )

    if prediction == 1:
        st.warning("Prediction: Customer is likely to churn.")
    else:
        st.success("Prediction: Customer is likely to stay.")

    st.caption(
        "This is a demonstration model trained on synthetic data, "
        "not actual customer records."
    )

# -------------------------------
# 7. Feature importance
# -------------------------------
st.subheader("Model Coefficients")

classifier = model.named_steps["logisticregression"]
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": classifier.coef_[0]
})

st.dataframe(coefficients, use_container_width=True)

fig, ax = plt.subplots()
ax.bar(
    coefficients["Feature"],
    coefficients["Coefficient"]
)
ax.axhline(0, linewidth=1)
ax.set_ylabel("Standardized coefficient")
ax.set_title("Feature Effect on Churn Log-Odds")
st.pyplot(fig)
plt.close(fig)