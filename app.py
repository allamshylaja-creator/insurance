# Medical Insurance Charges Estimator
# Streamlit | Regression | Academic Project (Updated for Python 3.11)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# -------------------
# Page Config
# -------------------
st.set_page_config(
    page_title="Insurance Charges Estimator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# Load Model & Preprocessor
# -------------------
@st.cache_resource
def load_artifacts():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "model.pkl")
    preprocessor_path = os.path.join(base_path, "preprocessor.pkl")
    data_path = os.path.join(base_path, "insurance.csv")
    metrics_path = os.path.join(base_path, "model_metrics.csv")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    data = pd.read_csv(data_path)
    metrics_df = pd.read_csv(metrics_path)
    return model, preprocessor, data, metrics_df

model, preprocessor, data, metrics_df = load_artifacts()

# -------------------
# Sidebar Menu
# -------------------
st.sidebar.title("📌 App Menu")
page = st.sidebar.radio(
    "Select Section",
    ["Project Overview", "Data Exploration", "Model Performance", "Cost Estimator"]
)

# -------------------
# Project Overview
# -------------------
if page == "Project Overview":
    st.title("💊 Medical Insurance Cost Estimation System")
    st.markdown(
        """
        This application predicts **annual medical insurance charges** using
        machine learning regression techniques based on demographic and lifestyle data.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Records", data.shape[0])
    col2.metric("Features", data.shape[1])
    col3.metric("Missing Entries", data.isnull().sum().sum())

    st.subheader("Sample Records")
    st.dataframe(data.sample(15, random_state=1))

    st.subheader("Statistical Summary")
    st.dataframe(data.describe())

# -------------------
# Data Exploration
# -------------------
elif page == "Data Exploration":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Distribution of Insurance Charges")
    fig1, ax1 = plt.subplots()
    sns.histplot(data["charges"], bins=40, kde=True, color="teal", ax=ax1)
    st.pyplot(fig1)

    st.subheader("Impact of Smoking on Charges")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="smoker", y="charges", data=data, palette="Set2", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Age vs Charges Trend")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x="age", y="charges", hue="smoker", data=data, alpha=0.6, ax=ax3)
    st.pyplot(fig3)

    st.subheader("Correlation Between Numeric Features")
    numeric_data = data.select_dtypes(include="number")
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax4)
    st.pyplot(fig4)

    st.info(
        """
        **Observations:**
        - Smoking status is the strongest cost driver.
        - Age and BMI show positive correlation with insurance charges.
        - Children count has minimal influence on cost.
        """
    )

# -------------------
# Model Performance
# -------------------
elif page == "Model Performance":
    st.title("📈 Model Evaluation & Results")

    st.subheader("Regression Model Comparison")
    st.dataframe(metrics_df)

    X = data.drop("charges", axis=1)
    y_actual = data["charges"]

    X_processed = preprocessor.transform(X)
    y_predicted = model.predict(X_processed)

    r2 = r2_score(y_actual, y_predicted)
    mae = mean_absolute_error(y_actual, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("Mean Absolute Error", f"{mae:.2f}")
    col3.metric("Root Mean Sq. Error", f"{rmse:.2f}")

    st.subheader("Actual vs Predicted Charges")
    fig5, ax5 = plt.subplots()
    ax5.scatter(y_actual, y_predicted, alpha=0.5)
    ax5.plot([y_actual.min(), y_actual.max()],
             [y_actual.min(), y_actual.max()],
             color="red", linestyle="--")
    ax5.set_xlabel("Actual Charges")
    ax5.set_ylabel("Predicted Charges")
    st.pyplot(fig5)

    st.success(
        "The final regression model was selected based on strong generalization "
        "performance and minimal prediction error."
    )

# -------------------
# Cost Estimator
# -------------------
elif page == "Cost Estimator":
    st.title("🧮 Predict Your Insurance Charges")

    with st.form("prediction_form"):
        age = st.slider("Age", 18, 100, 30)
        sex = st.selectbox("Gender", ["male", "female"])
        bmi = st.slider("Body Mass Index (BMI)", 10.0, 60.0, 25.0)
        children = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4, 5])
        smoker = st.radio("Smoking Habit", ["yes", "no"])
        region = st.selectbox("Residential Region", ["northeast", "northwest", "southeast", "southwest"])

        submitted = st.form_submit_button("Estimate Cost")

    if submitted:
        user_input = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region]
        })

        processed_input = preprocessor.transform(user_input)
        estimated_cost = model.predict(processed_input)[0]

        st.success(f"💰 Estimated Annual Insurance Cost: ₹ {estimated_cost:,.2f}")
        st.caption(
            "Prediction is based on historical data patterns and trained ML regression model."
        )
