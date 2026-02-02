# Medical Insurance Cost Prediction - Streamlit App (Regression)
# Academic, Multi-page, Streamlit Cloud Ready

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Medical Insurance Cost Prediction", layout="wide")

# Load files
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()
df = pd.read_csv("insurance.csv")
metrics_df = pd.read_csv("model_metrics.csv")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "EDA", "Model Metrics", "Prediction"])

# ---------------- OVERVIEW PAGE ----------------
if page == "Overview":
    st.title("Medical Insurance Cost Prediction Using Machine Learning")

    st.subheader("Project Description")
    st.write("This project aims to predict medical insurance charges based on demographic and lifestyle features using supervised regression models. The system helps estimate expected insurance cost for individuals.")

    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

# ---------------- EDA PAGE ----------------
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    st.subheader("Target Variable Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['charges'], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Feature vs Target Analysis")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=df['age'], y=df['charges'], ax=ax2)
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    st.subheader("Key Insights")
    st.write("- Age and smoking status have strong impact on insurance charges.\n"
             "- Higher BMI tends to increase medical cost.\n"
             "- Smokers show significantly higher charges compared to non-smokers.")

# ---------------- MODEL METRICS PAGE ----------------
elif page == "Model Metrics":
    st.title("Model Evaluation")

    st.subheader("Model Comparison")
    st.dataframe(metrics_df)

    st.subheader("Best Model Performance")
    y_true = df['charges']
    X = df.drop('charges', axis=1)
    X_transformed = preprocessor.transform(X)
    y_pred = model.predict(X_transformed)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", round(r2, 3))
    col2.metric("MAE", round(mae, 2))
    col3.metric("RMSE", round(rmse, 2))

    st.subheader("Actual vs Predicted Plot")
    fig4, ax4 = plt.subplots()
    ax4.scatter(y_true, y_pred)
    ax4.set_xlabel("Actual Charges")
    ax4.set_ylabel("Predicted Charges")
    st.pyplot(fig4)

    st.subheader("Hyperparameter Tuning")
    st.write("GridSearchCV was used to tune model parameters to achieve optimal generalization performance.")

    st.subheader("Final Model Selection")
    st.write("The final regression model was selected based on highest R² score and lowest RMSE on test data.")

# ---------------- PREDICTION PAGE ----------------
elif page == "Prediction":
    st.title("Insurance Cost Prediction")

    age = st.number_input("Age", 18, 100)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", 10.0, 60.0)
    children = st.number_input("Number of Children", 0, 5)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    if st.button("Predict Insurance Cost"):
        input_df = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region]
        })

        input_transformed = preprocessor.transform(input_df)
        prediction = model.predict(input_transformed)[0]

        st.success(f"Estimated Medical Insurance Cost: ₹ {prediction:,.2f}")
        st.info("This is the expected annual insurance charge based on the given personal details.")
