
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Header
st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a telecom customer may leave "
    "the service using machine learning."
)

st.divider()

# Sidebar
st.sidebar.title("About This Project")
st.sidebar.write(
    "A machine learning project using the IBM Telco "
    "Customer Churn dataset."
)
st.sidebar.info(
    "Predictions are estimates, not guarantees."
)

st.sidebar.subheader("Model Performance")
st.sidebar.metric("Test Accuracy", "80.55%")
st.sidebar.metric("Test ROC-AUC", "84.21%")
st.sidebar.caption("Model: Logistic Regression")

# Customer form
st.subheader("Enter Customer Details")

with st.form("customer_form"):

    st.markdown("### 👤 Personal Information")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    with c3:
        partner = st.selectbox("Has Partner", ["Yes", "No"])
    with c4:
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    st.markdown("### 📱 Services")
    c1, c2, c3 = st.columns(3)

    with c1:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with c2:
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with c3:
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    st.markdown("### 📄 Contract and Billing")
    c1, c2, c3 = st.columns(3)

    with c1:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
    with c2:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with c3:
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.markdown("### 💰 Charges and Tenure")
    c1, c2, c3 = st.columns(3)

    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with c2:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )
    with c3:
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=100000.0,
            value=840.0
        )

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn",
        type="primary",
        use_container_width=True
    )

# Show results only after submission
if submitted:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    prediction = int(model.predict(customer)[0])
    probability = float(model.predict_proba(customer)[0][1])

    st.divider()
    st.subheader("Prediction Results")

    result_col, probability_col = st.columns(2)

    with result_col:
        if prediction == 1:
            st.error("⚠️ Customer is predicted to churn")
        else:
            st.success("✅ Customer is predicted to stay")

    with probability_col:
        st.metric("Estimated Churn Probability", f"{probability:.1%}")

    st.progress(probability)

    st.caption(
        "This is a model estimate, not a guarantee of future behavior."
    )
