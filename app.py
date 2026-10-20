import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = Path(__file__).parent / "models" / "churn_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    h1 {
        font-weight: 750;
        letter-spacing: -0.5px;
    }

    .subtitle {
        color: #64748b;
        font-size: 16px;
        margin-bottom: 24px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }

    div[data-testid="stForm"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 22px;
    }

    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.15);
        padding: 16px;
        border-radius: 12px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        min-height: 44px;
        border-radius: 9px;
        font-weight: 600;
    }

    label {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("📊 Churn Predictor")
    st.caption("Machine Learning Project")
    st.divider()

    st.subheader("About this project")
    st.write(
        "This application predicts whether a telecom customer "
        "is likely to leave the service using a trained "
        "Logistic Regression model."
    )

    st.divider()
    st.subheader("Model performance")
    st.metric("Accuracy", "80.55%")
    st.metric("ROC-AUC", "84.21%")

    st.caption("Evaluation results from the held-out test set.")
    st.divider()
    st.caption("Built with Python, Pandas, Scikit-learn and Streamlit.")

# ---------------- PAGE HEADER ----------------
st.title("Customer Churn Prediction")
st.markdown(
    '<p class="subtitle">Enter customer details below to estimate their likelihood of leaving the service.</p>',
    unsafe_allow_html=True
)

# ---------------- INPUT FORM ----------------
with st.form("customer_details_form"):
    st.subheader("1. Customer Information")

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1],
                                      format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has Partner", ["Yes", "No"])

    with col2:
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    with col3:
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.divider()
    st.subheader("2. Internet and Additional Services")

    col4, col5, col6 = st.columns(3, gap="large")

    with col4:
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )

    with col5:
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )

    with col6:
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless_placeholder = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        device_placeholder = st.selectbox(
            "Online Account Type", ["Standard", "Other"]
        )

    st.divider()
    st.subheader("3. Billing Information")

    col7, col8 = st.columns(2, gap="large")

    with col7:
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0,
            max_value=2000.0, value=70.0, step=1.0
        )

    with col8:
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0,
            max_value=100000.0, value=840.0, step=10.0
        )

    st.caption("Check that the values reflect the customer you want to assess.")
    submitted = st.form_submit_button("🔍 Predict Customer Churn")

# ---------------- PREDICTION ----------------
if submitted:
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": paperless_placeholder,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Fill in the remaining expected fields if required by the trained model.
    # These values are explicit defaults, not additional user inputs.
    input_data["customerID"] = "NEW_CUSTOMER"
    input_data["OnlineAccountType"] = device_placeholder

    try:
        probability = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns([1, 2], gap="large")

        with result_col1:
            st.metric("Estimated Churn Probability", f"{probability:.1%}")

        with result_col2:
            if prediction == 1:
                st.error("⚠️ Higher churn risk: the model predicts this customer may leave.")
            else:
                st.success("✅ Lower churn risk: the model predicts this customer may stay.")

        st.progress(float(probability))
        st.caption(
            "This is a machine-learning estimate, not a guarantee of future customer behavior."
        )

    except Exception as error:
        st.error("The prediction could not be completed. Please check the input fields and trained model.")
        st.exception(error)
