import os
import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Customer Churn Prediction | AI Banking",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# 2. LOAD TRAINED MODEL AND PREPROCESSORS
# =========================================================
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "model.h5")
    )
    label_encoder = joblib.load(
        os.path.join(BASE_DIR, "label_encoder.pkl")
    )
    one_hot_encoder = joblib.load(
        os.path.join(BASE_DIR, "one_hot_encoder.pkl")
    )
    scaler = joblib.load(
        os.path.join(BASE_DIR, "scaler.pkl")
    )
    return model, label_encoder, one_hot_encoder, scaler


try:
    model, label_encoder, one_hot_encoder, scaler = load_artifacts()
except Exception as error:
    st.error(
        "Unable to load the model or preprocessing files. "
        "Check that model.h5, label_encoder.pkl, "
        "one_hot_encoder.pkl and scaler.pkl are in the same folder as app.py."
    )
    st.exception(error)
    st.stop()

# =========================================================
# 3. CUSTOM CSS - BACKGROUND AND PROFESSIONAL UI
# =========================================================
st.markdown("""
<style>
/* Professional full-page banking / corporate photo background */
.stApp {
    background: transparent !important;
    color: #F8FAFC;
}

/* Apply image to Streamlit's actual app viewport */
div[data-testid="stAppViewContainer"] {
    background-image:
        linear-gradient(
            135deg,
            rgba(3, 15, 35, 0.88),
            rgba(8, 47, 73, 0.82),
            rgba(7, 89, 133, 0.78)
        ),
        url("https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=2000&q=90") !important;
    background-size: cover !important;
    background-position: center center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
}

/* Keep Streamlit's content area transparent so the photo remains visible */
section[data-testid="stMain"],
div[data-testid="stMainBlockContainer"] {
    background: transparent !important;
}

[data-testid="stHeader"] {
    background: rgba(3, 15, 35, 0.15) !important;
}

/* Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main content width and spacing */
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

/* Hero banner */
.hero-card {
    background: linear-gradient(
        115deg,
        rgba(7, 25, 55, 0.94),
        rgba(8, 100, 145, 0.78)
    );
    padding: 42px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.22);
    margin: 12px 0 32px 0;
    box-shadow: 0 14px 42px rgba(0,0,0,0.34);
    backdrop-filter: blur(12px);
}
.hero-label {
    color: #67E8F9;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}
.hero-title {
    color: #FFFFFF;
    font-size: clamp(30px, 4vw, 46px);
    font-weight: 800;
    margin: 14px 0;
}
.hero-description {
    color: #DBEAFE;
    font-size: 16px;
    line-height: 1.8;
    max-width: 850px;
}

/* Section headings */
.section-heading {
    color: #BAE6FD;
    font-size: 25px;
    font-weight: 750;
    margin: 26px 0 16px 0;
}

/* Feature cards */
.feature-card {
    background: rgba(10, 35, 75, 0.82);
    padding: 24px 16px;
    border-radius: 17px;
    border: 1px solid rgba(103,232,249,0.28);
    text-align: center;
    min-height: 155px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    backdrop-filter: blur(10px);
}
.feature-icon {font-size: 34px; margin-bottom: 8px;}
.feature-card h3 {
    color: #FFFFFF;
    font-size: 19px;
    margin: 5px 0;
}
.feature-card p {
    color: #BAE6FD;
    font-size: 13px;
    margin-bottom: 0;
}

/* Form panel */
.form-panel {
    background: rgba(8, 30, 62, 0.76);
    border: 1px solid rgba(125,211,252,0.25);
    border-radius: 18px;
    padding: 22px;
    margin: 10px 0 20px 0;
    backdrop-filter: blur(10px);
}

/* Labels and input controls */
.stApp label, .stApp [data-testid="stWidgetLabel"] p {
    color: #E0F2FE !important;
    font-weight: 600 !important;
}
.stNumberInput input {
    background: #F0F9FF !important;
    color: #102D50 !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] > div {
    background: #F0F9FF !important;
    color: #102D50 !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] * {
    color: #102D50;
}

/* Submit button */
.stFormSubmitButton button {
    width: 100%;
    min-height: 52px;
    background: linear-gradient(90deg, #0891B2, #2563EB);
    color: white !important;
    font-size: 17px;
    font-weight: 750;
    border: 0;
    border-radius: 11px;
    transition: 0.2s ease-in-out;
}
.stFormSubmitButton button:hover {
    background: linear-gradient(90deg, #06B6D4, #3B82F6);
    border: 1px solid #BAE6FD;
    transform: translateY(-2px);
}

/* Prediction result */
.result-card {
    background: linear-gradient(
        120deg,
        rgba(15,40,85,0.96),
        rgba(7,89,133,0.90)
    );
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #67E8F9;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.result-number {
    color: #FFFFFF;
    font-size: 52px;
    font-weight: 850;
    margin: 8px 0;
}
.result-caption {
    color: #BAE6FD;
    font-size: 14px;
}

/* Footer */
.footer-card {
    background: rgba(7,25,55,0.90);
    padding: 26px;
    border-radius: 17px;
    text-align: center;
    color: #DBEAFE;
    margin-top: 42px;
    border: 1px solid rgba(103,232,249,0.25);
}
.footer-title {
    color: #67E8F9;
    font-size: 17px;
    font-weight: 800;
    margin-bottom: 12px;
}

/* Mobile adjustments */
@media (max-width: 700px) {
    .hero-card {padding: 25px;}
    .hero-description {font-size: 14px;}
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. HERO BANNER
# Use st.html so HTML is rendered, not shown as code.
# =========================================================
st.html("""
<div class="hero-card">
    <div class="hero-label">INTELLIGENT BANKING ANALYTICS</div>
    <h1 class="hero-title">🏦 Customer Churn Prediction</h1>
    <p class="hero-description">
        Predict customer behavior, identify potential churn risks,
        and make smarter customer retention decisions using
        Artificial Intelligence and Deep Learning.
    </p>
    <div class="hero-label">POWERED BY ARTIFICIAL NEURAL NETWORKS</div>
</div>
""")

# =========================================================
# 5. PLATFORM OVERVIEW
# =========================================================
st.markdown(
    '<div class="section-heading">🚀 Platform Overview</div>',
    unsafe_allow_html=True
)

overview_cols = st.columns(3)

feature_data = [
    ("🤖", "AI Powered", "ANN Prediction Model"),
    ("📊", "Smart Analytics", "Customer Risk Analysis"),
    ("🎯", "Prediction", "Customer Retention Insights")
]

for column, (icon, title, description) in zip(overview_cols, feature_data):
    with column:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# 6. CUSTOMER INPUT FORM
# =========================================================
st.markdown(
    '<div class="section-heading">👤 Customer Information</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="color:#DBEAFE;">Enter customer details below to analyze the possibility of customer churn.</p>',
    unsafe_allow_html=True
)

with st.form("customer_prediction_form"):
    st.markdown("### 📝 Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        geography = st.selectbox(
            "Geography",
            options=list(one_hot_encoder.categories_[0])
        )
    with col2:
        gender = st.selectbox(
            "Gender",
            options=list(label_encoder.classes_)
        )
    with col3:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

    st.markdown("### 💳 Banking Information")

    col4, col5, col6 = st.columns(3)

    with col4:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650,
            step=1
        )
    with col5:
        tenure = st.number_input(
            "Tenure (Years)",
            min_value=0,
            max_value=10,
            value=3,
            step=1
        )
    with col6:
        balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

    st.markdown("### 📈 Account Activity")

    col7, col8, col9 = st.columns(3)

    with col7:
        num_of_products = st.selectbox(
            "Number of Products",
            options=[1, 2, 3, 4]
        )
    with col8:
        has_cr_card = st.selectbox(
            "Has Credit Card",
            options=["Yes", "No"]
        )
    with col9:
        is_active_member = st.selectbox(
            "Is Active Member",
            options=["Yes", "No"]
        )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.form_submit_button("🔍 Predict Customer Churn")

# =========================================================
# 7. PREPROCESS INPUT AND PREDICT
# =========================================================
if predict_button:
    try:
        gender_encoded = label_encoder.transform([gender])[0]

        input_data = pd.DataFrame({
            "Gender": [gender_encoded],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [1 if has_cr_card == "Yes" else 0],
            "IsActiveMember": [1 if is_active_member == "Yes" else 0],
            "EstimatedSalary": [estimated_salary],
            "CreditScore": [credit_score]
        })

        geography_encoded = one_hot_encoder.transform(
            pd.DataFrame({"Geography": [geography]})
        )

        if hasattr(geography_encoded, "toarray"):
            geography_encoded = geography_encoded.toarray()

        geography_df = pd.DataFrame(
            geography_encoded,
            columns=one_hot_encoder.get_feature_names_out(["Geography"])
        )

        input_data = pd.concat(
            [input_data.reset_index(drop=True), geography_df.reset_index(drop=True)],
            axis=1
        )

        # Ensure the same feature names and order used during scaler training.
        expected_features = list(scaler.feature_names_in_)
        missing_features = [
            feature for feature in expected_features
            if feature not in input_data.columns
        ]
        if missing_features:
            raise ValueError(
                f"Missing model input features: {missing_features}. "
                "Check that the encoders and scaler were saved from the same training pipeline."
            )

        input_data = input_data[expected_features]
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled, verbose=0)
        churn_probability = float(prediction.reshape(-1)[0])

        # Assumes the ANN output is a sigmoid probability between 0 and 1.
        if not 0 <= churn_probability <= 1:
            raise ValueError(
                "Model output is outside 0–1. Confirm that the model uses a sigmoid output layer."
            )

        churn_percentage = churn_probability * 100

        st.markdown(
            '<div class="section-heading">📊 Prediction Results</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="result-card">
            <div class="hero-label">CUSTOMER CHURN PROBABILITY</div>
            <div class="result-number">{churn_percentage:.2f}%</div>
            <div class="result-caption">Estimated score from the ANN model</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(churn_probability)

        if churn_probability > 0.5:
            st.error(
                "⚠️ High churn prediction: the model predicts that this customer is likely to churn."
            )
            st.markdown("""
            **Suggested retention actions**
            - Contact the customer to understand their concerns.
            - Review account activity and service usage.
            - Consider relevant, personalized retention offers.
            - Monitor the customer's engagement.
            """)
        else:
            st.success(
                "✅ Low churn prediction: the model does not currently predict that this customer will churn."
            )
            st.markdown("""
            **Suggested customer actions**
            - Maintain consistent service quality.
            - Continue regular customer engagement.
            - Recommend relevant banking products when appropriate.
            - Monitor customer satisfaction over time.
            """)

        st.caption(
            "Note: This is a model estimate, not a guarantee of future customer behavior. "
            "The 0.5 classification threshold should be validated against your model's evaluation results."
        )

    except Exception as error:
        st.error("Unable to generate a prediction. Check the input processing and saved model files.")
        st.exception(error)

# =========================================================
# 8. FOOTER
# =========================================================
st.html("""
<div class="footer-card">
    <div class="footer-title">CUSTOMER CHURN PREDICTION SYSTEM</div>
    <div>Built with Python | Streamlit | TensorFlow | Artificial Neural Networks</div>
    <br>
    <div>AI-Powered Customer Intelligence Platform</div>
    <br>
    <strong>Developed by Gaurav Katre</strong>
    <br><br>
    <div>© 2026 All Rights Reserved</div>
</div>
""")
