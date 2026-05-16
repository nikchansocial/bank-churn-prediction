import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="ECB Churn Predictor", page_icon="🏦", layout="wide")

# ECB Theme CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f5f6fa;
    }
    
    /* Header bar */
    .ecb-header {
        background: linear-gradient(135deg, #003299, #0050cc);
        padding: 20px 30px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 6px solid #FFD700;
    }
    .ecb-header h1 {
        color: white;
        font-size: 28px;
        margin: 0;
        font-weight: 700;
    }
    .ecb-header p {
        color: #FFD700;
        margin: 5px 0 0 0;
        font-size: 14px;
        letter-spacing: 1px;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border-top: 4px solid #003299;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #003299;
    }
    .metric-label {
        font-size: 13px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section headers */
    .section-header {
        background: white;
        border-radius: 8px;
        padding: 12px 20px;
        margin: 15px 0 10px 0;
        border-left: 5px solid #003299;
        box-shadow: 0 1px 5px rgba(0,0,0,0.05);
    }
    .section-header h3 {
        color: #003299;
        margin: 0;
        font-size: 16px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Risk badges */
    .risk-high {
        background: #fff0f0;
        border: 2px solid #e74c3c;
        border-radius: 8px;
        padding: 12px 16px;
        color: #c0392b;
        font-weight: 600;
        margin: 5px 0;
    }
    .risk-medium {
        background: #fff8e6;
        border: 2px solid #f39c12;
        border-radius: 8px;
        padding: 12px 16px;
        color: #d35400;
        font-weight: 600;
        margin: 5px 0;
    }
    .risk-low {
        background: #f0fff4;
        border: 2px solid #27ae60;
        border-radius: 8px;
        padding: 12px 16px;
        color: #1e8449;
        font-weight: 600;
        margin: 5px 0;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #003299 !important;
    }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: white !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #FFD700 !important;
    }

    /* Performance table */
    .perf-card {
        background: #003299;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        color: white;
    }
    .perf-value {
        font-size: 28px;
        font-weight: 700;
        color: #FFD700;
    }
    .perf-label {
        font-size: 12px;
        color: #aac4ff;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ECB Header
st.markdown("""
<div class="ecb-header">
    <h1>🏦 European Central Bank</h1>
    <p>CUSTOMER CHURN PREDICTION SYSTEM — RETAIL BANKING DIVISION</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## ⚙️ Customer Profile")
st.sidebar.markdown("---")

credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
age = st.sidebar.slider("Age", 18, 92, 40)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Account Balance (€)", 0, 250000, 50000)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.sidebar.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
is_active = st.sidebar.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
salary = st.sidebar.number_input("Estimated Salary (€)", 0, 200000, 100000)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

# Feature engineering
balance_salary_ratio = balance / (salary + 1)
age_tenure_interaction = ((age - 38.92) / 10.49) * ((tenure - 5.01) / 2.89)
product_density = num_products / (tenure + 1)

# Encode
geo_france = 1 if geography == "France" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
gender_female = 1 if gender == "Female" else 0
gender_male = 1 if gender == "Male" else 0

# Scale
scaler = StandardScaler()
numerical = np.array([[credit_score, age, tenure, balance, salary]])
scaled = scaler.fit_transform(numerical)[0]

# Input dataframe
input_data = pd.DataFrame([[
    scaled[0], scaled[1], scaled[2], scaled[3],
    num_products, has_cr_card, is_active, scaled[4],
    geo_france, geo_germany, geo_spain,
    gender_female, gender_male,
    balance_salary_ratio, age_tenure_interaction, product_density
]], columns=[
    'CreditScore', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain',
    'Gender_Female', 'Gender_Male',
    'Balance_Salary_Ratio', 'Age_Tenure_Interaction', 'Product_Density'
])

# Load model
@st.cache_resource
def load_model():
    df = pd.read_csv('European_Bank.csv')
    df = df.drop(['CustomerId', 'Surname', 'Year'], axis=1)
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=False, dtype=int)
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    X_scaled = X.copy()
    sc = StandardScaler()
    X_scaled[['CreditScore','Age','Tenure','Balance','EstimatedSalary']] = sc.fit_transform(
        X[['CreditScore','Age','Tenure','Balance','EstimatedSalary']]
    )
    X_scaled['Balance_Salary_Ratio'] = X_scaled['Balance'] / (X_scaled['EstimatedSalary'] + 1)
    X_scaled['Age_Tenure_Interaction'] = X_scaled['Age'] * X_scaled['Tenure']
    X_scaled['Product_Density'] = X_scaled['NumOfProducts'] / (X_scaled['Tenure'] + 1)
    model = GradientBoostingClassifier(random_state=42, n_estimators=100)
    model.fit(X_scaled, y)
    return model

model = load_model()

# Prediction
prob = model.predict_proba(input_data)[0][1]
prediction = model.predict(input_data)[0]
risk_level = "HIGH RISK" if prob > 0.6 else "MEDIUM RISK" if prob > 0.3 else "LOW RISK"
risk_color = "#e74c3c" if prob > 0.6 else "#f39c12" if prob > 0.3 else "#27ae60"

# Churn Risk Assessment
st.markdown("""
<div class="section-header">
    <h3>🎯 Churn Risk Assessment</h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{prob*100:.1f}%</div>
        <div class="metric-label">Churn Probability</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{risk_color}">{risk_level}</div>
        <div class="metric-label">Risk Classification</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    pred_text = "⚠️ Will Churn" if prediction == 1 else "✅ Will Stay"
    pred_color = "#e74c3c" if prediction == 1 else "#27ae60"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{pred_color}; font-size:24px">{pred_text}</div>
        <div class="metric-label">Model Prediction</div>
    </div>
    """, unsafe_allow_html=True)

# Probability meter
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div class="section-header">
    <h3>📊 Churn Probability Meter</h3>
</div>
""", unsafe_allow_html=True)
st.progress(float(prob))
st.markdown(f"<p style='text-align:center; color:#003299; font-weight:600;'>Churn Risk: {prob*100:.1f}%</p>", unsafe_allow_html=True)

# Key Risk Factors
st.markdown("""
<div class="section-header">
    <h3>⚠️ Key Risk Factors</h3>
</div>
""", unsafe_allow_html=True)

risks = []
if num_products >= 3:
    risks.append(("high", "🔴 3+ Products Detected — Extremely high churn risk. Overselling detected."))
if is_active == 0:
    risks.append(("high", "🔴 Inactive Member — Customer disengagement is a major churn signal."))
if geography == "Germany":
    risks.append(("medium", "🟡 German Market — Highest churn region at 32.4% rate."))
if age >= 40 and age <= 55:
    risks.append(("medium", "🟡 Age 40-55 — Peak churn risk age group. Likely seeking better returns."))
if balance > 100000:
    risks.append(("medium", "🟡 High Balance Customer — Vulnerable to competitor interest rate offers."))
if not risks:
    risks.append(("low", "🟢 No major risk factors detected. Customer profile is stable."))

for risk_type, message in risks:
    css_class = f"risk-{risk_type}"
    st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)

# Model Performance
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <h3>📈 Model Performance Summary</h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
metrics = [("Gradient Boosting", "Model"), ("87%", "Accuracy"), ("50%", "Recall"), ("0.8675", "ROC-AUC")]
cols = [col1, col2, col3, col4]

for col, (value, label) in zip(cols, metrics):
    with col:
        st.markdown(f"""
        <div class="perf-card">
            <div class="perf-value">{value}</div>
            <div class="perf-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#999; font-size:12px; padding:20px; border-top:1px solid #ddd;">
    European Central Bank — Customer Intelligence Division | 
    Churn Prediction Model v1.0 | 
    Powered by Gradient Boosting ML
</div>
""", unsafe_allow_html=True)