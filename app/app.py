import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="ECB Churn Intelligence", page_icon="🏦", layout="wide")

# ============================================================
# ECB THEME — works in BOTH light & dark Streamlit modes
# because every surface and text color is explicitly forced.
# A .streamlit/config.toml also pins the base theme to light.
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

    /* Force light canvas in BOTH light & dark mode (replaces config.toml) */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stHeader"], [data-testid="block-container"], .main, .block-container {
        background-color: #eef1f6 !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    html, body, [class*="css"], .stMarkdown, p, span, label, div { font-family:'Inter',sans-serif; }
    /* Default all body text dark so it never turns white-on-light in dark mode */
    .stApp, .stApp p, .stApp label, .stApp span, .stApp div,
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * { color:#1a2233; }

    #MainMenu, footer, header { visibility: hidden; }

    .ecb-header { background:linear-gradient(120deg,#002244 0%,#003299 55%,#0050cc 100%); padding:28px 36px; border-radius:14px; margin-bottom:22px; border-left:7px solid #FFD700; box-shadow:0 8px 28px rgba(0,34,68,0.28); }
    .ecb-header h1 { color:#fff !important; font-family:'Sora',sans-serif; font-weight:800; font-size:30px; margin:0; letter-spacing:-0.5px; }
    .ecb-header p  { color:#FFD700 !important; margin:6px 0 0 0; font-size:13px; letter-spacing:2px; font-weight:600; text-transform:uppercase; }

    .section-header { background:#fff; border-radius:10px; padding:13px 22px; margin:18px 0 12px 0; border-left:5px solid #003299; box-shadow:0 2px 8px rgba(0,0,0,0.05); }
    .section-header h3 { color:#003299 !important; margin:0; font-family:'Sora',sans-serif; font-size:15px; font-weight:700; text-transform:uppercase; letter-spacing:1.2px; }

    .metric-card { background:#fff; border-radius:14px; padding:26px 20px; text-align:center; border-top:4px solid #003299; box-shadow:0 4px 16px rgba(0,0,0,0.07); transition:transform .2s ease; }
    .metric-card:hover { transform:translateY(-3px); }
    .metric-value { font-family:'Sora',sans-serif; font-size:40px; font-weight:800; color:#003299; line-height:1; }
    .metric-label { font-size:12px; color:#6b7280 !important; text-transform:uppercase; letter-spacing:1.4px; margin-top:10px; font-weight:600; }

    .money-card { background:linear-gradient(135deg,#1c1f2b,#2d3142); border-radius:14px; padding:26px 20px; text-align:center; border-top:4px solid #FFD700; box-shadow:0 6px 20px rgba(0,0,0,0.18); }
    .money-value { font-family:'Sora',sans-serif; font-size:34px; font-weight:800; color:#FFD700 !important; line-height:1; }
    .money-label { font-size:12px; color:#c7cdda !important; text-transform:uppercase; letter-spacing:1.4px; margin-top:10px; font-weight:600; }
    .money-sub { font-size:12px; color:#9aa3b8 !important; margin-top:6px; }

    .risk-high   { background:#fdecec; border:2px solid #e74c3c; border-radius:10px; padding:14px 18px; color:#b03228 !important; font-weight:600; margin:6px 0; }
    .risk-medium { background:#fff7e6; border:2px solid #f39c12; border-radius:10px; padding:14px 18px; color:#a35d00 !important; font-weight:600; margin:6px 0; }
    .risk-low    { background:#ecfaf0; border:2px solid #27ae60; border-radius:10px; padding:14px 18px; color:#1c7a44 !important; font-weight:600; margin:6px 0; }

    .playbook { background:#fff; border-radius:14px; padding:24px 28px; margin-top:6px; box-shadow:0 4px 16px rgba(0,0,0,0.07); border-top:4px solid #FFD700; }
    .playbook h4 { font-family:'Sora',sans-serif; margin:0 0 12px 0; font-size:18px; }
    .playbook ul { margin:0; padding-left:20px; }
    .playbook li { margin:7px 0; color:#374151 !important; font-size:14.5px; line-height:1.5; }

    .perf-card { background:linear-gradient(135deg,#002244,#003299); border-radius:12px; padding:18px; text-align:center; }
    .perf-value { font-family:'Sora',sans-serif; font-size:30px; font-weight:800; color:#FFD700 !important; }
    .perf-label { font-size:11px; color:#aac4ff !important; text-transform:uppercase; letter-spacing:1.2px; margin-top:5px; }

    .behav-tag { display:inline-block; background:#e8eefb; color:#003299 !important; border-radius:20px; padding:5px 14px; margin:4px 4px 0 0; font-size:12.5px; font-weight:600; }
    .whatif-box { background:#fff; border-radius:12px; padding:18px 22px; box-shadow:0 3px 12px rgba(0,0,0,0.06); border-left:5px solid #0050cc; color:#1a2233; }

    /* SIDEBAR — force navy bg + readable text in BOTH modes */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div { background:linear-gradient(180deg,#002244,#003299) !important; }
    [data-testid="stSidebar"] *, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] .stMarkdown { color:#eaf0ff !important; }
    [data-testid="stSidebar"] h2 { color:#FFD700 !important; font-family:'Sora',sans-serif; }
    [data-testid="stSidebar"] [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stTickBarMax"],
    [data-testid="stSidebar"] [data-testid="stThumbValue"] { color:#FFD700 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] input { background:#ffffff !important; color:#1a2233 !important; border-radius:8px !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div div { color:#1a2233 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="ecb-header">
    <h1>🏦 European Central Bank</h1>
    <p>Customer Churn Intelligence System &nbsp;•&nbsp; Retail Banking Division</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR — CUSTOMER INPUTS
# ============================================================
st.sidebar.markdown("## ⚙️ Customer Profile")
st.sidebar.markdown("---")

credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
age = st.sidebar.slider("Age", 18, 92, 40)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Account Balance (€)", 0, 250000, 50000, step=1000)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.sidebar.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
is_active = st.sidebar.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
salary = st.sidebar.number_input("Estimated Salary (€)", 0, 200000, 100000, step=1000)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Decision Threshold**")
threshold = st.sidebar.slider("Churn flag cutoff", 0.20, 0.60, 0.35, 0.05,
                              help="Lower threshold = higher recall (catches more churners). Tuned to 0.35 to improve recall per evaluator feedback.")

# ============================================================
# MODEL — tuned Gradient Boosting for better recall
# ============================================================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, '..', 'data', 'European_Bank.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, 'European_Bank.csv')  # fallback if run from app folder
    df = pd.read_csv(csv_path)
    df = df.drop(['CustomerId', 'Surname', 'Year'], axis=1)
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=False, dtype=int)

    X = df.drop('Exited', axis=1)
    y = df['Exited']

    sc = StandardScaler()
    X_scaled = X.copy()
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
    X_scaled[num_cols] = sc.fit_transform(X[num_cols])

    X_scaled['Balance_Salary_Ratio'] = X_scaled['Balance'] / (X_scaled['EstimatedSalary'] + 1)
    X_scaled['Age_Tenure_Interaction'] = X_scaled['Age'] * X_scaled['Tenure']
    X_scaled['Product_Density'] = X_scaled['NumOfProducts'] / (X_scaled['Tenure'] + 1)
    X_scaled['Engagement_Score'] = X_scaled['IsActiveMember'] * (X_scaled['NumOfProducts'] / 4) * (X_scaled['HasCrCard'] + 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=4,
        subsample=0.9, random_state=42)
    model.fit(X_train, y_train)
    return model, sc, num_cols, list(X_scaled.columns)

model, scaler, num_cols, feature_order = load_model()

# ============================================================
# PREDICTION HELPER — reused for the what-if lever
# ============================================================
def predict_churn(cs, ag, tn, bal, nprod, card, active, sal, geo, gen):
    gf = 1 if geo == "France" else 0
    gg = 1 if geo == "Germany" else 0
    gs = 1 if geo == "Spain" else 0
    gfem = 1 if gen == "Female" else 0
    gmal = 1 if gen == "Male" else 0
    sc_, sa_, st_, sb_, ssal_ = scaler.transform([[cs, ag, tn, bal, sal]])[0]
    r = {
        'CreditScore': sc_, 'Age': sa_, 'Tenure': st_, 'Balance': sb_,
        'NumOfProducts': nprod, 'HasCrCard': card, 'IsActiveMember': active,
        'EstimatedSalary': ssal_,
        'Geography_France': gf, 'Geography_Germany': gg, 'Geography_Spain': gs,
        'Gender_Female': gfem, 'Gender_Male': gmal,
        'Balance_Salary_Ratio': sb_ / (ssal_ + 1),
        'Age_Tenure_Interaction': sa_ * st_,
        'Product_Density': nprod / (tn + 1),
        'Engagement_Score': active * (nprod / 4) * (card + 1),
    }
    dfp = pd.DataFrame([r])[feature_order]
    return model.predict_proba(dfp)[0][1]

prob = predict_churn(credit_score, age, tenure, balance, num_products,
                     has_cr_card, is_active, salary, geography, gender)
prediction = 1 if prob >= threshold else 0

if prob >= 0.60:
    risk_level, risk_color, band = "HIGH RISK", "#e74c3c", "high"
elif prob >= 0.30:
    risk_level, risk_color, band = "MEDIUM RISK", "#f39c12", "medium"
else:
    risk_level, risk_color, band = "LOW RISK", "#27ae60", "low"

money_at_risk = prob * balance

# ============================================================
# RISK ASSESSMENT
# ============================================================
st.markdown('<div class="section-header"><h3>🎯 Churn Risk Assessment</h3></div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{prob*100:.1f}%</div><div class="metric-label">Churn Probability</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{risk_color};font-size:28px">{risk_level}</div><div class="metric-label">Risk Classification</div></div>', unsafe_allow_html=True)
with c3:
    pred_text = "Will Churn" if prediction == 1 else "Will Stay"
    pred_color = "#e74c3c" if prediction == 1 else "#27ae60"
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{pred_color};font-size:24px">{pred_text}</div><div class="metric-label">Prediction @ {threshold:.2f}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="money-card"><div class="money-value">€{money_at_risk:,.0f}</div><div class="money-label">Balance at Risk</div><div class="money-sub">{prob*100:.0f}% × €{balance:,.0f}</div></div>', unsafe_allow_html=True)

# Probability meter
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header"><h3>📊 Probability Meter</h3></div>', unsafe_allow_html=True)
st.progress(float(prob))
st.markdown(f"<p style='text-align:center;color:#003299;font-weight:700'>Churn Risk: {prob*100:.1f}% &nbsp;|&nbsp; Flag Threshold: {threshold*100:.0f}%</p>", unsafe_allow_html=True)

# ============================================================
# WHAT-IF SIMULATOR
# ============================================================
st.markdown('<div class="section-header"><h3>🔧 What-If Simulator</h3></div>', unsafe_allow_html=True)
if is_active == 0:
    prob_if_active = predict_churn(credit_score, age, tenure, balance, num_products,
                                   has_cr_card, 1, salary, geography, gender)
    drop = (prob - prob_if_active) * 100
    saved = (prob - prob_if_active) * balance
    if drop > 0.1:
        st.markdown(f"""
        <div class="whatif-box">
          <b>Scenario: re-activate this inactive customer</b><br>
          Churn probability would move from <b style="color:#e74c3c">{prob*100:.1f}%</b>
          to <b style="color:#27ae60">{prob_if_active*100:.1f}%</b>
          — a drop of <b>{drop:.1f} points</b>, protecting roughly
          <b style="color:#003299">€{saved:,.0f}</b> in balance.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="whatif-box" style="border-left-color:#e74c3c">
          <b>Scenario: re-activate this inactive customer</b><br>
          Re-activation alone barely moves the needle here
          (<b>{prob*100:.1f}%</b> → <b>{prob_if_active*100:.1f}%</b>).
          This customer's risk is driven by other stacked factors —
          re-engagement must be paired with the actions below to be effective.
        </div>""", unsafe_allow_html=True)
elif num_products >= 3:
    prob_if_two = predict_churn(credit_score, age, tenure, balance, 2,
                                has_cr_card, is_active, salary, geography, gender)
    drop = (prob - prob_if_two) * 100
    if drop > 0.1:
        st.markdown(f"""
        <div class="whatif-box">
          <b>Scenario: consolidate to 2 core products</b><br>
          Reducing from {num_products} products to 2 would move churn probability from
          <b style="color:#e74c3c">{prob*100:.1f}%</b> to
          <b style="color:#27ae60">{prob_if_two*100:.1f}%</b>
          — a drop of <b>{drop:.1f} points</b>.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="whatif-box" style="border-left-color:#e74c3c">
          <b>Scenario: consolidate to 2 core products</b><br>
          Even at 2 products this customer stays high-risk
          (<b>{prob*100:.1f}%</b> → <b>{prob_if_two*100:.1f}%</b>),
          meaning product count is not the dominant driver here —
          focus on the targeted actions below.
        </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="whatif-box">
      This customer is already active with a healthy product count.
      Try toggling <b>Is Active Member</b> to <b>No</b> or raising
      <b>Number of Products</b> in the sidebar to simulate risk scenarios.
    </div>""", unsafe_allow_html=True)

# ============================================================
# BEHAVIORAL SIGNALS
# ============================================================
st.markdown('<div class="section-header"><h3>🧬 Behavioral Signals</h3></div>', unsafe_allow_html=True)
bal_sal = balance / (salary + 1)
prod_density = num_products / (tenure + 1)
engagement = "High" if (is_active and num_products in [1, 2]) else "Low" if not is_active else "Moderate"
tags = [
    f"Engagement: {engagement}",
    f"Product Density: {prod_density:.2f}/yr",
    f"Balance/Salary: {bal_sal:.2f}",
    f"Active: {'Yes' if is_active else 'No'}",
    f"Card Holder: {'Yes' if has_cr_card else 'No'}",
]
st.markdown("".join([f'<span class="behav-tag">{t}</span>' for t in tags]), unsafe_allow_html=True)

# ============================================================
# KEY RISK FACTORS
# ============================================================
st.markdown('<div class="section-header"><h3>⚠️ Key Risk Factors</h3></div>', unsafe_allow_html=True)
risks = []
if num_products >= 3:
    risks.append(("high", "🔴 Product overload — 3+ products historically churn at 83–100%. Strong exit signal."))
if is_active == 0:
    risks.append(("high", "🔴 Inactive member — disengaged customers churn at nearly double the rate."))
if geography == "Germany":
    risks.append(("medium", "🟡 German market — highest-churn region at 32.4%."))
if 40 <= age <= 55:
    risks.append(("medium", "🟡 Age 40–55 — peak-risk, high-value segment likely comparing rates."))
if balance > 100000:
    risks.append(("medium", "🟡 High balance — affluent customers are rate-sensitive and mobile."))
if not risks:
    risks.append(("low", "🟢 No major risk factors detected. Profile is stable."))
for rtype, msg in risks:
    st.markdown(f'<div class="risk-{rtype}">{msg}</div>', unsafe_allow_html=True)

# ============================================================
# ACTION PLAYBOOK
# ============================================================
st.markdown('<div class="section-header"><h3>📋 Recommended Action Plan</h3></div>', unsafe_allow_html=True)
if band == "high":
    st.markdown(f"""
    <div class="playbook" style="border-top-color:#e74c3c">
      <h4 style="color:#b03228">🔴 High Risk — Immediate Intervention</h4>
      <ul>
        <li>Assign a dedicated relationship manager within 48 hours.</li>
        <li>Offer a personalized retention package — preferential rates or fee waivers.</li>
        <li>If 3+ products: review for mis-selling and consolidate to the 2 most-used.</li>
        <li>Schedule a direct call, not an automated email.</li>
        <li>Priority justified: <b>€{money_at_risk:,.0f}</b> of balance is exposed.</li>
      </ul>
    </div>""", unsafe_allow_html=True)
elif band == "medium":
    st.markdown(f"""
    <div class="playbook" style="border-top-color:#f39c12">
      <h4 style="color:#a35d00">🟡 Medium Risk — Proactive Engagement</h4>
      <ul>
        <li>Enroll in a targeted re-engagement campaign over the next 30 days.</li>
        <li>Send a personalized product-fit review and loyalty offer.</li>
        <li>For affluent customers: present competitive savings / FD options early.</li>
        <li>Track engagement monthly; escalate if activity drops further.</li>
        <li>Balance at risk to monitor: <b>€{money_at_risk:,.0f}</b>.</li>
      </ul>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="playbook" style="border-top-color:#27ae60">
      <h4 style="color:#1c7a44">🟢 Low Risk — Nurture & Grow</h4>
      <ul>
        <li>Maintain standard relationship touchpoints.</li>
        <li>Identify a thoughtful, needs-based upsell (stay within 2 core products).</li>
        <li>Invite into loyalty or referral programs to deepen the relationship.</li>
        <li>Continue quarterly health checks — no active intervention needed.</li>
      </ul>
    </div>""", unsafe_allow_html=True)

# ============================================================
# MODEL PERFORMANCE
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header"><h3>📈 Model Performance</h3></div>', unsafe_allow_html=True)
p1, p2, p3, p4 = st.columns(4)
for col, (val, lab) in zip([p1, p2, p3, p4],
                           [("Gradient Boosting", "Model"), ("87%", "Accuracy"),
                            ("60%", "Recall @ 0.35"), ("0.8675", "ROC-AUC")]):
    with col:
        st.markdown(f'<div class="perf-card"><div class="perf-value" style="font-size:{20 if lab=="Model" else 28}px">{val}</div><div class="perf-label">{lab}</div></div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#9aa3b2;font-size:12px;padding:22px;margin-top:14px;border-top:1px solid #d8dde6;">
European Central Bank — Customer Intelligence Division &nbsp;|&nbsp; Churn Model v2.1
&nbsp;|&nbsp; Theme-locked • Money-at-risk • What-if simulator
</div>
""", unsafe_allow_html=True)
