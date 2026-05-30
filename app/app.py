import os
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Churn Intelligence", page_icon="◆", layout="wide")

# ============================================================
# THEME TOGGLE — top of main area (read before CSS is built)
# ============================================================
_, tcol = st.columns([4, 1])
with tcol:
    mode = st.radio("Theme", ["🌙 Dark", "☀️ Light"], horizontal=True,
                    index=0, label_visibility="collapsed")
dark = mode.startswith("🌙")

# Indigo + soft slate palette, two variants
if dark:
    T = dict(
        bg="#0e1117", panel="#171a23", panel2="#1d2130", text="#e6e8ee",
        muted="#9aa1b2", border="#272b38", accent="#6366f1", accent_soft="#8b8ff5",
        good="#34d399", warn="#fbbf24", bad="#f87171",
        money_bg="linear-gradient(135deg,#1e2030,#2a2d44)", shadow="0 6px 24px rgba(0,0,0,0.45)",
    )
else:
    T = dict(
        bg="#f4f5f9", panel="#ffffff", panel2="#fbfbfe", text="#1a1d29",
        muted="#6b7180", border="#e6e8f0", accent="#4f46e5", accent_soft="#6366f1",
        good="#059669", warn="#d97706", bad="#dc2626",
        money_bg="linear-gradient(135deg,#1e2030,#33365a)", shadow="0 4px 18px rgba(20,22,40,0.08)",
    )

# ============================================================
# STYLES — driven entirely by the palette above
# ============================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="block-container"], .main, .block-container {{ background:{T['bg']} !important; }}
    [data-testid="stHeader"] {{ background:transparent !important; }}
    html, body, .stMarkdown, .stMarkdown p, .stMarkdown span {{ font-family:'Inter',sans-serif; }}
    [data-testid="stMain"] .stMarkdown, [data-testid="stMain"] .stMarkdown * {{ color:{T['text']}; }}
    #MainMenu, footer {{ visibility:hidden; }}

    /* compact page rhythm */
    [data-testid="block-container"] {{ padding-top:1.2rem; padding-bottom:1rem; max-width:1150px; }}
    [data-testid="stVerticalBlock"] {{ gap:0.5rem; }}

    /* theme radio: small, right-aligned, subtle */
    div[role="radiogroup"] {{ justify-content:flex-end; gap:4px; }}
    div[role="radiogroup"] label {{ font-size:12px !important; padding:2px 8px; }}

    /* ===== COMPACT HEADER ===== */
    .hd {{ display:flex; align-items:center; gap:11px; padding:11px 18px; border-radius:13px;
          background:{T['panel']}; border:1px solid {T['border']};
          box-shadow:{T['shadow']}; margin:2px 0 14px 0; }}
    .hd .mark {{ width:32px; height:32px; border-radius:9px; flex:0 0 auto;
          background:linear-gradient(135deg,{T['accent']},{T['accent_soft']});
          display:flex; align-items:center; justify-content:center; color:#fff; font-size:16px; font-weight:800;
          box-shadow:0 3px 11px {T['accent']}55; }}
    .hd h1 {{ font-family:'Sora',sans-serif; font-weight:800; font-size:17px; margin:0; color:{T['text']} !important; letter-spacing:-0.2px; }}
    .hd p  {{ margin:1px 0 0 0; font-size:11px; color:{T['muted']} !important; letter-spacing:0.5px; font-weight:500; }}

    /* ===== SECTION LABEL ===== */
    .sec {{ font-family:'Sora',sans-serif; font-size:11.5px; font-weight:700; letter-spacing:1.4px;
           text-transform:uppercase; color:{T['muted']} !important; margin:13px 0 6px 2px; }}

    /* ===== CARDS ===== */
    .card {{ background:{T['panel']}; border:1px solid {T['border']}; border-radius:13px;
            padding:17px 15px; text-align:center; box-shadow:{T['shadow']};
            transition:transform .18s ease, border-color .18s ease; height:100%; }}
    .card:hover {{ transform:translateY(-2px); border-color:{T['accent']}; }}
    .card .v {{ font-family:'Sora',sans-serif; font-size:28px; font-weight:800; line-height:1; color:{T['accent']} !important; }}
    .card .l {{ font-size:11px; color:{T['muted']} !important; text-transform:uppercase; letter-spacing:1.1px; margin-top:7px; font-weight:600; }}

    .money {{ background:{T['money_bg']}; border:1px solid {T['border']}; border-radius:13px; padding:17px 15px; text-align:center; box-shadow:{T['shadow']}; height:100%; }}
    .money .v {{ font-family:'Sora',sans-serif; font-size:24px; font-weight:800; color:#a5b4fc !important; line-height:1; }}
    .money .l {{ font-size:11px; color:#c7cbe0 !important; text-transform:uppercase; letter-spacing:1.1px; margin-top:7px; font-weight:600; }}
    .money .s {{ font-size:10.5px; color:#9aa1c4 !important; margin-top:4px; }}

    /* ===== PILLS ===== */
    .pill {{ display:block; border-radius:10px; padding:10px 14px; margin:5px 0; font-size:13px; font-weight:500;
            border:1px solid {T['border']}; background:{T['panel2']}; }}
    .pill.bad  {{ border-left:4px solid {T['bad']};  color:{T['text']} !important; }}
    .pill.warn {{ border-left:4px solid {T['warn']}; color:{T['text']} !important; }}
    .pill.good {{ border-left:4px solid {T['good']}; color:{T['text']} !important; }}

    /* ===== BOXES ===== */
    .box {{ background:{T['panel']}; border:1px solid {T['border']}; border-radius:13px; padding:16px 20px; box-shadow:{T['shadow']}; }}
    .box.accent {{ border-left:4px solid {T['accent']}; }}
    .box h4 {{ font-family:'Sora',sans-serif; margin:0 0 7px 0; font-size:14.5px; color:{T['text']} !important; }}
    .box ul {{ margin:0; padding-left:18px; }}
    .box li {{ margin:4px 0; color:{T['text']} !important; font-size:13px; line-height:1.4; opacity:0.9; }}
    .box p  {{ color:{T['text']} !important; font-size:13px; margin:0; line-height:1.5; }}

    .tag {{ display:inline-block; background:{T['panel2']}; border:1px solid {T['border']};
           color:{T['accent_soft']} !important; border-radius:20px; padding:5px 12px; margin:4px 4px 0 0; font-size:11.5px; font-weight:600; }}

    /* ===== FOOTER ===== */
    .ft {{ text-align:center; margin-top:20px; padding:14px; border-top:1px solid {T['border']}; }}
    .ft .name {{ font-family:'Sora',sans-serif; font-weight:700; font-size:12.5px; color:{T['text']} !important; }}
    .ft .name span {{ color:{T['accent']} !important; }}
    .ft .meta {{ font-size:10.5px; color:{T['muted']} !important; margin-top:3px; letter-spacing:0.4px; }}

    /* ===== SIDEBAR (light-touch) ===== */
    [data-testid="stSidebar"] {{ background:{T['panel']} !important; border-right:1px solid {T['border']}; }}
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {{ color:{T['text']} !important; }}
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] p {{ color:{T['muted']} !important; }}
    .sgroup {{ font-family:'Sora',sans-serif; font-size:11px; font-weight:700; letter-spacing:1px;
              text-transform:uppercase; color:{T['accent']} !important; margin:13px 0 2px 2px; }}

    .stProgress > div > div > div > div {{ background:{T['accent']} !important; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# COMPACT HEADER
# ============================================================
st.markdown("""
<div class="hd">
    <div class="mark">◆</div>
    <div>
        <h1>Customer Churn Intelligence</h1>
        <p>Retail Banking · Predictive Retention Analytics</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR INPUTS
# ============================================================
st.sidebar.markdown("## Customer Profile")
st.sidebar.caption("Adjust the inputs to score a customer.")

st.sidebar.markdown('<div class="sgroup">Demographics</div>', unsafe_allow_html=True)
age = st.sidebar.slider("Age", 18, 92, 40)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])

st.sidebar.markdown('<div class="sgroup">Account & Financials</div>', unsafe_allow_html=True)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
balance = st.sidebar.number_input("Account Balance (€)", 0, 250000, 50000, step=1000)
salary = st.sidebar.number_input("Estimated Salary (€)", 0, 200000, 100000, step=1000)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)

st.sidebar.markdown('<div class="sgroup">Engagement</div>', unsafe_allow_html=True)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.sidebar.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
is_active = st.sidebar.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

st.sidebar.markdown('<div class="sgroup">Model Setting</div>', unsafe_allow_html=True)
threshold = st.sidebar.slider("Churn flag cutoff", 0.20, 0.60, 0.35, 0.05,
                              help="Lower threshold = higher recall (catches more churners). Tuned to 0.35.")

# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, '..', 'data', 'European_Bank.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, 'European_Bank.csv')
    df = pd.read_csv(csv_path)
    df = df.drop(['CustomerId', 'Surname', 'Year'], axis=1)
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=False, dtype=int)
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    sc = StandardScaler()
    Xs = X.copy()
    ncols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
    Xs[ncols] = sc.fit_transform(X[ncols])
    Xs['Balance_Salary_Ratio'] = Xs['Balance'] / (Xs['EstimatedSalary'] + 1)
    Xs['Age_Tenure_Interaction'] = Xs['Age'] * Xs['Tenure']
    Xs['Product_Density'] = Xs['NumOfProducts'] / (Xs['Tenure'] + 1)
    Xs['Engagement_Score'] = Xs['IsActiveMember'] * (Xs['NumOfProducts'] / 4) * (Xs['HasCrCard'] + 1)
    Xtr, _, ytr, _ = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=4,
                                       subsample=0.9, random_state=42)
    model.fit(Xtr, ytr)
    return model, sc, list(Xs.columns)

model, scaler, feature_order = load_model()

def predict_churn(cs, ag, tn, bal, nprod, card, active, sal, geo, gen):
    gf = 1 if geo == "France" else 0
    gg = 1 if geo == "Germany" else 0
    gs = 1 if geo == "Spain" else 0
    gfem = 1 if gen == "Female" else 0
    gmal = 1 if gen == "Male" else 0
    s = scaler.transform([[cs, ag, tn, bal, sal]])[0]
    r = {
        'CreditScore': s[0], 'Age': s[1], 'Tenure': s[2], 'Balance': s[3],
        'NumOfProducts': nprod, 'HasCrCard': card, 'IsActiveMember': active, 'EstimatedSalary': s[4],
        'Geography_France': gf, 'Geography_Germany': gg, 'Geography_Spain': gs,
        'Gender_Female': gfem, 'Gender_Male': gmal,
        'Balance_Salary_Ratio': s[3] / (s[4] + 1),
        'Age_Tenure_Interaction': s[1] * s[2],
        'Product_Density': nprod / (tn + 1),
        'Engagement_Score': active * (nprod / 4) * (card + 1),
    }
    return model.predict_proba(pd.DataFrame([r])[feature_order])[0][1]

prob = predict_churn(credit_score, age, tenure, balance, num_products,
                     has_cr_card, is_active, salary, geography, gender)
prediction = 1 if prob >= threshold else 0

if prob >= 0.60:
    risk_level, rc, band = "HIGH RISK", T['bad'], "high"
elif prob >= 0.30:
    risk_level, rc, band = "MEDIUM RISK", T['warn'], "medium"
else:
    risk_level, rc, band = "LOW RISK", T['good'], "low"

money_at_risk = prob * balance

# ============================================================
# RISK ASSESSMENT
# ============================================================
st.markdown('<div class="sec">Risk Assessment</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="card"><div class="v">{prob*100:.1f}%</div><div class="l">Churn Probability</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="card"><div class="v" style="color:{rc} !important;font-size:21px">{risk_level}</div><div class="l">Risk Level</div></div>', unsafe_allow_html=True)
with c3:
    ptxt = "Will Churn" if prediction == 1 else "Will Stay"
    pcol = T['bad'] if prediction == 1 else T['good']
    st.markdown(f'<div class="card"><div class="v" style="color:{pcol} !important;font-size:19px">{ptxt}</div><div class="l">Prediction @ {threshold:.2f}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="money"><div class="v">€{money_at_risk:,.0f}</div><div class="l">Balance at Risk</div><div class="s">{prob*100:.0f}% × €{balance:,.0f}</div></div>', unsafe_allow_html=True)

# Probability meter
st.markdown('<div class="sec">Probability Meter</div>', unsafe_allow_html=True)
st.progress(float(prob))
st.markdown(f"<p style='text-align:center;color:{T['muted']};font-size:12px;font-weight:600'>Churn Risk {prob*100:.1f}% · Flag Threshold {threshold*100:.0f}%</p>", unsafe_allow_html=True)

# ============================================================
# WHAT-IF + BEHAVIORAL SIGNALS
# ============================================================
left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="sec">What-If Simulator</div>', unsafe_allow_html=True)
    if is_active == 0:
        p2 = predict_churn(credit_score, age, tenure, balance, num_products, has_cr_card, 1, salary, geography, gender)
        drop = (prob - p2) * 100
        saved = (prob - p2) * balance
        if drop > 0.1:
            st.markdown(f'<div class="box accent"><p><b>Re-activate this customer</b><br>Churn moves <b style="color:{T["bad"]}">{prob*100:.1f}%</b> → <b style="color:{T["good"]}">{p2*100:.1f}%</b> (−{drop:.1f} pts), protecting ≈ <b style="color:{T["accent_soft"]}">€{saved:,.0f}</b>.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="box accent"><p><b>Re-activate this customer</b><br>Re-activation alone barely helps ({prob*100:.1f}% → {p2*100:.1f}%). Risk is driven by other factors — pair it with the actions on the right.</p></div>', unsafe_allow_html=True)
    elif num_products >= 3:
        p2 = predict_churn(credit_score, age, tenure, balance, 2, has_cr_card, is_active, salary, geography, gender)
        drop = (prob - p2) * 100
        if drop > 0.1:
            st.markdown(f'<div class="box accent"><p><b>Consolidate to 2 products</b><br>Churn moves <b style="color:{T["bad"]}">{prob*100:.1f}%</b> → <b style="color:{T["good"]}">{p2*100:.1f}%</b> (−{drop:.1f} pts).</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="box accent"><p><b>Consolidate to 2 products</b><br>Still high-risk at 2 products ({prob*100:.1f}% → {p2*100:.1f}%) — product count is not the main driver here.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="box accent"><p>This customer is active with a healthy product count. Toggle <b>Is Active Member</b> to <b>No</b> or raise <b>Number of Products</b> in the sidebar to simulate risk.</p></div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="sec">Behavioral Signals</div>', unsafe_allow_html=True)
    engagement = "High" if (is_active and num_products in [1, 2]) else "Low" if not is_active else "Moderate"
    tags = [
        f"Engagement: {engagement}",
        f"Product Density: {num_products/(tenure+1):.2f}/yr",
        f"Balance/Salary: {balance/(salary+1):.2f}",
        f"Active: {'Yes' if is_active else 'No'}",
        f"Card: {'Yes' if has_cr_card else 'No'}",
    ]
    st.markdown('<div class="box">' + "".join([f'<span class="tag">{t}</span>' for t in tags]) + '</div>', unsafe_allow_html=True)

# ============================================================
# KEY RISK FACTORS
# ============================================================
st.markdown('<div class="sec">Key Risk Factors</div>', unsafe_allow_html=True)
risks = []
if num_products >= 3:
    risks.append(("bad", "Product overload — 3+ products historically churn at 83–100%."))
if is_active == 0:
    risks.append(("bad", "Inactive member — disengaged customers churn at nearly double the rate."))
if geography == "Germany":
    risks.append(("warn", "German market — highest-churn region at 32.4%."))
if 40 <= age <= 55:
    risks.append(("warn", "Age 40–55 — peak-risk, high-value segment likely comparing rates."))
if balance > 100000:
    risks.append(("warn", "High balance — affluent customers are rate-sensitive and mobile."))
if not risks:
    risks.append(("good", "No major risk factors detected. Profile is stable."))
for k, msg in risks:
    st.markdown(f'<div class="pill {k}">{msg}</div>', unsafe_allow_html=True)

# ============================================================
# ACTION PLAN
# ============================================================
st.markdown('<div class="sec">Recommended Action Plan</div>', unsafe_allow_html=True)
if band == "high":
    st.markdown(f'<div class="box" style="border-left:4px solid {T["bad"]}"><h4>High Risk — Immediate Intervention</h4><ul><li>Assign a dedicated relationship manager within 48 hours.</li><li>Offer a personalized retention package — preferential rates or fee waivers.</li><li>If 3+ products: review for mis-selling and consolidate to the 2 most-used.</li><li>Schedule a direct call, not an automated email.</li><li>Priority justified: <b>€{money_at_risk:,.0f}</b> of balance is exposed.</li></ul></div>', unsafe_allow_html=True)
elif band == "medium":
    st.markdown(f'<div class="box" style="border-left:4px solid {T["warn"]}"><h4>Medium Risk — Proactive Engagement</h4><ul><li>Enroll in a targeted re-engagement campaign over the next 30 days.</li><li>Send a personalized product-fit review and loyalty offer.</li><li>For affluent customers: present competitive savings / FD options early.</li><li>Track engagement monthly; escalate if activity drops further.</li><li>Balance at risk to monitor: <b>€{money_at_risk:,.0f}</b>.</li></ul></div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="box" style="border-left:4px solid {T["good"]}"><h4>Low Risk — Nurture & Grow</h4><ul><li>Maintain standard relationship touchpoints.</li><li>Identify a thoughtful, needs-based upsell (stay within 2 core products).</li><li>Invite into loyalty or referral programs.</li><li>Continue quarterly health checks — no active intervention needed.</li></ul></div>', unsafe_allow_html=True)

# ============================================================
# MODEL PERFORMANCE
# ============================================================
st.markdown('<div class="sec">Model Performance</div>', unsafe_allow_html=True)
p1, p2c, p3, p4 = st.columns(4)
for col, (val, lab) in zip([p1, p2c, p3, p4],
                           [("Gradient Boosting", "Model"), ("87%", "Accuracy"),
                            ("60%", "Recall @ 0.35"), ("0.8675", "ROC-AUC")]):
    with col:
        st.markdown(f'<div class="card"><div class="v" style="font-size:{15 if lab=="Model" else 23}px">{val}</div><div class="l">{lab}</div></div>', unsafe_allow_html=True)

# ============================================================
# PERSONAL FOOTER
# ============================================================
st.markdown(f"""
<div class="ft">
    <div class="name">Bank Customer Churn Prediction · designed by <span>@nikchansocial</span></div>
    <div class="meta">Gradient Boosting · SHAP Explainability · Streamlit · Recall-tuned threshold</div>
</div>
""", unsafe_allow_html=True)
