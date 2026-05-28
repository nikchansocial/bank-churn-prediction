```python
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECB Customer Churn Intelligence",
    page_icon="🏦",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────────────────────
ECB_NAVY    = "#003366"
ECB_YELLOW  = "#FFCC00"
ECB_RED     = "#CC0000"
ECB_GREEN   = "#006633"
ECB_GRAY    = "#F0F4F8"
ECB_MID     = "#336699"

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
.stApp {{
    background-color: #f5f7fa;
}}

.metric-card {{
    background: white;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
    border-left: 5px solid {ECB_NAVY};
}}

.metric-title {{
    color: gray;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.metric-value {{
    color: {ECB_NAVY};
    font-size: 30px;
    font-weight: bold;
}}

.section-title {{
    color: {ECB_NAVY};
    font-weight: bold;
    margin-top: 10px;
}}

.insight-box {{
    background: white;
    padding: 18px;
    border-radius: 10px;
    border-left: 5px solid {ECB_RED};
    margin-bottom: 15px;
}}

footer {{
    visibility: hidden;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'European_Bank.csv')

    df = pd.read_csv(DATA_PATH)

    df.drop(columns=['Surname', 'CustomerId', 'Year'], inplace=True)

    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[0,29,45,60,120],
        labels=['<30','30-45','46-60','60+']
    )

    df['BalanceSegment'] = pd.cut(
        df['Balance'],
        bins=[-1,0,50000,300000],
        labels=['Zero','Low','High']
    )

    threshold = df['Balance'].quantile(0.75)

    df['IsHighValue'] = (
        df['Balance'] >= threshold
    ).astype(int)

    return df

df = load_data()

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:{ECB_NAVY};
padding:24px;
border-radius:12px;
margin-bottom:20px;'>

<h1 style='color:{ECB_YELLOW};
margin:0;'>
🏦 European Bank Customer Churn Intelligence Platform
</h1>

<p style='color:white;
margin-top:8px;
font-size:14px;'>

Executive analytics dashboard for identifying
high-risk banking customers and improving
retention strategies across Europe.

</p>

</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔍 Customer Filters")

geo = st.sidebar.multiselect(
    "Country",
    df['Geography'].unique(),
    default=list(df['Geography'].unique())
)

age = st.sidebar.multiselect(
    "Age Group",
    df['AgeGroup'].unique(),
    default=list(df['AgeGroup'].unique())
)

gender = st.sidebar.radio(
    "Gender",
    ["All", "Male", "Female"]
)

member = st.sidebar.radio(
    "Member Status",
    ["All", "Active", "Inactive"]
)

# ─────────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────────
mask = (
    df['Geography'].isin(geo)
    &
    df['AgeGroup'].isin(age)
)

if gender != "All":
    mask &= df['Gender'] == gender

if member != "All":
    mask &= df['IsActiveMember'] == (
        1 if member == "Active" else 0
    )

filtered = df[mask]

# ─────────────────────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────────────────────
churn_rate = filtered['Exited'].mean() * 100
total_churn = int(filtered['Exited'].sum())
high_value_rate = filtered[
    filtered['IsHighValue'] == 1
]['Exited'].mean() * 100

balance_risk = filtered[
    (filtered['IsHighValue']==1)
    &
    (filtered['Exited']==1)
]['Balance'].sum() / 1e6

# ─────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Overall Churn Rate</div>
        <div class='metric-value'>{churn_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Customers Lost</div>
        <div class='metric-value'>{total_churn:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>High Value Churn</div>
        <div class='metric-value'>{high_value_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Balance At Risk</div>
        <div class='metric-value'>£{balance_risk:.1f}M</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────────
st.markdown("## 📌 Executive Summary")

st.info("""
### Key Findings

1. Germany shows the highest customer churn risk.
2. Customers aged 46–60 are the most vulnerable segment.
3. High-value inactive customers represent major financial risk.
4. Customers holding 3+ products exhibit significantly higher churn rates.
5. Customer inactivity strongly correlates with attrition behavior.
""")

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 Geographic Risk",
    "👥 Demographics",
    "💼 Product Analysis",
    "💰 Financial Risk",
    "⭐ Retention Strategy"
])

# ─────────────────────────────────────────────────────────────
# TAB 1
# ─────────────────────────────────────────────────────────────
with tab1:

    st.markdown("### Geographic Churn Analysis")

    geo_data = (
        filtered
        .groupby('Geography')['Exited']
        .mean()
        .reset_index()
    )

    geo_data['Exited'] = (
        geo_data['Exited'] * 100
    ).round(1)

    fig1 = px.bar(
        geo_data,
        x='Geography',
        y='Exited',
        color='Exited',
        text_auto='.1f',
        color_continuous_scale='Reds',
        title='Churn Rate by Country'
    )

    fig1.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig1, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 2
# ─────────────────────────────────────────────────────────────
with tab2:

    st.markdown("### Demographic Risk Analysis")

    age_data = (
        filtered
        .groupby('AgeGroup')['Exited']
        .mean()
        .reset_index()
    )

    age_data['Exited'] = (
        age_data['Exited'] * 100
    ).round(1)

    fig2 = px.bar(
        age_data,
        x='AgeGroup',
        y='Exited',
        color='Exited',
        text_auto='.1f',
        color_continuous_scale='OrRd',
        title='Churn Rate by Age Group'
    )

    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 3
# ─────────────────────────────────────────────────────────────
with tab3:

    st.markdown("### Product & Engagement Analysis")

    product_data = (
        filtered
        .groupby('NumOfProducts')['Exited']
        .mean()
        .reset_index()
    )

    product_data['Exited'] = (
        product_data['Exited'] * 100
    ).round(1)

    fig3 = px.bar(
        product_data,
        x='NumOfProducts',
        y='Exited',
        color='Exited',
        text_auto='.1f',
        color_continuous_scale='Sunsetdark',
        title='Churn Rate by Product Count'
    )

    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 4
# ─────────────────────────────────────────────────────────────
with tab4:

    st.markdown("### Financial Risk Exposure")

    scatter = filtered.sample(
        min(2000, len(filtered)),
        random_state=42
    )

    scatter['Status'] = scatter['Exited'].map({
        0: 'Retained',
        1: 'Churned'
    })

    fig4 = px.scatter(
        scatter,
        x='Balance',
        y='EstimatedSalary',
        color='Status',
        opacity=0.6,
        title='Balance vs Salary Risk Distribution'
    )

    st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 5
# ─────────────────────────────────────────────────────────────
with tab5:

    st.markdown("## ⭐ Retention Recommendations")

    recommendations = [
        (
            "🔴 Germany Retention Crisis",
            "Launch region-specific retention campaigns in Germany."
        ),
        (
            "🟡 46–60 Age Group",
            "Provide retirement and wealth-planning products."
        ),
        (
            "⚠ Product Overload",
            "Customers with 3+ products show elevated churn."
        ),
        (
            "💰 High-Value Customer Risk",
            "Assign dedicated relationship managers."
        )
    ]

    for title, body in recommendations:

        st.markdown(f"""
        <div class='insight-box'>
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DOWNLOAD SECTION
# ─────────────────────────────────────────────────────────────
csv = filtered.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Filtered Customer Data",
    data=csv,
    file_name='filtered_customer_data.csv',
    mime='text/csv'
)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<p style='text-align:center;
font-size:12px;
color:gray;'>

Built by Nikhil Chandrakar |
Banking Analytics & Predictive Intelligence Portfolio

</p>
""", unsafe_allow_html=True)
```
