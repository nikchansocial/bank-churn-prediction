
# ECB Banking Risk Intelligence Platform - Full Rebuild
import os
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="ECB Banking Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== DESIGN SYSTEM =====================
st.markdown("""
<style>
.stApp {background:#F8FAFC;}
.block-container {padding-top:1rem; max-width:1500px;}

.hero{
background:linear-gradient(135deg,#0F172A,#1E293B);
padding:28px;
border-radius:22px;
color:white;
margin-bottom:20px;
}

.kpi{
background:white;
padding:20px;
border-radius:18px;
box-shadow:0 8px 24px rgba(15,23,42,.08);
border:1px solid #E2E8F0;
height:120px;
}

.kpi-title{
font-size:12px;
letter-spacing:1px;
text-transform:uppercase;
color:#64748B;
}

.kpi-value{
font-size:34px;
font-weight:700;
color:#0F172A;
}

.panel{
background:white;
padding:18px;
border-radius:18px;
box-shadow:0 6px 18px rgba(15,23,42,.06);
border:1px solid #E2E8F0;
}

.alertbox{
background:#FEF2F2;
border-left:6px solid #DC2626;
padding:15px;
border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

# ===================== DATA =====================
@st.cache_data
def load_data():
    paths=["data/European_Bank.csv","European_Bank.csv"]
    df=None
    for p in paths:
        if os.path.exists(p):
            df=pd.read_csv(p)
            break
    if df is None:
        st.error("European_Bank.csv not found")
        st.stop()

    for c in ["Surname","CustomerId","Year"]:
        if c in df.columns:
            df.drop(columns=c,inplace=True)

    df["AgeGroup"]=pd.cut(
        df["Age"],
        [0,30,45,60,120],
        labels=["<30","30-45","46-60","60+"],
    )

    q75=df["Balance"].quantile(.75)
    df["IsHighValue"]=(df["Balance"]>=q75).astype(int)

    score=(df["IsActiveMember"].eq(0)*40 +
           (df["Age"]>45)*25 +
           (df["NumOfProducts"]>=3)*20 +
           (df["Balance"]>=q75)*15)

    df["RiskScore"]=score

    df["RiskLevel"]=pd.cut(
        score,
        [-1,25,50,75,100],
        labels=["Low","Medium","High","Critical"]
    )

    return df

df=load_data()

# ===================== SIDEBAR =====================
st.sidebar.title("⚙ Portfolio Controls")

geo=st.sidebar.multiselect(
    "Geography",
    sorted(df.Geography.unique()),
    default=list(df.Geography.unique())
)

risk=st.sidebar.multiselect(
    "Risk Level",
    list(df.RiskLevel.cat.categories),
    default=list(df.RiskLevel.cat.categories)
)

status=st.sidebar.selectbox(
    "Member Status",
    ["All","Active","Inactive"]
)

mask=(df.Geography.isin(geo)) & (df.RiskLevel.isin(risk))

if status=="Active":
    mask &= df.IsActiveMember==1
elif status=="Inactive":
    mask &= df.IsActiveMember==0

f=df[mask].copy()

# ===================== METRICS =====================
churn=f.Exited.mean()*100
lost=int(f.Exited.sum())
critical=(f.RiskLevel=="Critical").sum()
exposure=f.loc[f.Exited==1,"Balance"].sum()/1e6

country_rank=(f.groupby("Geography")["Exited"].mean()*100).sort_values(ascending=False)
top_country=country_rank.index[0]

# ===================== HERO =====================
st.markdown(f"""
<div class="hero">
<h1>🏦 ECB Banking Risk Intelligence Platform</h1>
<p style="font-size:18px;">
Executive Decision Support for Customer Attrition Risk
</p>
<hr>
<b>Portfolio Risk Status:</b> {"HIGH" if churn>20 else "MEDIUM"} |
<b>Balance Exposure:</b> £{exposure:.1f}M |
<b>Critical Customers:</b> {critical:,} |
<b>Highest Risk Region:</b> {top_country}
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class='alertbox'>
<b>Executive Risk Briefing</b><br>
{top_country} currently exhibits the highest customer churn concentration.
Customers aged 46–60 and inactive high-value clients represent the largest exposure pool.
Immediate retention activity is recommended.
</div>
""", unsafe_allow_html=True)

# ===================== KPI ROW =====================
cols=st.columns(5)
cards=[
("Churn Rate",f"{churn:.1f}%"),
("Customers Lost",f"{lost:,}"),
("Balance Exposure",f"£{exposure:.1f}M"),
("Critical Risk",f"{critical:,}"),
("Top Region",top_country)
]

for col,(t,v) in zip(cols,cards):
    with col:
        st.markdown(
            f"<div class='kpi'><div class='kpi-title'>{t}</div><div class='kpi-value'>{v}</div></div>",
            unsafe_allow_html=True
        )

# ===================== COMMAND CENTER =====================
left,right=st.columns([1,2])

with left:
    seg=f.RiskLevel.value_counts().reset_index()
    seg.columns=["Risk","Customers"]

    fig=px.pie(
        seg,
        names="Risk",
        values="Customers",
        hole=.6,
        title="Risk Segmentation"
    )
    st.plotly_chart(fig,use_container_width=True)

with right:
    geo_df=(f.groupby("Geography")["Exited"].mean()*100).reset_index()
    geo_df.columns=["Country","Churn Rate"]

    fig=px.bar(
        geo_df.sort_values("Churn Rate"),
        x="Churn Rate",
        y="Country",
        orientation="h",
        color="Churn Rate",
        title="Regional Risk Ranking"
    )
    st.plotly_chart(fig,use_container_width=True)

# ===================== TABS =====================
tab1,tab2,tab3,tab4,tab5=st.tabs([
"🌍 Regional Intelligence",
"👥 Customer Intelligence",
"💼 Product Intelligence",
"💰 Exposure Intelligence",
"⭐ Action Center"
])

with tab1:
    c1,c2=st.columns([1,2])

    with c1:
        st.subheader("Country Leaderboard")
        table=geo_df.sort_values("Churn Rate",ascending=False)
        st.dataframe(table,use_container_width=True)

    with c2:
        fig=px.bar(
            table,
            x="Country",
            y="Churn Rate",
            color="Churn Rate",
            title="Regional Churn Comparison"
        )
        st.plotly_chart(fig,use_container_width=True)

with tab2:
    c1,c2=st.columns(2)

    age=(f.groupby("AgeGroup")["Exited"].mean()*100).reset_index()

    with c1:
        fig=px.bar(
            age,
            x="AgeGroup",
            y="Exited",
            title="Age Risk Analysis"
        )
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        gender=(f.groupby("Gender")["Exited"].mean()*100).reset_index()

        fig=px.bar(
            gender,
            x="Gender",
            y="Exited",
            title="Gender Comparison"
        )
        st.plotly_chart(fig,use_container_width=True)

with tab3:
    prod=(f.groupby("NumOfProducts")["Exited"].mean()*100).reset_index()

    fig=px.line(
        prod,
        x="NumOfProducts",
        y="Exited",
        markers=True,
        title="Product Ownership Risk"
    )
    st.plotly_chart(fig,use_container_width=True)

    active=(f.groupby("IsActiveMember")["Exited"].mean()*100).reset_index()
    active["Status"]=active["IsActiveMember"].map({0:"Inactive",1:"Active"})

    fig=px.bar(
        active,
        x="Status",
        y="Exited",
        title="Activity Impact"
    )
    st.plotly_chart(fig,use_container_width=True)

with tab4:
    sample=f.sample(min(len(f),2500),random_state=42)

    fig=px.scatter(
        sample,
        x="Balance",
        y="EstimatedSalary",
        color="RiskLevel",
        size="CreditScore",
        hover_data=["Age","NumOfProducts"],
        title="Portfolio Exposure Map"
    )
    st.plotly_chart(fig,use_container_width=True)

    exposure_tbl=(
        f.groupby("RiskLevel")["Balance"]
        .sum()
        .reset_index()
    )

    fig=px.bar(
        exposure_tbl,
        x="RiskLevel",
        y="Balance",
        title="Balance Exposure by Risk Tier"
    )
    st.plotly_chart(fig,use_container_width=True)

with tab5:
    st.subheader("Recommended Actions")

    actions=[]

    if churn>20:
        actions.append("🔴 Launch enterprise retention programme immediately.")

    if critical>100:
        actions.append("🟠 Assign relationship managers to critical-risk customers.")

    actions.append(f"🌍 Prioritise intervention campaigns in {top_country}.")
    actions.append("👥 Target inactive customers aged 46–60.")
    actions.append("💰 Protect high-value balance holders.")
    actions.append("💼 Review customers holding 3+ products.")

    for a in actions:
        st.success(a)

# ===================== GOVERNANCE =====================
st.markdown("### Governance & Reporting")

g1,g2,g3,g4=st.columns(4)

g1.metric("Records",len(f))
g2.metric("Countries",f.Geography.nunique())
g3.metric("Version","5.0")
g4.metric("Generated",datetime.now().strftime("%H:%M"))

csv=f.to_csv(index=False).encode()

st.download_button(
    "📥 Export Portfolio Data",
    csv,
    "ecb_portfolio_export.csv",
    "text/csv"
)

st.caption("ECB Banking Risk Intelligence Platform • Enterprise Edition")
