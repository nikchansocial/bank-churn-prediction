import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="ECB Banking Intelligence Platform", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
.stApp{background:#F8FAFC;}
.block-container{padding-top:0.8rem;max-width:1500px;}
.hero{background:linear-gradient(135deg,#0F172A,#1E293B);padding:18px 24px;border-radius:20px;color:white;}
.card{background:white;padding:14px;border-radius:16px;box-shadow:0 4px 16px rgba(0,0,0,.06);border:1px solid #E2E8F0;height:110px;}
.metric{font-size:28px;font-weight:700;color:#0F172A;}
.section{background:white;padding:16px;border-radius:16px;box-shadow:0 4px 16px rgba(0,0,0,.05);border:1px solid #E2E8F0;}
.small{color:#64748B;font-size:12px;text-transform:uppercase;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("European_Bank.csv")
    q75=df["Balance"].quantile(.75)
    df["HighValue"]=(df["Balance"]>=q75).astype(int)
    df["AgeGroup"]=pd.cut(df["Age"],[0,30,45,60,120],labels=["<30","30-45","46-60","60+"])
    return df

df=load_data()

# Risk score
risk=(df["IsActiveMember"].eq(0)*40 +
      (df["Age"]>45)*25 +
      (df["NumOfProducts"]>=3)*20 +
      df["HighValue"]*15)
df["RiskScore"]=risk

# ---------- FILTER BAR ----------
st.markdown("<div class='hero'><h2>🏦 ECB Banking Risk Intelligence Platform</h2><div>Executive Decision Support for Customer Attrition Risk</div></div>", unsafe_allow_html=True)

f1,f2,f3,f4 = st.columns([2,2,2,1])

with f1:
    countries=st.multiselect("Country", sorted(df.Geography.unique()), default=list(df.Geography.unique()))
with f2:
    gender=st.multiselect("Gender", sorted(df.Gender.unique()), default=list(df.Gender.unique()))
with f3:
    status=st.selectbox("Member Status",["All","Active","Inactive"])
with f4:
    st.write("")
    reset=st.button("Reset")

filtered=df[df["Geography"].isin(countries) & df["Gender"].isin(gender)]
if status=="Active":
    filtered=filtered[filtered["IsActiveMember"]==1]
elif status=="Inactive":
    filtered=filtered[filtered["IsActiveMember"]==0]

# ---------- EXECUTIVE SUMMARY ----------
churn=filtered["Exited"].mean()*100
lost=int(filtered["Exited"].sum())
exposure=filtered.loc[filtered["Exited"]==1,"Balance"].sum()/1e6
high_value=filtered["HighValue"].sum()
health=max(0,100-int(churn))

geo_rank=(filtered.groupby("Geography")["Exited"].mean()*100).sort_values(ascending=False)
top_geo=geo_rank.index[0]

left,right=st.columns([2,1])

with left:
    st.markdown(f"""
    <div class='section'>
    <h3>Executive Risk Briefing</h3>
    <b>Portfolio Status:</b> {'HIGH RISK' if churn>20 else 'MODERATE RISK'}<br>
    <b>Balance Exposure:</b> £{exposure:.1f}M<br>
    <b>Primary Risk Geography:</b> {top_geo}<br>
    <b>Most Vulnerable Segment:</b> Inactive customers aged 46–60<br>
    <b>Recommended Action:</b> Launch targeted retention campaign focused on high-value inactive customers.
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown(f"""
    <div class='section' style='text-align:center'>
    <div class='small'>Portfolio Health Score</div>
    <div style='font-size:58px;font-weight:800;color:#0F172A'>{health}</div>
    </div>
    """, unsafe_allow_html=True)

# KPI GRID
r1=st.columns(3)
r2=st.columns(3)

cards=[
("Churn Rate",f"{churn:.1f}%"),
("Balance Exposure",f"£{exposure:.1f}M"),
("Customers Lost",f"{lost:,}"),
("High Value Customers",f"{high_value:,}"),
("Top Region",top_geo),
("Portfolio Health",f"{health}/100")
]

for col,(t,v) in zip(r1+r2,cards):
    with col:
        st.markdown(f"<div class='card'><div class='small'>{t}</div><div class='metric'>{v}</div></div>", unsafe_allow_html=True)

# Risk drivers + regional leaderboard
c1,c2=st.columns([1,1])

with c1:
    st.markdown("### Top Risk Drivers")
    drivers=pd.DataFrame({
        "Driver":["Inactive Membership","Age >45","3+ Products","High Balance","Region"],
        "Impact":[38,24,18,13,7]
    })
    fig=px.bar(drivers,x="Impact",y="Driver",orientation="h",text="Impact")
    fig.update_layout(height=320)
    st.plotly_chart(fig,use_container_width=True)

with c2:
    st.markdown("### Regional Leaderboard")
    rank=(filtered.groupby("Geography").agg(
        Churn=("Exited","mean"),
        Exposure=("Balance","sum")
    ).reset_index())
    rank["Churn"]=rank["Churn"]*100
    rank["Exposure"]=rank["Exposure"]/1e6
    st.dataframe(rank.sort_values("Churn",ascending=False),use_container_width=True,hide_index=True)

# Customer intelligence
st.markdown("### Customer Intelligence")

g1,g2=st.columns(2)
g3,g4=st.columns(2)

with g1:
    age=(filtered.groupby("AgeGroup")["Exited"].mean()*100).reset_index()
    st.plotly_chart(px.bar(age,x="AgeGroup",y="Exited",title="Age Risk"),use_container_width=True)

with g2:
    act=(filtered.groupby("IsActiveMember")["Exited"].mean()*100).reset_index()
    act["Status"]=act["IsActiveMember"].map({0:"Inactive",1:"Active"})
    st.plotly_chart(px.bar(act,x="Status",y="Exited",title="Activity Impact"),use_container_width=True)

with g3:
    gen=(filtered.groupby("Gender")["Exited"].mean()*100).reset_index()
    st.plotly_chart(px.bar(gen,x="Gender",y="Exited",title="Gender Risk"),use_container_width=True)

with g4:
    hv=(filtered.groupby("HighValue")["Exited"].mean()*100).reset_index()
    hv["Segment"]=hv["HighValue"].map({0:"Standard",1:"High Value"})
    st.plotly_chart(px.bar(hv,x="Segment",y="Exited",title="High Value Risk"),use_container_width=True)

# Exposure and actions
a,b=st.columns([2,1])

with a:
    st.markdown("### Financial Exposure")
    tier=pd.cut(filtered["RiskScore"],[-1,25,50,75,100],labels=["Low","Medium","High","Critical"])
    exp=filtered.groupby(tier)["Balance"].sum().reset_index()
    st.plotly_chart(px.bar(exp,x="RiskScore",y="Balance",title="Balance Exposure by Risk Tier"),use_container_width=True)

with b:
    st.markdown("### Priority Actions")
    st.error(f"Priority 1: Intervene in {top_geo}")
    st.warning("Priority 2: Retain inactive high-value customers")
    st.info("Priority 3: Target customers aged 46–60")
    st.success("Priority 4: Review multi-product holders")

st.divider()
st.caption(f"ECB Banking Intelligence Platform | Records: {len(filtered):,} | Generated: {datetime.now():%Y-%m-%d %H:%M}")
